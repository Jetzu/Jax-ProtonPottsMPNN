# Origin: cc (Claude Code, 2026-10-02) - ported from
#         ~/ProtonPottsMPNN/foundry/models/mpnn/src/mpnn/model/layers/graph_embeddings.py
#         (ProteinFeatures / PottsProteinFeatures) and positional_encoding.py (upstream read-only).
# Purpose: structure -> graph features (kNN graph, atomwise RBF edges, chain-aware positional
#          embedding) in JAX, numerically matching the torch featuriser.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ppjax.layers import gather_edges, gather_nodes
from ppjax.nn import Params, layer_norm, linear, one_hot
from ppjax.tokens import TokenEncoding

BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")
REPRESENTATIVE_ATOM_NAMES = ("CA",)
# (center, atom_1, atom_2) + the learned virtual-CB weights, verbatim from ProteinFeatures.
DATA_TO_CALCULATE_VIRTUAL_ATOMS = (
    (("CA", "N", "C"), (0.58273431, -0.56802827, -0.54067466)),
)
MAX_RELATIVE_FEATURE = 32


@dataclass(frozen=True)
class FeatureConfig:
    num_positional_embeddings: int = 16
    min_rbf_mean: float = 2.0
    max_rbf_mean: float = 22.0
    num_rbf: int = 16
    num_neighbors: int = 48

    @property
    def num_positional_features(self) -> int:
        return 2 * MAX_RELATIVE_FEATURE + 1 + 1


def construct_X_atoms(X, X_m, S, atom_index_table: jnp.ndarray):
    """``ProteinFeatures.construct_X_atoms``: per-residue gather of named atoms.

    ``atom_index_table`` is the ``[n_tokens, n_names]`` lookup built from the vocabulary, so the
    gather follows the token at each position exactly as the torch version does.
    """
    atom_indices = jnp.take(atom_index_table, S, axis=0)            # [B, L, n_names]
    X_atoms = jnp.take_along_axis(X, atom_indices[..., None], axis=2)
    X_m_atoms = jnp.take_along_axis(X_m, atom_indices, axis=2)
    return X_atoms, X_m_atoms


def construct_X_virtual_atoms(X, X_m, S, enc: TokenEncoding):
    X_v, X_m_v = [], []
    for (names, (w_n, w_b1, w_b2)) in DATA_TO_CALCULATE_VIRTUAL_ATOMS:
        table = jnp.asarray(enc.atom_indices(names))
        X_atoms, X_m_atoms = construct_X_atoms(X, X_m, S, table)
        c, a1, a2 = X_atoms[:, :, 0], X_atoms[:, :, 1], X_atoms[:, :, 2]
        bond_1, bond_2 = a1 - c, a2 - c
        normal = jnp.cross(bond_1, bond_2, axis=-1)
        X_v.append(w_n * normal + w_b1 * bond_1 + w_b2 * bond_2 + c)
        X_m_v.append(X_m_atoms[:, :, 0] * X_m_atoms[:, :, 1] * X_m_atoms[:, :, 2])
    return jnp.stack(X_v, axis=2), jnp.stack(X_m_v, axis=2)


def knn_graph(X_rep_atoms, X_m_rep_atoms, residue_mask, num_neighbors: int, eps: float = 1e-6):
    """``compute_representative_atom_pairwise_distances``: masked CA-CA distances -> top-K nearest.

    Masked pairs are pushed to that row's maximum distance before the selection, exactly as
    upstream, so invalid residues sort last without changing the valid ordering.
    """
    L = X_rep_atoms.shape[1]
    rm = residue_mask.astype(jnp.float32)
    mask_2D = (rm[:, None, :] * rm[:, :, None]) > 0
    X_c = jnp.sum(X_rep_atoms * X_m_rep_atoms[:, :, :, None], axis=2)      # [B, L, 3]
    dX = X_c[:, None, :, :] - X_c[:, :, None, :]
    D = mask_2D.astype(jnp.float32) * jnp.sqrt(jnp.sum(dX ** 2, axis=3) + eps)
    D_max = jnp.max(D, axis=-1, keepdims=True)
    D_adjust = D + (~mask_2D).astype(jnp.float32) * D_max
    k = min(num_neighbors, L)
    neg, E_idx = jax.lax.top_k(-D_adjust, k)
    # the k-th / (k+1)-th distance pair: where torch.topk and jax.lax.top_k could keep different
    # neighbours if they are exactly equal (see ppjax.compat.check_knn_tie).
    if k < L:
        neg_b, _ = jax.lax.top_k(-D_adjust, k + 1)
        boundary = jnp.stack([-neg_b[..., k - 1], -neg_b[..., k]], axis=-1)
    else:
        boundary = jnp.zeros(D_adjust.shape[:-1] + (2,), D_adjust.dtype)
    return -neg, E_idx, boundary


def rbf_embedding(D: jnp.ndarray, cfg: FeatureConfig) -> jnp.ndarray:
    mus = jnp.linspace(cfg.min_rbf_mean, cfg.max_rbf_mean, cfg.num_rbf, dtype=jnp.float32)
    sigma = (cfg.max_rbf_mean - cfg.min_rbf_mean) / cfg.num_rbf
    return jnp.exp(-(((D[..., None] - mus) / sigma) ** 2))


def pairwise_residue_rbf(X, E_idx, X_m, cfg: FeatureConfig, eps: float = 1e-6):
    """Atom-by-atom RBF encoding of every (residue, neighbour) pair -> [B, L, K, A*A*num_rbf]."""
    B, L, A, _ = X.shape
    K = E_idx.shape[2]
    X_g = gather_nodes(X.reshape(B, L, A * 3), E_idx).reshape(B, L, K, A, 3)
    D = jnp.sqrt(jnp.sum((X[:, :, None, :, None, :] - X_g[:, :, :, None, :, :]) ** 2, axis=-1) + eps)
    RBF = rbf_embedding(D, cfg)
    X_m_g = gather_nodes(X_m, E_idx)
    RBF = RBF * X_m[:, :, None, :, None, None] * X_m_g[:, :, :, None, :, None]
    return RBF.reshape(B, L, K, -1)


def positional_encoding(p: Params, R_idx, E_idx, chain_labels, cfg: FeatureConfig):
    """``PositionalEncodings``: clipped, chain-aware relative-position one-hot through a Linear."""
    offset = (R_idx[:, :, None] - R_idx[:, None, :]).astype(jnp.int32)
    offset_g = gather_edges(offset[..., None], E_idx)[..., 0]
    same_chain = (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
    same_chain_g = gather_edges(same_chain[..., None].astype(jnp.int32), E_idx)[..., 0] > 0

    clipped = jnp.clip(offset_g + MAX_RELATIVE_FEATURE, 0, 2 * MAX_RELATIVE_FEATURE)
    chain_aware = jnp.where(same_chain_g, clipped, 2 * MAX_RELATIVE_FEATURE + 1)
    oh = one_hot(chain_aware, cfg.num_positional_features)
    return linear(p["embed_positional_features"], oh)


def featurize(params: Params, enc: TokenEncoding, cfg: FeatureConfig,
              X, X_m, S, R_idx, chain_labels, residue_mask) -> Dict[str, jnp.ndarray]:
    """``ProteinFeatures.forward``. Coordinate noise is applied by the caller
    (``ppjax.model.apply_structure_noise``) so that the RNG draw happens in the same place in the
    stream as upstream's in-featuriser ``noise_structure``."""
    bb_table = jnp.asarray(enc.atom_indices(BACKBONE_ATOM_NAMES))
    rep_table = jnp.asarray(enc.atom_indices(REPRESENTATIVE_ATOM_NAMES))

    X_backbone, X_m_backbone = construct_X_atoms(X, X_m, S, bb_table)
    X_virtual, X_m_virtual = construct_X_virtual_atoms(X, X_m, S, enc)
    X_rep, X_m_rep = construct_X_atoms(X, X_m, S, rep_table)

    _, E_idx, knn_boundary = knn_graph(X_rep, X_m_rep, residue_mask, cfg.num_neighbors)

    X_all = jnp.concatenate((X_backbone, X_virtual), axis=-2)
    X_m_all = jnp.concatenate((X_m_backbone, X_m_virtual), axis=-1)

    RBF_all = pairwise_residue_rbf(X_all, E_idx, X_m_all, cfg)
    pos = positional_encoding(params["positional_embedding"], R_idx, E_idx, chain_labels, cfg)

    E_raw = jnp.concatenate((pos, RBF_all), axis=-1)
    E = layer_norm(params["edge_norm"], linear(params["edge_embedding"], E_raw))
    return {"E_idx": E_idx, "E": E, "E_raw": E_raw, "knn_boundary": knn_boundary}
