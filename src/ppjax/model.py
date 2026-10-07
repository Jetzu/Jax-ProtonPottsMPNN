# Origin: cc (Claude Code, 2026-10-02) - ported from
#         ~/ProtonPottsMPNN/foundry/models/mpnn/src/mpnn/model/mpnn.py (ProteinMPNN.encode,
#         sample_and_construct_masks) and model/pottsmpnn.py (PottsMPNN.compute_potts_context and
#         the Potts energy helpers). Upstream read-only.
# Purpose: the generation-path forward in JAX: masks -> graph featurisation -> encoder ->
#          Potts coupling tables (fields on the self-edge diagonal, reciprocal-edge merge).

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ppjax.features import FeatureConfig, featurize
from ppjax.layers import enc_layer, gather_nodes
from ppjax.nn import Params, embedding, linear
from ppjax.tokens import TokenEncoding

FIELD_SOURCES = ("self_edge", "node")
ETAB_SOURCES = ("edge", "node_edge_node")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_dim: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    field_source: str = "self_edge"
    etab_source: str = "edge"
    features: FeatureConfig = FeatureConfig()

    def __post_init__(self):
        if self.field_source not in FIELD_SOURCES:
            raise ValueError(f"Invalid field_source: {self.field_source!r}")
        if self.etab_source not in ETAB_SOURCES:
            raise ValueError(f"Invalid etab_source: {self.etab_source!r}")


def construct_masks(S, residue_mask, designed_residue_mask, unknown_indices) -> Dict[str, jnp.ndarray]:
    """``ProteinMPNN.sample_and_construct_masks``, without the in-place mutation."""
    residue_mask = residue_mask.astype(bool)
    unk = jnp.asarray(np.asarray(unknown_indices, dtype=np.int64))
    known = ~jnp.any(S[..., None] == unk, axis=-1)
    if designed_residue_mask is None:
        designed = residue_mask
    else:
        designed = designed_residue_mask.astype(bool)
    return {
        "residue_mask": residue_mask,
        "known_residue_mask": known,
        "designed_residue_mask": designed,
        "mask_for_loss": residue_mask & known & designed,
    }


def encode(params: Params, cfg: ModelConfig, residue_mask, E, E_idx):
    """``ProteinMPNN.encode``: zero node states, embedded edges, then the encoder stack."""
    B, L = E.shape[0], E.shape[1]
    h_V = jnp.zeros((B, L, cfg.hidden_dim), dtype=E.dtype)
    h_E = linear(params["W_e"], E)
    rm = residue_mask.astype(E.dtype)
    mask_E = rm[:, :, None] * gather_nodes(rm[..., None], E_idx)[..., 0]
    for layer in params["encoder_layers"]:
        h_V, h_E = enc_layer(layer, h_V, h_E, E_idx, mask_V=rm, mask_E=mask_E)
    return h_V, h_E


def get_pottshead_input(h_V, h_E, E_idx, etab_source: str):
    if etab_source == "edge":
        return h_E
    B, L, K, H = h_E.shape
    return jnp.concatenate([
        jnp.broadcast_to(h_V[:, :, None, :], (B, L, K, H)),
        h_E,
        gather_nodes(h_V, E_idx),
    ], axis=-1)


def compute_potts_context(params: Params, cfg: ModelConfig, h_V, h_E, E_idx, residue_mask):
    """``PottsMPNN.compute_potts_context``: pair tables, self-edge field, reciprocal merge."""
    B, L, K, _ = h_E.shape
    V = cfg.vocab_size
    rm = residue_mask.astype(bool)
    E_mask = rm[:, :, None] & (gather_nodes(rm[..., None].astype(jnp.int32), E_idx)[..., 0] > 0)

    head_in = get_pottshead_input(h_V, h_E, E_idx, cfg.etab_source)
    etab = linear(params["etab_out"], head_in)
    etab = etab * E_mask[..., None].astype(etab.dtype)
    etab = etab.reshape(B, L, K, V, V)

    # The self-edge (k == 0, E_idx[i,0] == i) carries the single-body field on its diagonal.
    if cfg.field_source == "node":
        field = linear(params["node_field"], h_V)
        field = field * E_mask[:, :, 0:1].astype(field.dtype)
        slot0 = jnp.zeros((B, L, V, V), etab.dtype).at[..., jnp.arange(V), jnp.arange(V)].set(field)
    else:
        slot0 = etab[:, :, 0] * jnp.eye(V, dtype=etab.dtype)
    etab = etab.at[:, :, 0].set(slot0)

    batch_idx = jnp.arange(B)[:, None, None]
    src_idx = jnp.arange(L)[None, :, None]
    neighbor_lists = etab_gather_idx = E_idx[batch_idx, E_idx, :]          # [B, L, K, K]
    reverse_matches = neighbor_lists == src_idx[..., None]
    has_reverse = jnp.any(reverse_matches, axis=-1)
    reverse_k = jnp.argmax(reverse_matches.astype(jnp.int32), axis=-1)

    reverse_etab = etab[batch_idx, E_idx, reverse_k]
    merged = 0.5 * (etab + jnp.swapaxes(reverse_etab, -1, -2))
    reverse_mask = E_mask[batch_idx, E_idx, reverse_k]
    valid_merge = has_reverse & E_mask & reverse_mask
    etab = jnp.where(valid_merge[..., None, None], merged, etab)
    return etab, E_idx, E_mask


@partial(jax.jit, static_argnums=(1, 2, 3))
def _forward_potts(params, enc: TokenEncoding, cfg: ModelConfig, unknown_indices: Tuple[int, ...],
                   X, X_m, S, R_idx, chain_labels, residue_mask, designed_residue_mask):
    masks = construct_masks(S, residue_mask, designed_residue_mask, unknown_indices)
    gf = featurize(params["graph_featurization_module"], enc, cfg.features,
                   X, X_m, S, R_idx, chain_labels, masks["residue_mask"])
    h_V, h_E = encode(params, cfg, masks["residue_mask"], gf["E"], gf["E_idx"])
    etab, E_idx, E_mask = compute_potts_context(params, cfg, h_V, h_E, gf["E_idx"],
                                                masks["residue_mask"])
    return {"etab_out": etab, "E_idx": E_idx, "E_mask": E_mask, "E": gf["E"],
            "E_raw": gf["E_raw"], "knn_boundary": gf["knn_boundary"],
            "h_V": h_V, "h_E": h_E, **masks}


def apply_structure_noise(X, structure_noise: float, gen=None):
    """``ProteinFeatures.noise_structure``: ``X + structure_noise * randn_like(X)``.

    torch applies this INSIDE the featuriser, so the normals are consumed before the decoder's
    decoding-order draw; the port keeps that order. The ``randn`` VALUES inherit the <=1e-6 libm
    difference documented in :mod:`ppjax.rng`, so a noised run is stream-aligned but not bitwise
    identical - at the default ``structure_noise=0.0`` nothing is drawn and the question is moot.
    """
    from ppjax import rng as trng
    if not structure_noise:
        return X
    gen = gen or trng.default_generator()
    X = np.asarray(X, dtype=np.float32)
    noise = gen.normal_float32(int(X.size)).reshape(X.shape)
    return X + np.float32(structure_noise) * noise


def forward_potts(params, enc, cfg, unknown_indices, features: Dict[str, Any]):
    """Encoder-side forward returning the Potts tables plus every intermediate the tests check."""
    if features.get("symmetry_equivalence_group") is not None:
        raise NotImplementedError(
            "symmetry_equivalence_group is not ported - see ppjax.decoder.sample_decoding_order.")
    X = apply_structure_noise(features["X"], float(features.get("structure_noise") or 0.0))
    return _forward_potts(
        params, enc, cfg, tuple(int(i) for i in unknown_indices),
        jnp.asarray(X, jnp.float32),
        jnp.asarray(features["X_m"], jnp.float32),
        jnp.asarray(features["S"], jnp.int32),
        jnp.asarray(features["R_idx"], jnp.int32),
        jnp.asarray(features["chain_labels"], jnp.int32),
        jnp.asarray(features["residue_mask"]).astype(bool),
        None if features.get("designed_residue_mask") is None
        else jnp.asarray(features["designed_residue_mask"]).astype(bool),
    )
