# Origin: cc (2026-10-02, from the BCNat session, at the user's request to make the model fast enough to sit inside a design loop)
# Purpose: evaluate the Potts energy of a SOFT sequence without materialising the V x V tables - 18.5x fewer multiply-accumulates
#          and no 40 MB tensor - so the term can be used as a per-step design loss. Adds a module; changes nothing existing.
"""The Potts energy of a SOFT sequence, without ever materialising the V x V tables.

`compute_potts_context` builds `etab [B, L, K, V, V]` and a caller then contracts it with a
sequence. For a design *loss* - which evaluates an energy and its gradient at every optimisation
step - that tensor is both the dominant cost and the dominant memory:

    etab          = head_in [L, K, H] @ W [H, V^2]        L*K*H*V^2 MACs, 4*L*K*V^2 bytes
    contraction   = sum_ikab P_i(a) etab[i,k,a,b] P_j(b)

but the contraction only ever asks for a quadratic form, so the head weights can be contracted
with the sequence FIRST, through V rather than V^2:

    U[i,b,c]  = sum_a P_i(a)  W3[a,b,c]                   L*V*V*H
    T[i,k,c]  = sum_b U[i,b,c] P_{E_idx[i,k]}(b)          L*K*V*H
    Q[i,k]    = sum_c h_E[i,k,c] T[i,k,c]  + bias term    L*K*H

At L=229, K=48, H=128, V=30 that is 1.27 G MAC and a 40 MB tensor against 0.069 G MAC and no
tensor: **18.5x fewer multiply-accumulates, and the tables never exist**. The saving is exactly
K*V^2 / (V^2 + K*V) and grows with the neighbour count, not with the chain length.

Everything `compute_potts_context` does to the tables is linear in them, so each step survives the
reordering exactly, and this module reproduces them rather than approximating:

* **masking** - `etab *= E_mask` becomes `Q *= E_mask`.
* **the self edge** (`k = 0`, `E_idx[i,0] = i`) carries the single-body field. With
  `field_source="self_edge"` upstream keeps only its diagonal (`etab[:, :, 0] * eye`), and because
  the self edge's neighbour is the residue itself, its quadratic form collapses to
  `sum_a P_i(a)^2 d_i(a)` with `d` the diagonal of that one table - V numbers per row, not V^2.
* **the reciprocal merge** - upstream replaces `etab[i,k]` by `0.5 * (etab[i,k] + etab[j,r]^T)`
  where `j = E_idx[i,k]` and `r` is the slot of the reverse edge. Since `E_idx[j,r] = i`, the
  quadratic form of that transpose against `(P_i, P_j)` is the quadratic form of `etab[j,r]`
  against `(P_j, P_i)` - which is `Q[j,r]`. The merge is therefore a **gather on the scalars**,
  `0.5 * (Q[i,k] + Q[j,r])`, with no transpose and no table.

`tests/test_PPJAX_contract_cc.py` asserts agreement with the materialised path on the reference
context, for a one-hot and for a random soft sequence.

Not covered here: the decoder, and `field_source="node"` (the shipped v6 checkpoint is
`self_edge`; the node variant raises rather than silently scoring something else).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import jax.numpy as jnp

from ppjax.layers import gather_nodes

__all__ = ["potts_edge_energies", "soft_potts_energy"]


def _edge_mask(E_idx: jnp.ndarray, residue_mask: jnp.ndarray) -> jnp.ndarray:
    """`compute_potts_context`'s `E_mask`: both endpoints of the edge are real residues."""
    real = residue_mask.astype(bool)
    neighbour_real = gather_nodes(real[..., None].astype(jnp.int32), E_idx)[..., 0] > 0
    return real[:, :, None] & neighbour_real


def _reverse_slot(E_idx: jnp.ndarray):
    """For every edge (i, k), the slot r with `E_idx[j, r] = i`, and whether one exists."""
    batch = jnp.arange(E_idx.shape[0])[:, None, None]
    source = jnp.arange(E_idx.shape[1])[None, :, None]
    neighbour_lists = E_idx[batch, E_idx, :]                       # [B, L, K, K]
    matches = neighbour_lists == source[..., None]
    return jnp.argmax(matches.astype(jnp.int32), axis=-1), jnp.any(matches, axis=-1)


def potts_edge_energies(params: Dict[str, Any], cfg, h_V, h_E, E_idx, residue_mask,
                        sequence: jnp.ndarray) -> jnp.ndarray:
    """Per-edge Potts energies `Q [B, L, K]` of a soft sequence, with no V x V table materialised.

    `sequence` is `[B, L, V]` and need not be one-hot; at a one-hot sequence `Q.sum()` equals the
    contraction of `compute_potts_context`'s tables exactly (up to float32 reassociation).
    """
    if getattr(cfg, "etab_source", "edge") != "edge":
        raise NotImplementedError(
            f"ppjax.contract covers etab_source='edge' (the shipped v6 checkpoint); got "
            f"{cfg.etab_source!r}. The node_edge_node head concatenates h_V, so factor that first.")
    if getattr(cfg, "field_source", "self_edge") != "self_edge":
        raise NotImplementedError(
            f"ppjax.contract covers field_source='self_edge'; got {cfg.field_source!r}.")

    vocab = int(cfg.vocab_size)
    weight = params["etab_out"]["weight"]                          # [V*V, H]
    hidden = weight.shape[-1]
    W3 = weight.reshape(vocab, vocab, hidden)
    bias = params["etab_out"].get("bias")
    B2 = None if bias is None else bias.reshape(vocab, vocab)

    P = sequence.astype(h_E.dtype)
    batch = jnp.arange(P.shape[0])[:, None, None]
    neighbour_P = P[batch, E_idx]                                   # [B, L, K, V]

    #   Q[i,k] = sum_c h_E[i,k,c] * (P_i^T W3[:, :, c] P_j)  +  P_i^T B2 P_j
    U = jnp.einsum("nia,avc->nivc", P, W3)                          # [B, L, V, H]
    T = jnp.einsum("nivc,nikv->nikc", U, neighbour_P)               # [B, L, K, H]
    Q = jnp.einsum("nikc,nikc->nik", h_E, T)
    if B2 is not None:
        Q = Q + jnp.einsum("nia,av,nikv->nik", P, B2, neighbour_P)

    E_mask = _edge_mask(E_idx, residue_mask)
    Q = Q * E_mask.astype(Q.dtype)

    # the self edge keeps only its diagonal, and its neighbour is the residue itself
    diagonal = jnp.einsum("aac,nic->nia", W3, h_E[:, :, 0, :])      # [B, L, V]
    if B2 is not None:
        diagonal = diagonal + jnp.diagonal(B2)[None, None, :]
    self_energy = jnp.einsum("nia,nia,nia->ni", P, diagonal, P) * E_mask[:, :, 0].astype(Q.dtype)
    Q = Q.at[:, :, 0].set(self_energy)

    # the reciprocal merge, as a gather on the scalars
    reverse_k, has_reverse = _reverse_slot(E_idx)
    Q_reverse = Q[batch, E_idx, reverse_k]
    reverse_mask = E_mask[batch, E_idx, reverse_k]
    valid = has_reverse & E_mask & reverse_mask
    return jnp.where(valid, 0.5 * (Q + Q_reverse), Q)


def soft_potts_energy(params: Dict[str, Any], cfg, h_V, h_E, E_idx, residue_mask,
                      sequence: jnp.ndarray, rows: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """`sum over edges` of :func:`potts_edge_energies`, optionally only over `rows`.

    `rows` is a `[B, L]` weight (a 0/1 mask, or soft) selecting which residues' edges are summed -
    a binder loss wants its own designed rows, not the target's internal energy. With `rows=None`
    every row counts, which is the whole-complex Hamiltonian."""
    Q = potts_edge_energies(params, cfg, h_V, h_E, E_idx, residue_mask, sequence)
    if rows is None:
        return Q.sum()
    return (Q * rows.astype(Q.dtype)[:, :, None]).sum()
