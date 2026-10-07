# Origin: cc (Claude Code, 2026-10-02) - ported from mpnn.model.pottsmpnn (calc_potts_eners,
#         potts_candidate_energies, _directed_pair_edges, _incoming_adjacency) and
#         mpnn.inference_engines.potts_mpnn_ph._PottsScorer. Upstream read-only.
# Purpose: slice the frozen Potts tables in JAX: Hamiltonian, per-position conditional energies,
#          and the block decomposition the block-descent optimiser enumerates over.

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def directed_pair_edges(eidx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten the non-self directed edges (neighbour slots k >= 1), as ``_directed_pair_edges``."""
    L, K = eidx.shape
    src = np.repeat(np.arange(L), K - 1)
    slot = np.tile(np.arange(1, K), L)
    tgt = eidx[:, 1:].reshape(-1)
    keep = tgt != src
    return src[keep], slot[keep], tgt[keep]


def incoming_adjacency(eidx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Padded graph transpose: for each position the ``(m, slot)`` pairs with ``E_idx[m, slot] == p``.

    The kNN graph is ~16% non-reciprocal, so a position's identity also enters H through these
    incoming edges; the conditional energy must add them to match ``calc_potts_eners``.
    Zero-padded to the maximum in-degree, with a validity mask (upstream keeps ragged lists; the
    padded entries are multiplied by an exact 0.0, so the sums are unchanged).
    """
    src, slot, tgt = directed_pair_edges(eidx)
    L = eidx.shape[0]
    indeg = np.bincount(tgt, minlength=L)
    D = int(indeg.max()) if indeg.size else 0
    in_src = np.zeros((L, D), dtype=np.int64)
    in_slot = np.zeros((L, D), dtype=np.int64)
    in_mask = np.zeros((L, D), dtype=bool)
    order = np.argsort(tgt, kind="stable")
    tgt_s, src_s, slot_s = tgt[order], src[order], slot[order]
    group_start = np.zeros(L, dtype=np.int64)
    group_start[1:] = np.cumsum(indeg)[:-1]
    within = np.arange(tgt_s.size) - group_start[tgt_s]
    in_src[tgt_s, within] = src_s
    in_slot[tgt_s, within] = slot_s
    in_mask[tgt_s, within] = True
    return in_src, in_slot, in_mask


@partial(jax.jit, static_argnums=())
def _H_of(etab, eidx, S):
    L, K = eidx.shape
    E_aa_j = S[eidx]                                     # [L, K]
    s_i = jnp.broadcast_to(S[:, None], (L, K))
    vals = etab[jnp.arange(L)[:, None], jnp.arange(K)[None, :], s_i, E_aa_j]
    return jnp.sum(vals)


@jax.jit
def _cond_energy(etab, eidx, self_term, in_src, in_slot, in_mask, S):
    """``potts_candidate_energies`` for one sequence: self field + outgoing + incoming."""
    L, K, V, _ = etab.shape
    nbr_idx = eidx[:, 1:]
    pe = etab[:, 1:]
    s_nbr = S[nbr_idx]                                   # [L, K-1]
    outgoing = jnp.sum(jnp.take_along_axis(
        pe, s_nbr[:, :, None, None].repeat(V, axis=2), axis=-1)[..., 0], axis=1)
    inc = etab[in_src, in_slot, S[in_src], :] * in_mask[..., None]
    return self_term + outgoing + jnp.sum(inc, axis=1)


@jax.jit
def _cond_energy_at(etab, nbr_idx, pe, self_term, in_src, in_slot, in_mask, S, i):
    """Position ``i``'s [V] row only - identical value to ``_cond_energy(...)[i]``."""
    V = etab.shape[-1]
    out = self_term[i]
    nb = nbr_idx[i]
    s_nbr = S[nb]
    pei = pe[i]
    out = out + jnp.sum(jnp.take_along_axis(
        pei, s_nbr[:, None, None].repeat(V, axis=1), axis=-1)[..., 0], axis=0)
    msrc, mslot, mmask = in_src[i], in_slot[i], in_mask[i]
    out = out + jnp.sum(etab[msrc, mslot, S[msrc], :] * mmask[:, None], axis=0)
    return out


@dataclass
class PottsScorer:
    """The frozen Potts tables plus the index bookkeeping every scoring path reuses."""

    etab: jnp.ndarray          # [L, K, V, V]
    eidx: jnp.ndarray          # [L, K] int32
    self_term: jnp.ndarray     # [L, V]
    nbr_idx: jnp.ndarray       # [L, K-1]
    pe: jnp.ndarray            # [L, K-1, V, V]
    in_src: jnp.ndarray        # [L, D]
    in_slot: jnp.ndarray
    in_mask: jnp.ndarray
    eidx_np: np.ndarray
    L: int
    K: int
    V: int
    has_self_loops: bool = False

    @classmethod
    def from_tables(cls, etab_out, E_idx) -> "PottsScorer":
        etab = jnp.asarray(etab_out)
        if etab.ndim == 5:
            etab = etab[0]
        eidx_np = np.asarray(E_idx)
        if eidx_np.ndim == 3:
            eidx_np = eidx_np[0]
        eidx_np = eidx_np.astype(np.int64)
        L, K, V, _ = etab.shape
        in_src, in_slot, in_mask = incoming_adjacency(eidx_np)
        # Whether any position is its OWN neighbour at a slot k >= 1. When that happens,
        # cond_energy_at(S, i) depends on S[i] and the one-row shortcut in
        # engine._selective_energy_sum stops being an identity; the engine falls back to the
        # literal per-token re-evaluation. Measured 0 for every structure in tests/data.
        self_loops = bool((eidx_np[:, 1:] == np.arange(L)[:, None]).any())
        return cls(
            etab=etab, eidx=jnp.asarray(eidx_np, jnp.int32),
            self_term=jnp.diagonal(etab[:, 0], axis1=-2, axis2=-1),
            nbr_idx=jnp.asarray(eidx_np[:, 1:], jnp.int32), pe=etab[:, 1:],
            in_src=jnp.asarray(in_src, jnp.int32), in_slot=jnp.asarray(in_slot, jnp.int32),
            in_mask=jnp.asarray(in_mask), eidx_np=eidx_np, L=L, K=K, V=V,
            has_self_loops=self_loops,
        )

    # --- scoring -------------------------------------------------------------
    def H_of(self, S) -> float:
        return float(_H_of(self.etab, self.eidx, jnp.asarray(S, jnp.int32)))

    def cond_energy(self, S) -> jnp.ndarray:
        return _cond_energy(self.etab, self.eidx, self.self_term,
                            self.in_src, self.in_slot, self.in_mask, jnp.asarray(S, jnp.int32))

    def cond_energy_at(self, S, i) -> jnp.ndarray:
        return _cond_energy_at(self.etab, self.nbr_idx, self.pe, self.self_term,
                               self.in_src, self.in_slot, self.in_mask,
                               jnp.asarray(S, jnp.int32), jnp.asarray(i, jnp.int32))

    def rows_at(self, S, positions) -> jnp.ndarray:
        """[n, V] conditional rows at several positions, one vmapped kernel."""
        return _rows_at(self.etab, self.nbr_idx, self.pe, self.self_term,
                        self.in_src, self.in_slot, self.in_mask,
                        jnp.asarray(S, jnp.int32), jnp.asarray(positions, jnp.int32))

    def rows_for_candidates(self, S, p, centers) -> jnp.ndarray:
        """[V_cand, n_centers, V] - the centres' conditional rows with position ``p`` set to each
        candidate amino acid in turn. One vmapped kernel instead of V*n_centers dispatches."""
        return _rows_for_candidates(self.etab, self.nbr_idx, self.pe, self.self_term,
                                    self.in_src, self.in_slot, self.in_mask,
                                    jnp.asarray(S, jnp.int32), jnp.asarray(p, jnp.int32),
                                    jnp.asarray(centers, jnp.int32), self.V)

    def reweighted(self, w_self: float, w_pair: float) -> "PottsScorer":
        """H = w_self*sum h_i + w_pair*sum J_ij. Slot 0 is the pure field, slots >= 1 the pure
        couplings, so the two scale independently."""
        etab = self.etab.at[:, 0].multiply(float(w_self)).at[:, 1:].multiply(float(w_pair))
        return PottsScorer(  # noqa: E501
            etab=etab, eidx=self.eidx,
            self_term=jnp.diagonal(etab[:, 0], axis1=-2, axis2=-1),
            nbr_idx=self.nbr_idx, pe=etab[:, 1:],
            in_src=self.in_src, in_slot=self.in_slot, in_mask=self.in_mask,
            eidx_np=self.eidx_np, L=self.L, K=self.K, V=self.V,
            has_self_loops=self.has_self_loops)

    # --- block decomposition -------------------------------------------------
    def block_potentials(self, S, block: Sequence[int]):
        """``_PottsScorer.block_stability_potentials`` -> (unary [B,V], edges [(bi,bj,M)]).

        ``M[a_bi, a_bj]`` is the DIRECTED table of the outgoing edge ``block[bi] -> block[bj]``;
        both directions of a reciprocal pair are emitted separately, exactly as upstream, because
        the pair tables are genuinely asymmetric.
        """
        block = [int(p) for p in block]
        B = len(block)
        blk = jnp.asarray(block, jnp.int32)
        unary, edge_tabs, edge_hit = _block_potentials(
            self.etab, self.nbr_idx, self.pe, self.self_term,
            self.in_src, self.in_slot, self.in_mask, jnp.asarray(S, jnp.int32), blk)
        hit = np.asarray(edge_hit)
        edges = [(bi, bj, edge_tabs[bi, bj]) for bi in range(B) for bj in range(B)
                 if bi != bj and hit[bi, bj]]
        return unary, edges


@partial(jax.jit, static_argnums=())
def _rows_at(etab, nbr_idx, pe, self_term, in_src, in_slot, in_mask, S, positions):
    f = lambda i: _cond_energy_at(etab, nbr_idx, pe, self_term, in_src, in_slot, in_mask, S, i)
    return jax.vmap(f)(positions)


@partial(jax.jit, static_argnums=(10,))
def _rows_for_candidates(etab, nbr_idx, pe, self_term, in_src, in_slot, in_mask, S, p, centers, V):
    def one(a):
        Sa = S.at[p].set(a)
        f = lambda c: _cond_energy_at(etab, nbr_idx, pe, self_term, in_src, in_slot, in_mask, Sa, c)
        return jax.vmap(f)(centers)
    return jax.vmap(one)(jnp.arange(V, dtype=S.dtype))


@jax.jit
def _block_potentials(etab, nbr_idx, pe, self_term, in_src, in_slot, in_mask, S, block):
    """Vectorised port of ``block_stability_potentials`` for a fixed-size block."""
    V = etab.shape[-1]
    B = block.shape[0]

    def per_position(bi):
        p = block[bi]
        nb = nbr_idx[p]                                     # [K-1]
        pep = pe[p]                                         # [K-1, V, V]
        col = jnp.take_along_axis(pep, S[nb][:, None, None].repeat(V, axis=1), axis=-1)[..., 0]
        in_block = jnp.any(nb[:, None] == block[None, :], axis=-1)
        fixed = (~in_block) & (nb != p)
        u = self_term[p] + jnp.sum(jnp.where(fixed[:, None], col, 0.0), axis=0)

        msrc, mslot, mmask = in_src[p], in_slot[p], in_mask[p]
        inc_fixed = mmask & (~jnp.any(msrc[:, None] == block[None, :], axis=-1))
        u = u + jnp.sum(etab[msrc, mslot, S[msrc], :] * inc_fixed[:, None], axis=0)

        # directed within-block tables: for each bj, the (at most one) slot with nb == block[bj]
        def to_bj(bj):
            sel = (nb == block[bj]) & (block[bj] != p)
            M = jnp.sum(jnp.where(sel[:, None, None], pep, 0.0), axis=0)
            return M, jnp.any(sel)
        return u, jax.vmap(to_bj)(jnp.arange(B))

    unary, (edge_tabs, edge_hit) = jax.vmap(per_position)(jnp.arange(B))
    return unary, edge_tabs, edge_hit
