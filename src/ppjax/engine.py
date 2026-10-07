# Origin: cc (Claude Code, 2026-10-02) - ported from
#         ~/ProtonPottsMPNN/foundry/models/mpnn/src/mpnn/inference_engines/potts_mpnn_ph.py
#         (PottsMPNNPHEngine, _PHContext and the design/placement helpers). Upstream read-only.
# Purpose: the pH-switch design engine in JAX. Control flow, visit orders, tie-breaks and RNG
#          consumption follow the torch original exactly so the same seed yields the same designs.

from __future__ import annotations

import copy
import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ppjax import rng as trng
from ppjax.compat import check_knn_tie_boundary, warn_on_near_tie, warn_on_ties
from ppjax.criteria import (
    PHDesignCriteria, PHDesignOutput, PHDesignSet, PlacementPin, PlacementPlan,
    WHOLE_CHAIN_METHODS,
)
from ppjax.scorer import PottsScorer
from ppjax.tokens import TokenEncoding, three_to_one, unknown_aa

INF = float("inf")


# ---------------------------------------------------------------------------
# small numeric helpers (float32 throughout, as torch runs them)
# ---------------------------------------------------------------------------

def _softmax32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = np.max(x)
    e = np.exp((x - m).astype(np.float32), dtype=np.float32)
    return (e / np.sum(e, dtype=np.float32)).astype(np.float32)


def _pick(score: np.ndarray, temperature: float) -> int:
    """``_pick``: argmin at T<=0, else one ``torch.multinomial`` draw from softmax(-score/T)."""
    score = np.asarray(score, dtype=np.float32)
    if temperature <= 0:
        return int(np.argmin(score))
    return trng.default_generator().multinomial1(
        _softmax32(-score / np.float32(temperature)))


def _cond_dist(E_LV: np.ndarray, valid_mask: np.ndarray, temperature: float) -> np.ndarray:
    t = temperature if temperature > 0 else 1e-3
    valid_idx = np.nonzero(valid_mask)[0]
    x = (-np.asarray(E_LV, np.float32)[:, valid_idx] / np.float32(t)).astype(np.float32)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x, dtype=np.float32)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def _entropy_bits(p: np.ndarray) -> float:
    p = np.maximum(np.asarray(p, np.float32), np.float32(1e-12))
    return float(-(p * np.log2(p)).sum(-1))


def _zscale(deltas: np.ndarray, valid_mask: np.ndarray) -> float:
    """Std of the finite, valid single-mutation deltas, floored away from zero."""
    vals = np.asarray(deltas)[:, valid_mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    return max(float(np.std(vals.astype(np.float32))), 1e-6)


def _seq_entropy_bits(letters: str) -> float:
    from collections import Counter
    n = len(letters)
    if n == 0:
        return 0.0
    p = np.array([v / n for v in Counter(letters).values()], dtype=float)
    return float(-(p * np.log2(p)).sum())


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class PHContext:
    """Everything one featurised backbone contributes to every design run against it."""

    encoding: TokenEncoding
    extended_vocab: Any
    canonical_map: np.ndarray
    token_res_id: np.ndarray
    token_res_name: np.ndarray
    token_chain_id: np.ndarray
    S_native: np.ndarray
    scorer: PottsScorer
    free_mask: np.ndarray
    chainA: np.ndarray
    binder_chain: str
    V: int
    L: int
    K: int
    eidx_np: np.ndarray
    unknown_indices: List[int]
    region_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    decoder: Any = None
    base_seed: int = 0
    external_initial_sequences: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.chA_free_idx = np.nonzero(self.free_mask & self.chainA)[0]
        self.chainA_all_idx = np.nonzero(self.chainA)[0]
        self.res_id_to_pos = {int(self.token_res_id[int(p)]): int(p)
                              for p in self.chainA_all_idx}
        self._unk_one = three_to_one().get(unknown_aa(), "X")
        self._one_to_idx: Dict[str, int] = {}
        for idx in range(self.V):
            one = three_to_one().get(str(self.encoding.idx_to_token[idx]))
            if one is not None and one not in self._one_to_idx:
                self._one_to_idx[one] = idx
        self._stab_cache: Dict[Tuple[float, float], PottsScorer] = {}

    # --- fields --------------------------------------------------------------
    def field_potts(self, S: np.ndarray) -> np.ndarray:
        return np.asarray(self.scorer.cond_energy(S))

    def field_mpnn(self, S: np.ndarray) -> np.ndarray:
        if self.decoder is None:
            raise RuntimeError(
                "backend='mpnn' needs the decoder field; build the context with "
                "with_decoder=True, which requires the per-residue 'temperature' feature.")
        return self.decoder.field(S)

    def sample_decoder(self, n: int) -> np.ndarray:
        if self.decoder is None:
            raise RuntimeError("method='mpnn_sample' needs the decoder; build the context "
                               "with with_decoder=True.")
        return self.decoder.sample(self.S_native, n)

    def stability_scorer(self, w_self: float, w_pair: float) -> PottsScorer:
        if w_self == w_pair:
            return self.scorer
        key = (round(float(w_self), 6), round(float(w_pair), 6))
        sc = self._stab_cache.get(key)
        if sc is None:
            sc = self.scorer.reweighted(w_self, w_pair)
            self._stab_cache[key] = sc
        return sc

    # --- sequence helpers ----------------------------------------------------
    def encode_initial_sequence(self, one_letter: str) -> np.ndarray:
        binder_idx = self.chainA_all_idx.tolist()
        if len(one_letter) != len(binder_idx):
            raise ValueError(
                f"seed length {len(one_letter)} != binder length {len(binder_idx)} "
                f"(chain {self.binder_chain}) - seed must be the binder-chain sequence")
        t2i = self.encoding.token_to_idx
        unk = t2i.get("UNK", self.unknown_indices[0] if self.unknown_indices else 0)
        S = self.S_native.copy()
        for pos, ch in zip(binder_idx, one_letter):
            S[pos] = self._one_to_idx.get(ch, unk)
        return S

    def valid_aa_mask(self, forbidden_tokens: Sequence[str]) -> np.ndarray:
        mask = np.ones(self.V, dtype=bool)
        for idx in self.unknown_indices:
            if idx < self.V:
                mask[idx] = False
        t2i = self.encoding.token_to_idx
        unk = t2i.get("UNK")
        if unk is not None and unk < self.V:
            mask[unk] = False
        for tok in forbidden_tokens:
            idx = t2i.get(tok)
            if idx is not None and idx < self.V:
                mask[idx] = False
        return mask

    def neighbour_mask(self, center: int, k_max: int = 0) -> np.ndarray:
        K = self.K if k_max <= 0 else min(self.K, int(k_max) + 1)
        m = np.zeros(self.L, dtype=bool)
        for k in range(1, K):
            j = int(self.eidx_np[center, k])
            if j != center and bool(self.free_mask[j]) and bool(self.chainA[j]):
                m[j] = True
        for i, _ in np.argwhere(self.eidx_np[:, 1:K] == center):
            i = int(i)
            if i != center and bool(self.free_mask[i]) and bool(self.chainA[i]):
                m[i] = True
        return m

    def region_mask(self, names: Sequence[str]) -> np.ndarray:
        free = self.free_mask & self.chainA
        if not names or "all" in names:
            return free.copy()
        base = np.zeros(self.L, dtype=bool)
        for n in names:
            m = self.region_masks.get(n)
            if m is not None:
                base = base | m
        return base & free

    def binder_pos_of_res_id(self, res_id: int) -> int:
        pos = self.res_id_to_pos.get(int(res_id))
        if pos is None:
            raise ValueError(f"res_id {res_id} is not a binder position on chain {self.binder_chain}.")
        return pos

    def decode_canonical(self, seq: np.ndarray, positions: Sequence[int]) -> str:
        idx_to_token = self.encoding.idx_to_token
        t2o = three_to_one()
        return "".join(t2o.get(str(idx_to_token[int(self.canonical_map[int(seq[i])])]), self._unk_one)
                       for i in positions)

    def extended_tokens(self, seq: np.ndarray, positions: Sequence[int]) -> List[str]:
        idx_to_token = self.encoding.idx_to_token
        return [str(idx_to_token[int(seq[i])]) for i in positions]


# ---------------------------------------------------------------------------
# Outcome metrics
# ---------------------------------------------------------------------------

def _selective_energy_sum(ctx: PHContext, seq: np.ndarray, pins: Sequence[PlacementPin]) -> float:
    """Sum over pinned centres of ``e_P - mean_d e_D``, all centres locked protonated.

    ``cond_energy_at(S, c)`` is independent of ``S[c]`` whenever no position is its own neighbour
    at a slot k >= 1 - the self field is a full row and the pair terms read the NEIGHBOURS' tokens.
    Under that condition one conditional row per centre yields both states and is the same
    arithmetic the torch helper performs, with its redundant re-evaluations dropped. When the graph
    DOES contain such a self-loop the shortcut is not an identity, so the literal per-token path is
    taken instead (``PottsScorer.has_self_loops``, measured False for every shipped structure).
    """
    if not pins:
        return 0.0
    s = seq.copy()
    for p in pins:
        s[p.position] = p.prot_idx
    if ctx.scorer.has_self_loops:
        return float(sum(_selective_energy_of(ctx, s, p.position, p.prot_idx, p.dep_idxs)
                         for p in pins))
    rows = np.asarray(ctx.scorer.rows_at(s, [p.position for p in pins]))
    total = 0.0
    for r, p in enumerate(pins):
        ed = sum(float(rows[r, d]) for d in p.dep_idxs) / len(p.dep_idxs)
        total += float(rows[r, p.prot_idx]) - ed
    return float(total)


def _selective_energy_of(ctx: PHContext, seq: np.ndarray, center: int, prot_idx: int,
                         dep_idxs: Sequence[int]) -> float:
    s = seq.copy()
    s[center] = prot_idx
    row = np.asarray(ctx.scorer.cond_energy_at(s, center))
    e_p = float(row[prot_idx])
    if ctx.scorer.has_self_loops:                     # the centre's own token enters its row
        ed = 0.0
        for d in dep_idxs:
            s[center] = d
            ed += float(np.asarray(ctx.scorer.cond_energy_at(s, center))[d])
        return e_p - ed / len(dep_idxs)
    return e_p - sum(float(row[d]) for d in dep_idxs) / len(dep_idxs)


def _global_protonation_dH(ctx: PHContext, seq: np.ndarray) -> Optional[float]:
    """Binder pH-response ``H(all titratable protonated) - H(deprotonated)``."""
    t2i = ctx.encoding.token_to_idx
    if any(t not in t2i for t in ("HIS-P", "ASP-P", "ASP-D", "GLU-P", "GLU-D")):
        return None
    if "HID" in t2i and "HIE" in t2i:
        his_dep_tokens = ("HID", "HIE")
    elif "HIS-S" in t2i:
        his_dep_tokens = ("HIS-S",)
    else:
        return None
    rn = ctx.token_res_name
    his = (rn == "HIS") & ctx.chainA
    asp = (rn == "ASP") & ctx.chainA
    glu = (rn == "GLU") & ctx.chainA
    sp = seq.copy()
    sp[his] = t2i["HIS-P"]; sp[asp] = t2i["ASP-P"]; sp[glu] = t2i["GLU-P"]
    h_prot = ctx.scorer.H_of(sp)
    sd = seq.copy(); sd[asp] = t2i["ASP-D"]; sd[glu] = t2i["GLU-D"]
    h_his_dep = 0.0
    for tok in his_dep_tokens:
        sd[his] = t2i[tok]
        h_his_dep += ctx.scorer.H_of(sd)
    return h_prot - h_his_dep / len(his_dep_tokens)


def _energy_snapshot(ctx: PHContext, S, pins) -> Dict[str, Any]:
    binder_idx = ctx.chainA_all_idx.tolist()
    return {
        "selective_energy": _selective_energy_sum(ctx, S, pins),
        "global_protonation_dH": _global_protonation_dH(ctx, S),
        "potts_energy": ctx.scorer.H_of(S),
        "canonical_sequence": ctx.decode_canonical(S, binder_idx),
        "extended_tokens": " ".join(ctx.extended_tokens(S, binder_idx)),
    }


def _record(trajectory, ctx, S, pins, *, initial=False) -> None:
    if trajectory is None:
        return
    if initial and trajectory:
        return
    snap = {"step": len(trajectory)}
    snap.update(_energy_snapshot(ctx, S, pins))
    trajectory.append(snap)


# ---------------------------------------------------------------------------
# block_descent machinery
# ---------------------------------------------------------------------------

def _block_partners(ctx: PHContext, p: int, neigh_set, block_size: int) -> List[int]:
    """``p`` plus its closest coupled design-set partners, in the model's own kNN order."""
    block = [int(p)]
    if block_size <= 1:
        return block
    for j in ctx.eidx_np[p, 1:].tolist():
        j = int(j)
        if j != p and j in neigh_set and j not in block:
            block.append(j)
            if len(block) == block_size:
                break
    return block


def _selective_rows_for_candidates(ctx, S, p, pins) -> np.ndarray:
    """[V] selective energy with position ``p`` set to each candidate AA (centres locked)."""
    s = S.copy()
    for q in pins:
        s[q.position] = q.prot_idx
    if ctx.scorer.has_self_loops:                     # vectorised shortcut is not an identity here
        out = np.zeros(ctx.V, dtype=np.float64)
        cur = int(s[p])
        for a in range(ctx.V):
            s[p] = a
            out[a] = _selective_energy_sum(ctx, s, pins)
        s[p] = cur
        return out
    rows = np.asarray(ctx.scorer.rows_for_candidates(s, p, [q.position for q in pins]))
    out = np.zeros(ctx.V, dtype=np.float64)
    for r, q in enumerate(pins):
        ed = np.zeros(ctx.V, dtype=np.float64)
        for d in q.dep_idxs:
            ed += rows[:, r, d]
        out += rows[:, r, q.prot_idx] - ed / len(q.dep_idxs)
    return out


def _block_mutation_tables(ctx, valid_mask, S, neigh, pins, want_global):
    """Single-mutation delta tables over the design positions (used to freeze the z-scale)."""
    N, V = len(neigh), ctx.V
    dH = np.full((N, V), INF, dtype=np.float32)
    dSel = np.zeros((N, V), dtype=np.float32)
    dGlob = np.zeros((N, V), dtype=np.float32)
    valid_idx = np.nonzero(valid_mask)[0]
    S = S.copy()
    for p in pins:
        S[p.position] = p.prot_idx
    sel0 = _selective_energy_sum(ctx, S, pins) if pins else 0.0
    glob0 = _global_protonation_dH(ctx, S) if want_global else None
    for n, i in enumerate(neigh):
        cur = int(S[i])
        e_i = np.asarray(ctx.scorer.cond_energy_at(S, i), dtype=np.float32)
        row = e_i - e_i[cur]
        dH[n, valid_idx] = row[valid_idx]
        if pins:
            sel_all = _selective_rows_for_candidates(ctx, S, int(i), pins)
            for a in valid_idx:
                if a == cur:
                    continue
                dSel[n, a] = np.float32(sel_all[a] - sel0)
        if want_global:
            for a in valid_idx:
                if a == cur:
                    continue
                S[i] = a
                g = _global_protonation_dH(ctx, S)
                dGlob[n, a] = (g - glob0) if g is not None else 0.0
            S[i] = cur
    return dH, dSel, dGlob


def _broadcast_sum(vecs: List[np.ndarray], V: int) -> np.ndarray:
    B = len(vecs)
    J = vecs[0].reshape([V] + [1] * (B - 1)).astype(np.float32)
    for bi in range(1, B):
        shape = [1] * B
        shape[bi] = V
        J = J + vecs[bi].reshape(shape).astype(np.float32)
    return J


def _block_zscales(ctx, crit, valid_mask, S, neigh, pins, want_global, stab_scorer=None):
    """Freeze the z-scales from the BLOCK (V**block_size) joint spread (``zscale_mode='block'``)."""
    V = ctx.V
    stab_scorer = stab_scorer if stab_scorer is not None else ctx.scorer
    bsize = max(1, int(crit.block_size))
    neigh_set = set(int(x) for x in neigh)
    _dH0, dSel0, dGlob0 = _block_mutation_tables(ctx, valid_mask, S, neigh, pins, want_global)
    row_of = {int(p): n for n, p in enumerate(neigh)}
    inf_mask = np.where(~valid_mask, np.float32(INF), np.float32(0.0))

    def unary_block_var(vecs):
        B = len(vecs)
        J = _broadcast_sum(vecs, V)
        for bi in range(B):
            shape = [1] * B
            shape[bi] = V
            J = J + inf_mask.reshape(shape)
        v = J[np.isfinite(J)]
        return float(np.var(v)) if v.size > 1 else None

    stab_v, sel_v, glob_v = [], [], []
    for p in neigh:
        block = _block_partners(ctx, p, neigh_set, bsize)
        B = len(block)
        su, sedges = stab_scorer.block_potentials(S, block)
        su = np.asarray(su, dtype=np.float32)
        J = _broadcast_sum([su[bi] for bi in range(B)], V)
        for (bi, bj, M) in sedges:
            J = J + _pair_broadcast(np.asarray(M, np.float32), bi, bj, B, V)
        for bi in range(B):
            shape = [1] * B
            shape[bi] = V
            J = J + inf_mask.reshape(shape)
        vals = J[np.isfinite(J)]
        if vals.size > 1:
            stab_v.append(float(np.var(vals)))
        srows = [dSel0[row_of[int(b)]] for b in block]
        if any(float(np.abs(r).sum()) > 0 for r in srows):
            var = unary_block_var(srows)
            if var is not None:
                sel_v.append(var)
        if want_global:
            grows = [dGlob0[row_of[int(b)]] for b in block]
            if any(float(np.abs(r).sum()) > 0 for r in grows):
                var = unary_block_var(grows)
                if var is not None:
                    glob_v.append(var)

    def pool(vs):
        return max((sum(vs) / len(vs)) ** 0.5, 1e-6) if vs else 1.0
    return pool(stab_v), pool(sel_v), (pool(glob_v) if want_global else 1.0)


def _pair_broadcast(M: np.ndarray, bi: int, bj: int, B: int, V: int) -> np.ndarray:
    """``M[a_bi, a_bj]`` laid onto block axes ``bi`` and ``bj`` (either order), as the upstream
    ``bi < bj`` orientation guarantee intends."""
    shape = [1] * B
    shape[bi] = V
    shape[bj] = V
    return (M if bi < bj else M.T).reshape(shape)


def _parent_vocab_mask(ctx, parents) -> np.ndarray:
    idx_to_token = ctx.encoding.idx_to_token
    cm = ctx.canonical_map
    pset = set(parents)
    return np.array([str(idx_to_token[int(cm[a])]) in pset for a in range(ctx.V)], dtype=bool)


def _centre_energy_rank(ctx, pins, S, neigh) -> List[int]:
    """|contribution to the centres' selective gap| per designable position, descending."""
    sc = ctx.scorer
    etab = np.asarray(sc.etab)
    nbr = ctx.eidx_np[:, 1:]
    in_src, in_slot, in_mask = np.asarray(sc.in_src), np.asarray(sc.in_slot), np.asarray(sc.in_mask)
    score = {int(j): 0.0 for j in neigh}
    for pin in pins:
        c, prot = int(pin.position), int(pin.prot_idx)
        deps = [int(d) for d in pin.dep_idxs]
        slot_of = {int(j): t for t, j in enumerate(nbr[c].tolist())}
        inc_of = {int(in_src[c, e]): int(in_slot[c, e])
                  for e in range(in_src.shape[1]) if in_mask[c, e]}
        for j in score:
            sj = int(S[j])
            e_p = e_d = 0.0
            if j in slot_of:
                t = slot_of[j]
                e_p += float(etab[c, t + 1, prot, sj])
                e_d += sum(float(etab[c, t + 1, d, sj]) for d in deps) / max(len(deps), 1)
            if j in inc_of:
                row = etab[j, inc_of[j], sj, :]
                e_p += float(row[prot])
                e_d += sum(float(row[d]) for d in deps) / max(len(deps), 1)
            score[j] += abs(e_p - e_d)
    return sorted(score, key=lambda j: (-score[j], j))


def _knn_rank(ctx, pins, neigh) -> List[int]:
    best = {int(j): 10 ** 6 for j in neigh}
    for pin in pins:
        c = int(pin.position)
        for r, j in enumerate(ctx.eidx_np[c, 1:].tolist()):
            j = int(j)
            if j in best:
                best[j] = min(best[j], r)
    return sorted(best, key=lambda j: (best[j], j))


def _block_descent(ctx, crit, valid_mask, pins, neigh, initial_sequence, rng, trajectory=None):
    """Block coordinate descent on the z-scaled combined objective
    ``J = (1-lam)*z(H_stab) + lam*z(sum sel) [+ gw*z(global)]``, centres pinned to RES-P."""
    V = ctx.V
    lam = float(crit.combined_lambda)
    gw = float(crit.global_weight)
    want_global = gw != 0.0
    arw = float(crit.adjacent_repeat_weight)
    want_repeat = arw != 0.0
    rw = float(crit.repetitive_window_weight)
    rw_parents = tuple(crit.repetitive_window_parents)
    _rw_gate = set(crit.repetitive_window_gate_types)
    want_rep = rw != 0.0 and bool(rw_parents) and (
        not _rw_gate or any(p.protonation_type in _rw_gate for p in pins))
    rrad = max(1, int(crit.repetitive_window_radius))
    w_self, w_pair = crit.stability_weights()
    stab_scorer = ctx.stability_scorer(w_self, w_pair)
    bsize = max(1, int(crit.block_size))

    S = initial_sequence.copy()
    for pin in pins:
        S[pin.position] = pin.prot_idx
    if not neigh:
        return S

    if crit.sweep_order == "energy":
        neigh = _centre_energy_rank(ctx, pins, S, neigh)
    elif crit.sweep_order == "knn":
        neigh = _knn_rank(ctx, pins, neigh)

    if crit.zscale_mode == "block":
        sdH, sdSel, sdGlob = _block_zscales(ctx, crit, valid_mask, S, neigh, pins, want_global,
                                            stab_scorer)
    else:
        dH0, dSel0, dGlob0 = _block_mutation_tables(ctx, valid_mask, S, neigh, pins, want_global)
        sdH = _zscale(dH0, valid_mask)
        sel_rows = np.abs(dSel0).sum(1) > 0
        sdSel = _zscale(dSel0[sel_rows] if sel_rows.any() else dSel0, valid_mask)
        glob_rows = np.abs(dGlob0).sum(1) > 0
        sdGlob = (_zscale(dGlob0[glob_rows], valid_mask) if (want_global and glob_rows.any()) else 1.0)
    wH, wSel, wGlob = (1.0 - lam) / sdH, lam / sdSel, (gw / sdGlob if want_global else 0.0)

    cmap = ctx.canonical_map
    same_canon = ((cmap.reshape(-1, 1) == cmap.reshape(1, -1)).astype(np.float32)
                  if want_repeat else None)
    rep_mask = _parent_vocab_mask(ctx, rw_parents).astype(np.float32) if want_rep else None
    res_id = ctx.token_res_id

    invalid = ~valid_mask
    inf_vec = np.where(invalid, np.float32(INF), np.float32(0.0))
    neigh_set = set(int(x) for x in neigh)

    def combined_unary(p, block_set):
        cur = int(S[p])
        u = np.zeros(V, dtype=np.float32)
        if pins:
            sel0 = _selective_energy_sum(ctx, S, pins)
            sel_all = _selective_rows_for_candidates(ctx, S, int(p), pins)
            for a in range(V):
                if invalid[a] or a == cur:
                    continue
                u[a] += np.float32(wSel * (sel_all[a] - sel0))
        if want_global:
            glob0 = _global_protonation_dH(ctx, S)
            Sl = S.copy()
            for a in range(V):
                if invalid[a] or a == cur:
                    continue
                Sl[p] = a
                g = _global_protonation_dH(ctx, Sl)
                if g is not None:
                    u[a] += np.float32(wGlob * (g - glob0))
            Sl[p] = cur
        if want_rep:
            ri = int(res_id[p])
            n_ctx = 0
            for dd in range(-rrad, rrad + 1):
                if dd == 0:
                    continue
                q = ctx.res_id_to_pos.get(ri + dd)
                if q is not None and int(q) not in block_set and bool(rep_mask[int(S[int(q)])]):
                    n_ctx += 1
            if n_ctx:
                u = u + np.float32(rw * n_ctx) * rep_mask
        return u

    changed_any = True
    rounds = 0
    _record(trajectory, ctx, S, pins, initial=True)
    while changed_any and rounds < int(crit.block_max_rounds):
        changed_any = False
        rounds += 1
        for p in neigh:
            block = _block_partners(ctx, p, neigh_set, bsize)
            B = len(block)
            su, sedges = stab_scorer.block_potentials(S, block)
            su = np.asarray(su, dtype=np.float32)
            cu = np.float32(wH) * su
            block_set = set(int(x) for x in block)
            for bi, p_b in enumerate(block):
                cu[bi] = cu[bi] + combined_unary(p_b, block_set)
            J = _broadcast_sum([cu[bi] for bi in range(B)], V)
            for (bi, bj, M) in sedges:
                J = J + _pair_broadcast(np.float32(wH) * np.asarray(M, np.float32), bi, bj, B, V)
            if want_repeat:
                bres = [int(res_id[p_b]) for p_b in block]
                bset = set(int(p_b) for p_b in block)
                for bi in range(B):
                    for bj in range(bi + 1, B):
                        if abs(bres[bi] - bres[bj]) == 1:
                            J = J + _pair_broadcast(np.float32(arw) * same_canon, bi, bj, B, V)
                    for dd in (-1, 1):
                        q = ctx.res_id_to_pos.get(bres[bi] + dd)
                        if q is not None and int(q) not in bset:
                            shape = [1] * B
                            shape[bi] = V
                            J = J + (np.float32(arw) * same_canon[:, int(S[int(q)])]).reshape(shape)
            if want_rep:
                bres_r = [int(res_id[p_b]) for p_b in block]
                pair_pen = np.outer(rep_mask, rep_mask).astype(np.float32)
                for bi in range(B):
                    for bj in range(bi + 1, B):
                        if abs(bres_r[bi] - bres_r[bj]) <= rrad:
                            J = J + _pair_broadcast(np.float32(rw) * pair_pen, bi, bj, B, V)
            for bi in range(B):
                shape = [1] * B
                shape[bi] = V
                J = J + inf_vec.reshape(shape)
            flat = J.reshape(-1).astype(np.float32)
            T = float(crit.temperature)
            if T <= 0:
                choice = int(np.argmin(flat))
            else:
                probs = _softmax32(-(flat - flat.min()) / np.float32(T))
                choice = trng.default_generator().multinomial1(probs)
            assign = list(np.unravel_index(choice, [V] * B))
            moved = False
            for bi, p_b in enumerate(block):
                a = int(assign[bi])
                if a != int(S[p_b]):
                    S[p_b] = a
                    moved = True
            if moved:
                changed_any = True
                _record(trajectory, ctx, S, pins)
    return S


def _greedy_energy_block(ctx, crit, valid_mask, pins, neigh, initial_sequence, rng, trajectory=None):
    """Centre-free, dynamically re-ranked block descent on the pure Potts stability energy."""
    V = ctx.V
    K = max(1, int(crit.block_size))
    T = float(crit.temperature)
    tol = 1e-4
    w_self, w_pair = crit.stability_weights()
    stab_scorer = ctx.stability_scorer(w_self, w_pair)

    rw = float(crit.repetitive_window_weight)
    rw_parents = tuple(crit.repetitive_window_parents)
    _rw_gate = set(crit.repetitive_window_gate_types)
    rep_all = "ALL" in rw_parents
    want_rep = rw != 0.0 and (rep_all or (bool(rw_parents) and (
        not _rw_gate or any(p.protonation_type in _rw_gate for p in pins))))
    rrad = max(1, int(crit.repetitive_window_radius))
    cmap = ctx.canonical_map
    same_canon = ((cmap.reshape(-1, 1) == cmap.reshape(1, -1)).astype(np.float32)
                  if (want_rep and rep_all) else None)
    rep_mask = (_parent_vocab_mask(ctx, rw_parents).astype(np.float32)
                if (want_rep and not rep_all) else None)
    res_id = ctx.token_res_id

    invalid = ~valid_mask
    inf_vec = np.where(invalid, np.float32(INF), np.float32(0.0))

    S = initial_sequence.copy()
    if not neigh:
        return S

    for i in neigh:
        ci = int(cmap[int(S[int(i)])])
        if bool(valid_mask[ci]):
            S[int(i)] = ci

    dH0, _, _ = _block_mutation_tables(ctx, valid_mask, S, neigh, [], False)
    wH = 1.0 / _zscale(dH0, valid_mask)

    best_S = S.copy()
    best_H = stab_scorer.H_of(S)
    _record(trajectory, ctx, S, pins, initial=True)
    patience, cap = crit.cv_patience * len(neigh), crit.cv_max * len(neigh)
    since = step = 0
    while since < patience and step < cap:
        rows = np.asarray(stab_scorer.rows_at(S, [int(i) for i in neigh]), dtype=np.float32)
        improving = []
        for n, i in enumerate(neigh):
            e_i = rows[n]
            cur = int(S[int(i)])
            e_valid = e_i.copy()
            e_valid[invalid] = np.float32(INF)
            delta = float(e_valid.min()) - float(e_i[cur])
            if delta < -tol:
                improving.append((delta, int(i)))
        if not improving:
            break
        improving.sort(key=lambda t: (t[0], t[1]))
        # The block is whichever positions improve MOST, re-ranked every step. As the search
        # converges the improvements crowd towards zero, so this cut is the one place in the whole
        # engine where a sub-1e-4 margin routinely decides a discrete choice - and it is why
        # greedy_energy_block is the only reference case that diverges from torch. (block_descent
        # picks its block from E_idx order, i.e. integers, and is immune.)
        warn_on_near_tie([dd for dd, _ in improving], K,
                         "greedy_energy_block block selection")
        block = [i for _, i in improving[:K]]
        B = len(block)

        su, sedges = stab_scorer.block_potentials(S, block)
        su = np.asarray(su, dtype=np.float32)
        cu = np.float32(wH) * su
        if want_rep:
            block_set = set(int(x) for x in block)
            for bi, p_b in enumerate(block):
                ri = int(res_id[p_b])
                for dd in range(-rrad, rrad + 1):
                    if dd == 0:
                        continue
                    q = ctx.res_id_to_pos.get(ri + dd)
                    if q is None or int(q) in block_set:
                        continue
                    if rep_all:
                        cu[bi] = cu[bi] + np.float32(rw) * same_canon[:, int(S[int(q)])]
                    elif bool(rep_mask[int(S[int(q)])]):
                        cu[bi] = cu[bi] + np.float32(rw) * rep_mask
        J = _broadcast_sum([cu[bi] for bi in range(B)], V)
        for (bi, bj, M) in sedges:
            J = J + _pair_broadcast(np.float32(wH) * np.asarray(M, np.float32), bi, bj, B, V)
        if want_rep:
            bres = [int(res_id[p_b]) for p_b in block]
            pair_pen = same_canon if rep_all else (
                np.outer(rep_mask, rep_mask).astype(np.float32) if rep_mask is not None else None)
            if pair_pen is not None:
                for bi in range(B):
                    for bj in range(bi + 1, B):
                        if abs(bres[bi] - bres[bj]) <= rrad:
                            J = J + _pair_broadcast(np.float32(rw) * pair_pen, bi, bj, B, V)
        for bi in range(B):
            shape = [1] * B
            shape[bi] = V
            J = J + inf_vec.reshape(shape)
        flat = J.reshape(-1).astype(np.float32)
        if T <= 0:
            choice = int(np.argmin(flat))
        else:
            probs = _softmax32(-(flat - flat.min()) / np.float32(T))
            choice = trng.default_generator().multinomial1(probs)
        assign = list(np.unravel_index(choice, [V] * B))
        for bi, p_b in enumerate(block):
            S[p_b] = int(assign[bi])
        step += 1

        H = stab_scorer.H_of(S)
        if H < best_H - 1e-9:
            best_H, best_S, since = H, S.copy(), 0
        else:
            since += 1
        _record(trajectory, ctx, S, pins)
    return best_S


# ---------------------------------------------------------------------------
# single-site MCMC variants
# ---------------------------------------------------------------------------

def _score_at(ctx, field_at, valid_mask, S, i, pins, *, selective, lam=None):
    for p in pins:
        S[p.position] = p.prot_idx
    e_stab = np.asarray(field_at(S, i), dtype=np.float32).copy()
    if not selective and lam is None:
        return np.where(~valid_mask, np.float32(INF), e_stab)
    ed = np.zeros_like(e_stab)
    for p in pins:
        c = np.zeros_like(e_stab)
        for d in p.dep_idxs:
            S[p.position] = d
            c = c + np.asarray(field_at(S, i), dtype=np.float32)
        S[p.position] = p.prot_idx
        ed = ed + c / np.float32(len(p.dep_idxs))
    ed = ed / np.float32(len(pins))
    sc = (np.float32(1.0 + lam) * e_stab - np.float32(lam) * ed) if lam is not None else (e_stab - ed)
    return np.where(~valid_mask, np.float32(INF), sc)


def _converge(ctx, crit, field_at, valid_mask, pins, neigh, initial_sequence, S0=None,
              trajectory=None):
    lam = crit.combined_lambda if crit.method == "converged_mcmc_combined" else None
    selective = crit.selective if lam is None else False
    S = (S0.copy() if S0 is not None else initial_sequence.copy())
    for p in pins:
        S[p.position] = p.prot_idx
    _record(trajectory, ctx, S, pins, initial=True)
    patience, cap = crit.cv_patience * len(neigh), crit.cv_max * len(neigh)
    since = step = 0
    while neigh and since < patience and step < cap:
        i = neigh[int(trng.default_generator().randint(len(neigh), 1)[0])]
        tok = _pick(_score_at(ctx, field_at, valid_mask, S, i, pins, selective=selective, lam=lam),
                    crit.temperature)
        since = 0 if tok != int(S[i]) else since + 1
        S[i] = tok
        step += 1
        _record(trajectory, ctx, S, pins)
    return S


def _two_phase(ctx, crit, field_at, valid_mask, pins, neigh, initial_sequence, trajectory=None):
    S = initial_sequence.copy()
    for p in pins:
        S[p.position] = p.prot_idx
    _record(trajectory, ctx, S, pins, initial=True)
    sel_pick, disruption = {}, {}
    for i in neigh:
        a = _pick(_score_at(ctx, field_at, valid_mask, S, i, pins, selective=True), crit.temperature)
        e_stab = np.asarray(field_at(S, i), dtype=np.float32)
        sel_pick[i] = a
        disruption[i] = float(e_stab[a] - e_stab[int(S[i])])
    order = sorted(neigh, key=lambda i: disruption[i])
    k = max(1, min(len(neigh) - 1, int(round(crit.two_phase_frac * len(neigh)))))
    for i in order[:k]:
        S[i] = sel_pick[i]
        _record(trajectory, ctx, S, pins)
    rest = [i for i in neigh if i not in set(order[:k])]
    crit2 = copy.copy(crit)
    crit2.method = "converged_mcmc"
    crit2.selective = False
    return _converge(ctx, crit2, field_at, valid_mask, pins, rest, initial_sequence, S0=S,
                     trajectory=trajectory)


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------

def _ranked_candidates(ctx, crit, field_fn, prot_idx, dep_idxs, initial_sequence, region_idx,
                       mask_seq) -> List[Tuple[int, float]]:
    region_idx = np.asarray(region_idx, dtype=np.int64)
    placement_seq = initial_sequence
    if mask_seq:
        placement_seq = initial_sequence.copy()
        placement_seq[ctx.chA_free_idx] = ctx.encoding.token_to_idx["UNK"]
    ef = np.asarray(field_fn(placement_seq), dtype=np.float32)
    base = ef[region_idx, placement_seq[region_idx]]
    score = ef[region_idx, prot_idx] - base
    if crit.selective:
        dep_dE = np.min(np.stack([ef[region_idx, d] - base for d in dep_idxs], 0), axis=0)
        score = score - dep_dE
    warn_on_ties(score, "placement ranking (_ranked_candidates)")
    order = np.argsort(score, kind="stable").tolist()
    return [(int(region_idx[k]), float(score[k])) for k in order]


def _place_scan_potts(ctx, crit, region_idx, prot_idx, dep_idxs, initial_sequence, rng):
    return _ranked_candidates(ctx, crit, ctx.field_potts, prot_idx, dep_idxs, initial_sequence,
                              region_idx, mask_seq=False)


def _place_scan_mpnn(ctx, crit, region_idx, prot_idx, dep_idxs, initial_sequence, rng):
    return _ranked_candidates(ctx, crit, ctx.field_mpnn, prot_idx, dep_idxs, initial_sequence,
                              region_idx, mask_seq=True)


def _place_random(ctx, crit, region_idx, prot_idx, dep_idxs, initial_sequence, rng):
    idx = [int(p) for p in np.asarray(region_idx).tolist()]
    rng.shuffle(idx)
    return [(int(p), 0.0) for p in idx]


_PLACEMENT_FNS = {"random": _place_random, "scan_potts": _place_scan_potts,
                  "scan_mpnn": _place_scan_mpnn}


def placement_fn(placement_by: str) -> Callable:
    try:
        return _PLACEMENT_FNS[placement_by]
    except KeyError:
        raise ValueError(f"Unknown placement_by '{placement_by}'. Available: {list(_PLACEMENT_FNS)}")


def _locked_selective_score(ctx, field_fn, seq, pins) -> float:
    S = seq.copy()
    for p in pins:
        S[p.position] = p.prot_idx
    ef = np.asarray(field_fn(S), dtype=np.float32)
    total = 0.0
    for p in pins:
        e_p = float(ef[p.position, p.prot_idx])
        dep = min(float(ef[p.position, d]) for d in p.dep_idxs)
        total += e_p - dep
    return total


def _combos_or_sample(pool, k, cap, rng):
    n = len(pool)
    if k > n:
        return []
    if math.comb(n, k) <= cap * 4:
        combos = [c for c in itertools.combinations(pool, k) if len({p for p, _ in c}) == k]
        rng.shuffle(combos)
        return combos[:cap]
    out, seen, tries = [], set(), 0
    while len(out) < cap and tries < cap * 40:
        tries += 1
        combo = tuple(rng.sample(pool, k))
        if len({p for p, _ in combo}) != k:
            continue
        key = tuple(sorted(combo))
        if key in seen:
            continue
        seen.add(key)
        out.append(combo)
    return out


def _finalize_plan(ctx, crit, pins) -> Optional[PlacementPlan]:
    pins = sorted(pins, key=lambda p: p.position)
    pin_pos = {p.position for p in pins}
    if crit.infill_scope == "chain":
        designable = sorted(set(int(x) for x in ctx.chA_free_idx.tolist()) - pin_pos)
    else:
        dz = set()
        for p in pins:
            dz.update(int(x) for x in np.nonzero(
                ctx.neighbour_mask(p.position, crit.neighbour_k))[0].tolist())
        designable = sorted(dz - pin_pos)
    if not designable:
        return None
    if crit.max_mutations and len(designable) > crit.max_mutations:
        designable = sorted(_knn_rank(ctx, pins, designable)[:crit.max_mutations])
    label = "+".join(sorted(p.protonation_type for p in pins))
    return PlacementPlan(pins=pins, designable=designable, label=label)


def _decoded_prob_score(ctx, crit, field_fn, valid_mask, seq, positions) -> float:
    Pf = _cond_dist(field_fn(seq), valid_mask, crit.temperature)
    valid_idx = np.nonzero(valid_mask)[0]
    vp = {int(t): i for i, t in enumerate(valid_idx.tolist())}
    vals = [float(Pf[p, vp[int(seq[p])]]) for p in positions if int(seq[p]) in vp]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Whole-chain baseline: Gibbs sweeps on the Potts Hamiltonian
# ---------------------------------------------------------------------------

def potts_gibbs_optimize(scorer: PottsScorer, seq_init: np.ndarray, free_mask: np.ndarray,
                         temperature: float = 0.01, max_iters: int = 1000,
                         convergence_mode: bool = True, valid_aa_mask: Optional[np.ndarray] = None):
    """``PottsMPNN.potts_gibbs_optimize``: one random-order sweep per iteration, immediate update."""
    seqs = np.asarray(seq_init).copy()
    N = seqs.shape[0]
    free_positions = np.nonzero(free_mask)[0]
    gen = trng.default_generator()
    for n in range(N):
        for _ in range(max_iters):
            mutations = 0
            perm = gen.randperm(len(free_positions))
            order = free_positions[perm]
            for pos in order:
                pos = int(pos)
                E_i = np.asarray(scorer.cond_energy_at(seqs[n], pos), dtype=np.float32)
                if valid_aa_mask is not None:
                    E_i = np.where(~valid_aa_mask, np.float32(INF), E_i)
                probs = _softmax32(-E_i / np.float32(temperature))
                new_aa = gen.multinomial1(probs)
                if new_aa != int(seqs[n, pos]):
                    mutations += 1
                seqs[n, pos] = new_aa
            if convergence_mode and mutations == 0:
                break
    energies = np.array([scorer.H_of(seqs[n]) for n in range(N)], dtype=np.float64)
    return seqs, energies


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PottsMPNNPHEngine:
    """JAX port of ``mpnn.inference_engines.potts_mpnn_ph.PottsMPNNPHEngine``.

    Build it from a torch checkpoint with :meth:`from_checkpoint`, then call
    :meth:`run_ph_redesign` with a featurised backbone (see :mod:`ppjax.frontend`).
    """

    def __init__(self, params, enc: TokenEncoding, cfg, extended_vocab="v6") -> None:
        self.params = params
        self.encoding = enc
        self.cfg = cfg
        self.extended_vocab = extended_vocab

    # --- construction --------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, extended_vocab="v6",
                        etab_source: Optional[str] = None,
                        field_source: Optional[str] = None) -> "PottsMPNNPHEngine":
        from ppjax.checkpoint import load_state_dict
        from ppjax.model import ModelConfig
        from ppjax.params import (build_params, infer_etab_source, infer_field_source,
                                  infer_shapes)
        from ppjax.tokens import vocab_encoding
        flat = load_state_dict(checkpoint_path)
        shapes = infer_shapes(flat)
        enc = vocab_encoding(extended_vocab)
        if enc.n_tokens != shapes["vocab_size"]:
            raise ValueError(
                f"vocabulary {extended_vocab!r} has {enc.n_tokens} tokens but the checkpoint was "
                f"trained with {shapes['vocab_size']} - the token sets are weight-incompatible.")
        cfg = ModelConfig(
            vocab_size=shapes["vocab_size"], hidden_dim=shapes["hidden_dim"],
            num_encoder_layers=shapes["num_encoder_layers"],
            num_decoder_layers=shapes["num_decoder_layers"],
            field_source=field_source or infer_field_source(flat),
            etab_source=etab_source or infer_etab_source(flat))
        return cls(build_params(flat), enc, cfg, extended_vocab)

    def forward(self, features: Dict[str, Any]):
        from ppjax.model import forward_potts
        return forward_potts(self.params, self.encoding, self.cfg,
                             self.encoding.unknown_indices(), features)

    def build_context(self, features: Dict[str, Any], binder_chain: str, *,
                      token_res_id, token_res_name, token_chain_id,
                      region_masks: Optional[Dict[str, np.ndarray]] = None,
                      base_seed: int = 0, with_decoder: bool = True) -> PHContext:
        """``_build_context``: one forward pass, then everything the design loop slices off it."""
        out = self.forward(features)
        boundary = np.asarray(out["knn_boundary"])
        if boundary.size:
            check_knn_tie_boundary(boundary)
        scorer = PottsScorer.from_tables(out["etab_out"], out["E_idx"])
        S_native = np.asarray(features["S"]).reshape(-1).astype(np.int64)
        free_mask = np.asarray(out["designed_residue_mask"]).reshape(-1).astype(bool)
        chainA = np.asarray(token_chain_id) == binder_chain
        decoder = None
        if with_decoder and features.get("temperature") is not None:
            from ppjax.decoder import DecoderField
            decoder = DecoderField(
                self.params, self.cfg, self.encoding.unknown_indices(),
                out["h_V"], out["h_E"], np.asarray(out["E_idx"]),
                np.asarray(out["residue_mask"]), np.asarray(out["designed_residue_mask"]),
                temperature=features["temperature"],
                token_to_idx=self.encoding.token_to_idx,
                symmetry_equivalence_group=features.get("symmetry_equivalence_group"))
        return PHContext(
            encoding=self.encoding, extended_vocab=self.extended_vocab,
            canonical_map=self.encoding.canonical_map(),
            token_res_id=np.asarray(token_res_id), token_res_name=np.asarray(token_res_name),
            token_chain_id=np.asarray(token_chain_id), S_native=S_native, scorer=scorer,
            free_mask=free_mask, chainA=chainA, binder_chain=binder_chain,
            V=int(self.cfg.vocab_size), L=int(S_native.shape[0]), K=int(scorer.K),
            eidx_np=scorer.eidx_np, unknown_indices=self.encoding.unknown_indices(),
            region_masks=dict(region_masks or {}), base_seed=int(base_seed), decoder=decoder)

    # --- dispatch ------------------------------------------------------------
    @staticmethod
    def _criteria_fields(ctx: PHContext, crit: PHDesignCriteria):
        field_fn = ctx.field_potts if crit.backend == "potts" else ctx.field_mpnn
        if crit.backend == "potts":
            field_at = lambda S, i: np.asarray(ctx.scorer.cond_energy_at(S, i))
        else:
            field_at = lambda S, i: ctx.field_mpnn(S)[i]
        return field_fn, field_at, ctx.valid_aa_mask(crit.forbidden_tokens)

    def run_ph_redesign(self, *, ctx: PHContext, criteria_list: Sequence[PHDesignCriteria],
                        seed: int = 0, initial_sequences: Optional[Sequence[str]] = None
                        ) -> PHDesignSet:
        """Run each criteria against the shared Potts tables; deduped, lowest energy first.

        The torch engine's ``n_jobs`` fork pool is intentionally not reproduced: it only changes the
        ORDER results are produced in, and the serial path it is defined to equal is what runs here.
        """
        ctx.external_initial_sequences = list(initial_sequences) if initial_sequences else []
        seed_energies = []
        for s in ctx.external_initial_sequences:
            try:
                S = ctx.encode_initial_sequence(s)
                g = _global_protonation_dH(ctx, S)
                seed_energies.append({"potts_energy": ctx.scorer.H_of(S),
                                      "global_protonation_dH": (float(g) if g is not None else None)})
            except Exception:
                seed_energies.append({"potts_energy": None, "global_protonation_dH": None})

        criteria_list = list(criteria_list)
        tasks = []
        for ci, crit in enumerate(criteria_list):
            if crit.method in WHOLE_CHAIN_METHODS:
                tasks.append((ci, 0))
            else:
                for seq_i in range(self._n_placement_seeds(ctx, crit) or 1):
                    tasks.append((ci, seq_i))

        results = PHDesignSet()
        for ci, seq_i in tasks:
            crit = criteria_list[ci]
            field_fn, field_at, valid_mask = self._criteria_fields(ctx, crit)
            task_seed = seed + 1009 * ci
            if crit.method in WHOLE_CHAIN_METHODS:
                results.extend(self._run_whole_chain(ctx, crit, field_fn, valid_mask,
                                                     seed=task_seed))
            else:
                initial_sequence, seeded = self._placement_initial(ctx, crit, seq_i)
                results.extend(self._run_placement_one_seed(
                    ctx, crit, field_fn, field_at, valid_mask, task_seed, seq_i,
                    initial_sequence, seeded))
        results = results.deduped().sorted_by_energy()
        results.seed_energies = seed_energies
        return results

    def _run_whole_chain(self, ctx, crit, field_fn, valid_mask, *, seed) -> PHDesignSet:
        trng.manual_seed(seed)
        N = crit.num_designs
        if crit.method == "gibbs":
            seq_init = np.broadcast_to(ctx.S_native, (N, ctx.L)).copy()
            seq_opt, _ = potts_gibbs_optimize(
                ctx.scorer, seq_init, ctx.free_mask, temperature=max(crit.temperature, 1e-3),
                max_iters=1000, convergence_mode=True, valid_aa_mask=valid_mask)
            seqs = [seq_opt[i] for i in range(N)]
        else:  # mpnn_sample
            seqs = list(ctx.sample_decoder(N))
        out = PHDesignSet()
        for s, seq in enumerate(seqs):
            out.append(self._score_design(ctx, crit, seq, scheme=crit.scheme_label(), sample=s,
                                          field_fn=field_fn, valid_mask=valid_mask))
        return out

    @staticmethod
    def _n_placement_seeds(ctx, crit) -> int:
        if crit.seed_source == "inverse":
            return len(ctx.external_initial_sequences)
        return 1

    @staticmethod
    def _placement_initial(ctx, crit, seq_i):
        if crit.seed_source == "inverse":
            if not ctx.external_initial_sequences:
                raise ValueError("seed_source='inverse' but no initial_sequences provided.")
            return ctx.encode_initial_sequence(ctx.external_initial_sequences[seq_i]), True
        return ctx.S_native, False

    def _run_placement_one_seed(self, ctx, crit, field_fn, field_at, valid_mask, seed, seq_i,
                                initial_sequence, seeded) -> PHDesignSet:
        rng = random.Random(seed * 100003 + seq_i * 7919)
        out = PHDesignSet()
        seed_idx = seq_i if seeded else None
        P_init = _cond_dist(field_fn(initial_sequence), valid_mask, crit.temperature)
        plans = self.enumerate_placement_plans(ctx, crit, field_fn, valid_mask, initial_sequence,
                                               rng=rng)
        for plan in plans:
            if not plan.designable:
                continue
            for s in range(crit.samples_per_site):
                trng.manual_seed(seed * 100003 + seq_i * 7919 + plan.seed_key + s)
                seq, traj = self._design_one(ctx, crit, field_fn, field_at, valid_mask, plan, rng,
                                             initial_sequence, sample=s)
                out.append(self._score_design(
                    ctx, crit, seq, scheme=crit.scheme_label(), sample=s, field_fn=field_fn,
                    valid_mask=valid_mask, seed_idx=seed_idx, plan=plan, P_native=P_init,
                    energy_trajectory=traj))
        return out

    # --- placement plans -----------------------------------------------------
    def _pin_idxs(self, ctx, crit, ptype):
        t2i = ctx.encoding.token_to_idx
        if ptype not in t2i:
            raise KeyError(
                f"centre protonation token {ptype!r} is not in the model vocabulary {sorted(t2i)}.")
        deps = crit.dep_map.get(ptype)
        if not deps:
            raise KeyError(f"dep_map has no contrast tokens for centre {ptype!r}.")
        missing = [t for t in deps if t not in t2i]
        if missing:
            raise KeyError(f"dep_map[{ptype!r}] references tokens {missing} absent from the vocabulary.")
        return t2i[ptype], [t2i[t] for t in deps]

    def _pin(self, ctx, crit, position, ptype) -> PlacementPin:
        prot_idx, dep_idxs = self._pin_idxs(ctx, crit, ptype)
        return PlacementPin(position=int(position), protonation_type=ptype, prot_idx=prot_idx,
                            dep_idxs=dep_idxs, res_id=int(ctx.token_res_id[int(position)]))

    def enumerate_placement_plans(self, ctx, crit, field_fn, valid_mask, initial_sequence, *, rng):
        if crit.center_count == 0:
            free = set(int(x) for x in ctx.chA_free_idx.tolist())
            if crit.placement_region and "all" not in crit.placement_region:
                region = set(int(x) for x in np.nonzero(
                    ctx.region_mask(crit.placement_region))[0].tolist())
                free = free & region
            designable = sorted(free)
            if not designable:
                return []
            return [PlacementPlan(pins=[], designable=designable, label="")]

        if crit.explicit_centers:
            pins = [self._pin(ctx, crit, ctx.binder_pos_of_res_id(int(c["res_id"])),
                              str(c["protonation_type"])) for c in crit.explicit_centers]
            plan = _finalize_plan(ctx, crit, pins)
            return [plan] if plan is not None else []

        if crit.center_types:
            region_idx = np.nonzero(ctx.region_mask(crit.placement_region))[0]
            if region_idx.size < len(crit.center_types):
                return []
            place = placement_fn(crit.placement_by)
            used, pins = set(), []
            for t in crit.center_types:
                prot_idx, dep_idxs = self._pin_idxs(ctx, crit, t)
                ranked = place(ctx, crit, region_idx, prot_idx, dep_idxs, initial_sequence, rng)
                pos = next((p for p, _ in ranked if int(p) not in used), None)
                if pos is None:
                    return []
                used.add(int(pos))
                pins.append(self._pin(ctx, crit, int(pos), t))
            plan = _finalize_plan(ctx, crit, pins)
            return [plan] if plan is not None else []

        region_idx = np.nonzero(ctx.region_mask(crit.placement_region))[0]
        if region_idx.size < crit.center_count:
            return []
        place = placement_fn(crit.placement_by)
        pool = []
        for t in crit.center_protonation_types:
            prot_idx, dep_idxs = self._pin_idxs(ctx, crit, t)
            ranked = place(ctx, crit, region_idx, prot_idx, dep_idxs, initial_sequence, rng)
            pool.extend((pos, t) for pos, _ in ranked[: max(1, crit.candidate_pool)])
        combos = _combos_or_sample(pool, crit.center_count, max(1, crit.n_plan_samples), rng)

        rank = crit.placement_by != "random"
        place_field = ctx.field_mpnn if crit.placement_by == "scan_mpnn" else ctx.field_potts
        cands = []
        for combo in combos:
            pins = [self._pin(ctx, crit, pos, t) for pos, t in combo]
            score = _locked_selective_score(ctx, place_field, initial_sequence, pins) if rank else 0.0
            cands.append((score, pins))
        if rank:
            cands.sort(key=lambda x: x[0])
        plans = []
        for score, pins in cands[: (crit.max_plans_per_seed or len(cands))]:
            plan = _finalize_plan(ctx, crit, pins)
            if plan is not None:
                plan.placement_score = float(score)
                plans.append(plan)
        return plans

    # --- one design ----------------------------------------------------------
    def _design_one(self, ctx, crit, field_fn, field_at, valid_mask, plan, rng, initial_sequence, *,
                    sample):
        trajectory = [] if crit.record_trajectory else None
        pins, neigh = plan.pins, plan.designable
        method = crit.method
        if method == "autoregressive":
            from ppjax.decoder import masked_infill
            order = rng.sample(neigh, len(neigh))
            seq = masked_infill(ctx, crit, valid_mask, pins, order, initial_sequence,
                                trajectory=trajectory)
        elif method == "two_phase":
            seq = _two_phase(ctx, crit, field_at, valid_mask, pins, neigh, initial_sequence,
                             trajectory=trajectory)
        elif method == "block_descent":
            seq = _block_descent(ctx, crit, valid_mask, pins, neigh, initial_sequence, rng,
                                 trajectory=trajectory)
        elif method == "greedy_energy_block":
            seq = _greedy_energy_block(ctx, crit, valid_mask, pins, neigh, initial_sequence, rng,
                                       trajectory=trajectory)
        else:
            seq = _converge(ctx, crit, field_at, valid_mask, pins, neigh, initial_sequence,
                            trajectory=trajectory)
        return seq, trajectory

    def _score_design(self, ctx, crit, seq, *, scheme, sample, field_fn, valid_mask,
                      seed_idx=None, plan=None, P_native=None,
                      energy_trajectory=None) -> PHDesignOutput:
        binder_idx = ctx.chainA_all_idx.tolist()
        canonical = ctx.decode_canonical(seq, binder_idx)
        ext = ctx.extended_tokens(seq, binder_idx)
        final_e = ctx.scorer.H_of(seq)

        sel_e = glob = pl_prob = pl_ent = None
        n_neigh = center_res = proto = None
        n_centers = center_res_ids = center_types = sel_each = None
        score_positions = binder_idx
        if plan is not None:
            pins = plan.pins
            n_neigh = len(plan.designable)
            proto = plan.label
            n_centers = len(pins)
            glob = _global_protonation_dH(ctx, seq)
            score_positions = plan.designable or binder_idx
            if pins:
                primary = pins[0]
                sel_each = [_selective_energy_of(ctx, seq, p.position, p.prot_idx, p.dep_idxs)
                            for p in pins]
                sel_e = float(sum(sel_each))
                valid_idx = np.nonzero(valid_mask)[0]
                vp = {int(t): i for i, t in enumerate(valid_idx.tolist())}
                if P_native is not None and primary.prot_idx in vp:
                    pl_prob = float(P_native[primary.position, vp[primary.prot_idx]])
                    pl_ent = float(_entropy_bits(P_native[primary.position]))
                center_res = primary.res_id
                center_res_ids = [p.res_id for p in pins]
                center_types = [p.protonation_type for p in pins]
            else:
                sel_each, sel_e = [], 0.0

        dec = _decoded_prob_score(ctx, crit, field_fn, valid_mask, seq, score_positions)
        seq_entropy = _seq_entropy_bits(ctx.decode_canonical(seq, score_positions))

        return PHDesignOutput(
            binder_chain=ctx.binder_chain, canonical_sequence=canonical, extended_tokens=ext,
            extended_vocab=ctx.extended_vocab, scheme=scheme, sample=sample, seed_idx=seed_idx,
            final_potts_energy=final_e, protonation_type=proto, site_rank=None,
            center_res_id=center_res, n_neighbours=n_neigh, selective_energy=sel_e,
            n_centers=n_centers, center_res_ids=center_res_ids,
            center_protonation_types=center_types, selective_energies=sel_each,
            global_protonation_dH=glob, placement_prob=pl_prob, placement_entropy=pl_ent,
            sequence_decoded_prob_score=dec, sequence_entropy=seq_entropy,
            method=crit.method, backend=crit.backend, selective_source=crit.selective_source,
            placement_label=crit.placement_label,
            placement_region="+".join(crit.placement_region), placement_by=crit.placement_by,
            combined_lambda=crit.combined_lambda,
            repetitive_window_weight=crit.repetitive_window_weight, neighbour_k=crit.neighbour_k,
            max_mutations=crit.max_mutations, block_size=crit.block_size,
            sweep_order=crit.sweep_order, energy_trajectory=energy_trajectory)
