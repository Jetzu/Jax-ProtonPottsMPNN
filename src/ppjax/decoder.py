# Origin: cc (Claude Code, 2026-10-02) - ported from mpnn.model.mpnn (setup_causality_masks,
#         repeat_along_batch, decode_setup, logits_to_sample, decode_teacher_forcing,
#         decode_auto_regressive) and mpnn.inference_engines.potts_mpnn_ph (_masked_infill,
#         _selective_row, _selective_reward_decoder, _sampler_zscales). Upstream read-only.
# Purpose: the decoder half of the generation path - the MPNN log-probability field
#          (backend="mpnn"), autoregressive sampling (method="mpnn_sample") and the masked-infill
#          designer (method="autoregressive").

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ppjax import rng as trng
from ppjax.layers import cat_neighbors_nodes, dec_layer, gather_nodes
from ppjax.nn import embedding, linear, one_hot

DECODING_EPS = 1e-4
CAUSALITY_PATTERNS = ("auto_regressive", "unconditional", "conditional", "conditional_minus_self")


def sample_decoding_order(decode_last_mask: np.ndarray, gen=None,
                          symmetry_equivalence_group=None) -> np.ndarray:
    """``setup_causality_masks``' decoding order: ``argsort((mask + eps) * |randn|)``.

    The noise is drawn from the torch-compatible generator so the stream stays aligned with the
    original; see the note in :mod:`ppjax.rng` about the ~1e-6 libm difference in randn VALUES.

    ``symmetry_equivalence_group`` is NOT supported: upstream draws ``randn((B, G))`` per symmetry
    group rather than ``randn((B, L))`` (model/mpnn.py), so both the order and every later draw in
    the stream would diverge. Nothing in this pipeline sets it, and a silent mismatch here is
    exactly the kind of divergence this port exists to avoid - so it raises.
    """
    if symmetry_equivalence_group is not None:
        raise NotImplementedError(
            "symmetry_equivalence_group is not ported: upstream draws the decoding-order noise "
            "per symmetry GROUP (randn((B, G))), not per residue, so the RNG stream would "
            "diverge. Run this input through the torch engine instead.")
    gen = gen or trng.default_generator()
    B, L = decode_last_mask.shape
    noise = gen.normal_float32(B * L).reshape(B, L)
    key = (decode_last_mask.astype(np.float32) + np.float32(DECODING_EPS)) * np.abs(noise)
    return np.argsort(key, axis=-1, kind="stable")


def causality_masks(decode_last_mask: np.ndarray, residue_mask: np.ndarray, E_idx: np.ndarray,
                    causality_pattern: str, decoding_order: np.ndarray):
    """``causal_mask`` / ``anti_causal_mask`` [B, L, K, 1] for the chosen causality pattern."""
    if causality_pattern not in CAUSALITY_PATTERNS:
        raise ValueError(f"Unknown causality pattern: {causality_pattern}")
    B, L = decode_last_mask.shape
    perm_rev = np.eye(L, dtype=np.float32)[decoding_order]           # [B, L, L]
    if causality_pattern == "auto_regressive":
        base = 1.0 - np.triu(np.ones((L, L), dtype=np.float32))
    elif causality_pattern == "unconditional":
        base = np.zeros((L, L), dtype=np.float32)
    elif causality_pattern == "conditional":
        base = np.ones((L, L), dtype=np.float32)
    else:
        base = np.ones((L, L), dtype=np.float32) - np.eye(L, dtype=np.float32)
    permuted = np.einsum("ij,biq,bjp->bqp", base, perm_rev, perm_rev)
    gathered = np.take_along_axis(permuted, E_idx, axis=2)[..., None]  # [B, L, K, 1]
    rm = residue_mask.astype(np.float32).reshape(B, L, 1, 1)
    return gathered * rm, (1.0 - gathered) * rm


def _logits_to_sample(params, logits, temperature, unknown_indices, gen=None):
    """``logits_to_sample`` with ``bias``/``pair_bias`` None (what this pipeline ever supplies)."""
    gen = gen or trng.default_generator()
    logits = np.asarray(logits, dtype=np.float32)
    B, L, V = logits.shape
    if temperature is not None:
        logits = logits / np.asarray(temperature, np.float32)[..., None]
    m = logits.max(-1, keepdims=True)
    e = np.exp((logits - m).astype(np.float32), dtype=np.float32)
    ssum = e.sum(-1, keepdims=True, dtype=np.float32)
    probs = (e / ssum).astype(np.float32)
    log_probs = ((logits - m) - np.log(ssum, dtype=np.float32)).astype(np.float32)
    probs_sample = probs.copy()
    probs_sample[:, :, list(unknown_indices)] = 0.0
    probs_sample = (probs_sample / probs_sample.sum(-1, keepdims=True, dtype=np.float32)
                    ).astype(np.float32)
    S_sampled = gen.multinomial_rows(probs_sample.reshape(B * L, V)).reshape(B, L)
    S_argmax = np.argmax(probs_sample, axis=-1)
    return {"log_probs": log_probs, "probs": probs, "probs_sample": probs_sample,
            "S_sampled": S_sampled, "S_argmax": S_argmax}


@jax.jit
def _decoder_stack(params, h_V_enc, h_E, E_idx, h_S, causal_mask, anti_causal_mask,
                   residue_mask, mask_E):
    """Teacher-forced decoder: every layer sees the causal mixture of decoder and encoder edges."""
    h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_V_enc), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)
    h_EXV_anti = h_EXV_encoder * anti_causal_mask
    h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
    h_V = h_V_enc
    for layer in params["decoder_layers"]:
        h_ESV_dec = cat_neighbors_nodes(h_V, h_ES, E_idx)
        h_ESV = causal_mask * h_ESV_dec + h_EXV_anti
        h_V = dec_layer(layer, h_V, h_ESV, mask_V=residue_mask, mask_E=mask_E)
    return linear(params["W_out"], h_V), h_V


class DecoderField:
    """The decoder-side callables the design engine needs, bound to one featurised backbone.

    ``log_probs(S)`` runs the teacher-forced ``conditional_minus_self`` decode the torch engine
    uses for its design field; ``sample(n)`` runs the autoregressive decode (``mpnn_sample``).
    Both consume the torch RNG stream in the same places and amounts as the original.
    """

    def __init__(self, params, cfg, unknown_indices, h_V, h_E, E_idx, residue_mask,
                 designed_residue_mask, temperature=None, token_to_idx=None,
                 symmetry_equivalence_group=None) -> None:
        self.symmetry_group = symmetry_equivalence_group
        self.params = params
        self.cfg = cfg
        self.unknown_indices = list(unknown_indices)
        self.h_V = jnp.asarray(h_V)
        self.h_E = jnp.asarray(h_E)
        self.E_idx_np = np.asarray(E_idx)
        self.E_idx = jnp.asarray(self.E_idx_np, jnp.int32)
        self.residue_mask = np.asarray(residue_mask).astype(bool)
        self.designed_residue_mask = np.asarray(designed_residue_mask).astype(bool)
        self.temperature = None if temperature is None else np.asarray(temperature, np.float32)
        self.token_to_idx = token_to_idx or {}
        rm = self.residue_mask.astype(np.float32)
        B, L, K = self.E_idx_np.shape
        nbr_rm = np.take_along_axis(np.broadcast_to(rm[:, None, :], (B, L, L)),
                                    self.E_idx_np, axis=2)      # [B, L, K]
        self.mask_E = jnp.asarray(rm[:, :, None] * nbr_rm)
        self.decode_last_mask = self.residue_mask & self.designed_residue_mask

    # --- the design field ----------------------------------------------------
    def log_probs(self, S: np.ndarray, causality_pattern: str = "conditional_minus_self",
                  initialize_with_ground_truth: bool = True) -> np.ndarray:
        """[L, V] decoder log-probabilities for sequence ``S`` (B = 1)."""
        S2 = np.asarray(S, dtype=np.int64).reshape(1, -1)
        order = sample_decoding_order(self.decode_last_mask,
                                      symmetry_equivalence_group=self.symmetry_group)
        causal, anti = causality_masks(self.decode_last_mask, self.residue_mask, self.E_idx_np,
                                       causality_pattern, order)
        h_S = (embedding(self.params["W_s"], jnp.asarray(S2, jnp.int32))
               if initialize_with_ground_truth else jnp.zeros_like(self.h_V))
        logits, _ = _decoder_stack(self.params, self.h_V, self.h_E, self.E_idx, h_S,
                                   jnp.asarray(causal), jnp.asarray(anti),
                                   jnp.asarray(self.residue_mask, jnp.float32), self.mask_E)
        out = _logits_to_sample(self.params, np.asarray(logits), self.temperature,
                                self.unknown_indices)
        return out["log_probs"][0]

    def field(self, S: np.ndarray) -> np.ndarray:
        """``field_mpnn``: the decoder design field, an ENERGY (``-log p``, lower = better)."""
        return -self.log_probs(S)

    # --- autoregressive sampling --------------------------------------------
    def sample(self, S_native: np.ndarray, n: int,
               causality_pattern: str = "auto_regressive",
               initialize_with_ground_truth: bool = False) -> np.ndarray:
        """``decode_auto_regressive`` for ``repeat_sample_num = n`` -> [n, L] sampled sequences."""
        L = self.E_idx_np.shape[1]
        S = np.broadcast_to(np.asarray(S_native, np.int64).reshape(1, -1), (n, L)).copy()
        residue_mask = np.broadcast_to(self.residue_mask[0], (n, L)).copy()
        decode_last = np.broadcast_to(self.decode_last_mask[0], (n, L)).copy()
        E_idx = np.broadcast_to(self.E_idx_np[0], (n, L, self.E_idx_np.shape[2])).copy()
        temperature = (None if self.temperature is None
                       else np.broadcast_to(self.temperature[0], (n, L)).copy())
        h_V_enc = jnp.broadcast_to(self.h_V[0], (n,) + self.h_V.shape[1:])
        h_E = jnp.broadcast_to(self.h_E[0], (n,) + self.h_E.shape[1:])

        order = sample_decoding_order(decode_last,
                                      symmetry_equivalence_group=self.symmetry_group)
        causal, anti = causality_masks(decode_last, residue_mask, E_idx, causality_pattern, order)

        rm = residue_mask.astype(np.float32)
        mask_E = jnp.asarray(rm[:, :, None] * np.take_along_axis(
            np.broadcast_to(rm[:, None, :], (n, L, L)), E_idx, axis=2))

        unk = self.token_to_idx.get("UNK", self.unknown_indices[0])
        S_sampled = (S.copy() if initialize_with_ground_truth
                     else np.full((n, L), unk, dtype=np.int64))
        h_S = np.array(embedding(self.params["W_s"], jnp.asarray(S, jnp.int32))
                       if initialize_with_ground_truth else jnp.zeros_like(h_V_enc))
        stack = [np.array(h_V_enc)] + [np.zeros(np.shape(h_V_enc), dtype=np.float32)
                                       for _ in self.params["decoder_layers"]]
        h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_V_enc), h_E, jnp.asarray(E_idx))
        h_EXV_anti = np.array(
            cat_neighbors_nodes(h_V_enc, h_EX_encoder, jnp.asarray(E_idx))) * anti

        batch = np.arange(n)
        for step in range(L):
            i = order[:, step]
            S_i = S[batch, i]
            dlm_i = decode_last[batch, i]
            logits_i = _decode_one_position(
                self.params, stack, h_S, h_E, E_idx, h_EXV_anti, causal, residue_mask,
                np.asarray(mask_E), batch, i)
            temp_i = None if temperature is None else temperature[batch, i][:, None]
            sd = _logits_to_sample(self.params, logits_i[:, None, :], temp_i,
                                   self.unknown_indices)
            smp = sd["S_sampled"][:, 0]
            S_sampled[batch, i] = smp * dlm_i + S_i * (~dlm_i)
            h_S[batch, i] = np.asarray(embedding(
                self.params["W_s"], jnp.asarray(S_sampled[batch, i], jnp.int32)))
        return S_sampled


def _decode_one_position(params, stack, h_S, h_E, E_idx, h_EXV_anti, causal, residue_mask,
                         mask_E, batch, i):
    """One autoregressive step: run every decoder layer for the single position ``i`` per batch."""
    E_idx_i = E_idx[batch, i]                                      # [B, K]
    h_E_i = h_E[batch, i]                                          # [B, K, H]
    h_ES_i = np.asarray(cat_neighbors_nodes(
        jnp.asarray(h_S), jnp.asarray(h_E_i)[:, None], jnp.asarray(E_idx_i)[:, None]))
    anti_i = h_EXV_anti[batch, i][:, None]
    causal_i = causal[batch, i][:, None]
    mask_E_i = mask_E[batch, i][:, None]
    rm_i = residue_mask[batch, i].astype(np.float32)[:, None]
    for layer_idx, layer in enumerate(params["decoder_layers"]):
        h_ESV_dec = np.asarray(cat_neighbors_nodes(
            jnp.asarray(stack[layer_idx]), jnp.asarray(h_ES_i), jnp.asarray(E_idx_i)[:, None]))
        h_ESV = causal_i * h_ESV_dec + anti_i
        h_V_i = np.asarray(dec_layer(
            layer, jnp.asarray(stack[layer_idx][batch, i])[:, None], jnp.asarray(h_ESV),
            mask_V=jnp.asarray(rm_i), mask_E=jnp.asarray(mask_E_i)))
        stack[layer_idx + 1][batch, i] = h_V_i[:, 0]
    return np.asarray(linear(params["W_out"], jnp.asarray(stack[-1][batch, i])))


# ---------------------------------------------------------------------------
# masked-infill designer (method="autoregressive")
# ---------------------------------------------------------------------------

def _selective_row(ctx, valid_idx, S, j, pins) -> np.ndarray:
    from ppjax.engine import _selective_energy_sum
    out = np.full(ctx.V, np.float32(np.inf), dtype=np.float32)
    cur = int(S[j])
    for a in valid_idx:
        S[j] = a
        out[a] = _selective_energy_sum(ctx, S, pins)
    S[j] = cur
    return out


def _selective_reward_decoder(ctx, valid_idx, S, j, pins, lam) -> np.ndarray:
    """Pure-decoder two-state contrast ``p(a | centres TARGET) - lam * p(a | centres OFF)``."""
    p_target = np.exp(-ctx.field_mpnn(S)[j], dtype=np.float32)
    n_off = max(len(p.dep_idxs) for p in pins)
    p_off = np.zeros(ctx.V, dtype=np.float32)
    saved = [int(S[p.position]) for p in pins]
    for t in range(n_off):
        for p in pins:
            S[p.position] = p.dep_idxs[min(t, len(p.dep_idxs) - 1)]
        p_off += np.exp(-ctx.field_mpnn(S)[j], dtype=np.float32)
    for p, tok in zip(pins, saved):
        S[p.position] = tok
    reward = p_target - np.float32(lam) * (p_off / np.float32(n_off))
    out = np.full(ctx.V, np.float32(-np.inf), dtype=np.float32)
    out[np.asarray(valid_idx, dtype=np.int64)] = reward[np.asarray(valid_idx, dtype=np.int64)]
    return out


def _sampler_zscales(ctx, S, order, pins, valid_idx):
    """Freeze the std of the sampler's two single-mutation delta distributions."""
    from ppjax.engine import _selective_energy_sum
    Sloc = S.copy()
    for p in pins:
        Sloc[p.position] = p.prot_idx
    nat_full = ctx.field_mpnn(Sloc)
    sel0 = _selective_energy_sum(ctx, Sloc, pins)
    nat_d, sel_d = [], []
    for j in order:
        cur = int(Sloc[j])
        base = float(nat_full[j, cur])
        for a in valid_idx:
            nat_d.append(float(nat_full[j, a]) - base)
            Sloc[j] = a
            sel_d.append(_selective_energy_sum(ctx, Sloc, pins) - sel0)
        Sloc[j] = cur
    sdNat = max(float(np.std(nat_d)) if nat_d else 1.0, 1e-6)
    sdSel = max(float(np.std(sel_d)) if sel_d else 1.0, 1e-6)
    return sdNat, sdSel


def masked_infill(ctx, crit, valid_mask, pins, order, initial_sequence, trajectory=None):
    """``_masked_infill``: lock the centres, mask the neighbourhood to UNK, decode one by one."""
    from ppjax.engine import _pick, _record
    lam = float(crit.combined_lambda)
    valid_idx = np.nonzero(valid_mask)[0].tolist()
    S = initial_sequence.copy()
    for p in pins:
        S[p.position] = p.prot_idx
    decoder_sel = (crit.selective_source == "decoder")
    if not decoder_sel:
        sdNat, sdSel = _sampler_zscales(ctx, S, order, pins, valid_idx)
    unk = ctx.encoding.token_to_idx["UNK"]
    for i in order:
        S[i] = unk
    _record(trajectory, ctx, S, pins, initial=True)
    T = max(float(crit.temperature), 1e-3)
    for i in order:
        if decoder_sel:
            Jv = -_selective_reward_decoder(ctx, valid_idx, S, i, pins, lam)
        else:
            nat = ctx.field_mpnn(S)[i]
            sel = _selective_row(ctx, valid_idx, S, i, pins)
            Jv = (np.float32(1.0 - lam) * (nat / np.float32(sdNat))
                  + np.float32(lam) * (sel / np.float32(sdSel)))
        S[i] = _pick(np.where(~valid_mask, np.float32(np.inf), Jv), T)
        _record(trajectory, ctx, S, pins)
    return S
