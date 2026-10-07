# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX.
# Purpose: find WHY pdl1/greedy_energy_block is the one reference case whose sequence differs.
#          The method makes two threshold comparisons per step (delta < -1e-4 to enter the
#          improving set; H < best_H - 1e-9 to accept a new best). If either sits closer to its
#          threshold than the port's ~1e-2 energy agreement, the divergence is an inherent
#          sensitivity of the ALGORITHM, not a transcription error.
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np

from ppjax.paths import resolve_checkpoint
from ppjax.criteria import PHDesignCriteria
from ppjax.engine import PottsMPNNPHEngine, _block_mutation_tables, _zscale

SUITE = Path("tests/data/ref_suite")
ref = json.loads((SUITE / "PPJAX_reference_suite.json").read_text())["pdl1/greedy_energy_block"]
crit = PHDesignCriteria(**{k: v for k, v in ref["criteria"][0].items()
                           if k in PHDesignCriteria.__dataclass_fields__})
d = np.load(SUITE / "PPJAX_ctx_pdl1.npz")
meta = json.loads((Path("tests/data/ref_pdl1") / "PPJAX_reference_meta.json").read_text())
eng = PottsMPNNPHEngine.from_checkpoint(resolve_checkpoint(meta.get("ckpt")), extended_vocab="v6")
ctx = eng.build_context({k[3:]: d[k] for k in d.files if k.startswith("ni_")}, "A",
                        token_res_id=d["token_res_id"], token_res_name=d["token_res_name"],
                        token_chain_id=d["token_chain_id"], with_decoder=False)

valid = ctx.valid_aa_mask(crit.forbidden_tokens)
invalid = ~valid
free = sorted(int(x) for x in ctx.chA_free_idx.tolist())
S = ctx.S_native.copy()
cmap = ctx.canonical_map
for i in free:
    ci = int(cmap[int(S[i])])
    if bool(valid[ci]):
        S[i] = ci
sc = ctx.stability_scorer(*crit.stability_weights())
dH0, _, _ = _block_mutation_tables(ctx, valid, S, free, [], False)
wH = 1.0 / _zscale(dH0, valid)
tol, K = 1e-4, max(1, int(crit.block_size))

print(f"designable={len(free)}  block_size={K}  tol={tol}  wH={wH:.6f}")
print("step | n_improving | closest delta to -tol | 2nd-vs-Kth gap | H - best_H")
best_H = sc.H_of(S); step = 0; since = 0
margins, bestmargins, blockgaps = [], [], []
while since < crit.cv_patience * len(free) and step < crit.cv_max * len(free):
    rows = np.asarray(sc.rows_at(S, free), dtype=np.float32)
    improving = []
    deltas = []
    for n, i in enumerate(free):
        e_i = rows[n]; cur = int(S[int(i)])
        ev = e_i.copy(); ev[invalid] = np.float32(np.inf)
        delta = float(ev.min()) - float(e_i[cur])
        deltas.append(delta)
        if delta < -tol:
            improving.append((delta, int(i)))
    if not improving:
        print(f"{step:4d} | converged (no position improves)")
        break
    improving.sort(key=lambda t: (t[0], t[1]))
    margin = min(abs(dd + tol) for dd in deltas)       # closest decision to the -tol threshold
    margins.append(margin)
    gap = (abs(improving[K - 1][0] - improving[K][0]) if len(improving) > K else float("inf"))
    blockgaps.append(gap)
    block = [i for _, i in improving[:K]]
    su, sedges = sc.block_potentials(S, block)
    from ppjax.engine import _broadcast_sum, _pair_broadcast, _softmax32
    from ppjax import rng as trng
    su = np.asarray(su, np.float32); V = ctx.V; B = len(block)
    J = _broadcast_sum([np.float32(wH) * su[b] for b in range(B)], V)
    for (bi, bj, M) in sedges:
        J = J + _pair_broadcast(np.float32(wH) * np.asarray(M, np.float32), bi, bj, B, V)
    infv = np.where(invalid, np.float32(np.inf), np.float32(0.0))
    for b in range(B):
        sh = [1] * B; sh[b] = V; J = J + infv.reshape(sh)
    flat = J.reshape(-1).astype(np.float32)
    assign = list(np.unravel_index(int(np.argmin(flat)), [V] * B))
    for bi, pb in enumerate(block):
        S[pb] = int(assign[bi])
    step += 1
    H = sc.H_of(S)
    bm = abs(H - best_H)
    bestmargins.append(bm)
    if H < best_H - 1e-9:
        best_H = H; since = 0
    else:
        since += 1
    if step <= 12:
        print(f"{step:4d} | {len(improving):11d} | {margin:21.3e} | {gap:14.3e} | {H-best_H:+.4e}")
print(f"\nSMALLEST margin to the -tol improvement threshold over the run : {min(margins):.3e}")
print(f"SMALLEST gap between the K-th and (K+1)-th improving position   : {min(blockgaps):.3e}")
print(f"SMALLEST |H - best_H| at the accept test                        : {min(bestmargins):.3e}")
print(f"port-vs-torch energy agreement on this system                   : ~1e-2 (of 5.2e4)")
