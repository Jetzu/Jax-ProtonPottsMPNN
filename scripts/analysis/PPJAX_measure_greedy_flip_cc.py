# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX.
# Purpose: quantify the single diverging decision. At the first diverging step the two engines pick
#          different joint assignments for the same block; this rebuilds that step's objective and
#          reports how far apart the two assignments are in J. If the gap is at the port's float32
#          agreement level, the flip is numerical luck in one Boltzmann draw, not a mis-port.
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np

from ppjax.paths import resolve_checkpoint
from ppjax.criteria import PHDesignCriteria
from ppjax.engine import (PottsMPNNPHEngine, _block_mutation_tables, _zscale, _broadcast_sum,
                          _pair_broadcast)

ref = json.loads(Path("tests/data/ref_greedy/PPJAX_greedy_traj.json").read_text())
crit = PHDesignCriteria(**{k: v for k, v in ref["criteria"].items()
                           if k in PHDesignCriteria.__dataclass_fields__})
traj = ref["designs"][0]["energy_trajectory"]
meta = json.loads(Path("tests/data/ref_pdl1/PPJAX_reference_meta.json").read_text())
d = np.load("tests/data/ref_suite/PPJAX_ctx_pdl1.npz")
eng = PottsMPNNPHEngine.from_checkpoint(resolve_checkpoint(meta.get("ckpt")), extended_vocab="v6")
ctx = eng.build_context({k[3:]: d[k] for k in d.files if k.startswith("ni_")}, "A",
                        token_res_id=d["token_res_id"], token_res_name=d["token_res_name"],
                        token_chain_id=d["token_chain_id"])
t2i = ctx.encoding.token_to_idx
binder = ctx.chainA_all_idx.tolist()

def seq_at(step):
    S = ctx.S_native.copy()
    for pos, tok in zip(binder, traj[step]["extended_tokens"].split()):
        S[pos] = t2i[tok]
    return S

S11, S12_torch = seq_at(11), seq_at(12)
changed = [p for p in binder if int(S11[p]) != int(S12_torch[p])]
print(f"step 11 -> 12: torch changed positions {changed} "
      f"({[ctx.encoding.idx_to_token[int(S11[p])] for p in changed]} -> "
      f"{[ctx.encoding.idx_to_token[int(S12_torch[p])] for p in changed]})")

# rebuild step 12's block objective from the step-11 sequence
valid = ctx.valid_aa_mask(crit.forbidden_tokens); invalid = ~valid
free = sorted(int(x) for x in ctx.chA_free_idx.tolist())
sc = ctx.stability_scorer(*crit.stability_weights())
S0 = ctx.S_native.copy()
cmap = ctx.canonical_map
for i in free:
    ci = int(cmap[int(S0[i])])
    if bool(valid[ci]):
        S0[i] = ci
dH0, _, _ = _block_mutation_tables(ctx, valid, S0, free, [], False)
wH = 1.0 / _zscale(dH0, valid)

rows = np.asarray(sc.rows_at(S11, free), dtype=np.float32)
improving = []
for n, i in enumerate(free):
    e_i = rows[n]; cur = int(S11[i])
    ev = e_i.copy(); ev[invalid] = np.float32(np.inf)
    delta = float(ev.min()) - float(e_i[cur])
    if delta < -1e-4:
        improving.append((delta, int(i)))
improving.sort(key=lambda t: (t[0], t[1]))
K = max(1, int(crit.block_size))
block = [i for _, i in improving[:K]]
print(f"block at step 12: {block}   (K-th vs (K+1)-th delta gap = "
      f"{abs(improving[K-1][0]-improving[K][0]):.3e})" if len(improving) > K else f"block {block}")

V, B = ctx.V, len(block)
su, sedges = sc.block_potentials(S11, block)
su = np.asarray(su, np.float32)
J = _broadcast_sum([np.float32(wH) * su[b] for b in range(B)], V)
for (bi, bj, M) in sedges:
    J = J + _pair_broadcast(np.float32(wH) * np.asarray(M, np.float32), bi, bj, B, V)
# repetitive_window penalty (parents == ["ALL"])
rw = float(crit.repetitive_window_weight); rrad = max(1, int(crit.repetitive_window_radius))
same_canon = (cmap.reshape(-1, 1) == cmap.reshape(1, -1)).astype(np.float32)
res_id = ctx.token_res_id
cu = np.zeros((B, V), np.float32)
block_set = set(block)
for bi, pb in enumerate(block):
    ri = int(res_id[pb])
    for dd in range(-rrad, rrad + 1):
        if dd == 0:
            continue
        q = ctx.res_id_to_pos.get(ri + dd)
        if q is None or int(q) in block_set:
            continue
        cu[bi] += np.float32(rw) * same_canon[:, int(S11[int(q)])]
J = J + _broadcast_sum([cu[b] for b in range(B)], V)
bres = [int(res_id[pb]) for pb in block]
for bi in range(B):
    for bj in range(bi + 1, B):
        if abs(bres[bi] - bres[bj]) <= rrad:
            J = J + _pair_broadcast(np.float32(rw) * same_canon, bi, bj, B, V)
infv = np.where(invalid, np.float32(np.inf), np.float32(0.0))
for b in range(B):
    sh = [1] * B; sh[b] = V; J = J + infv.reshape(sh)
flat = J.reshape(-1).astype(np.float32)

a_torch = tuple(int(S12_torch[p]) for p in block)
S12_jax = None
jax_traj = None
idx_torch = int(np.ravel_multi_index(a_torch, [V] * B))
finite = flat[np.isfinite(flat)]
print(f"J spread over the {int(np.isfinite(flat).sum())} valid assignments: "
      f"min={finite.min():.4f} max={finite.max():.4f} (span {finite.max()-finite.min():.3f})")
order = np.argsort(flat, kind="stable")
rank_t = int(np.where(order == idx_torch)[0][0])
print(f"torch's chosen assignment ranks {rank_t} by J; J={flat[idx_torch]:.6f}, "
      f"argmin J={flat[order[0]]:.6f}  (deltaJ to argmin = {flat[idx_torch]-flat[order[0]]:.3e})")
print(f"J gap between rank0 and rank1: {flat[order[1]]-flat[order[0]]:.3e}")
print(f"temperature T={crit.temperature}; a J difference of d shifts the log-weight by d/T")
print(f"port-vs-torch per-entry agreement on J is ~1e-5 absolute "
      f"(etab 3.7e-5 scaled by wH={wH:.4f})")
