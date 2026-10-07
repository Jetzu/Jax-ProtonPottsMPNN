# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX.
# Purpose: locate the FIRST step at which the ported greedy_energy_block trajectory leaves the
#          torch one, and quantify how close that step's decision was - so the single diverging
#          reference case is characterised, not hand-waved.
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np

from ppjax.paths import resolve_checkpoint
from ppjax.criteria import PHDesignCriteria
from ppjax.engine import PottsMPNNPHEngine

ref = json.loads(Path("tests/data/ref_greedy/PPJAX_greedy_traj.json").read_text())
crit = PHDesignCriteria(**{k: v for k, v in ref["criteria"].items()
                           if k in PHDesignCriteria.__dataclass_fields__})
meta = json.loads(Path("tests/data/ref_pdl1/PPJAX_reference_meta.json").read_text())
d = np.load("tests/data/ref_suite/PPJAX_ctx_pdl1.npz")
eng = PottsMPNNPHEngine.from_checkpoint(resolve_checkpoint(meta.get("ckpt")), extended_vocab="v6")
ctx = eng.build_context({k[3:]: d[k] for k in d.files if k.startswith("ni_")}, "A",
                        token_res_id=d["token_res_id"], token_res_name=d["token_res_name"],
                        token_chain_id=d["token_chain_id"],
                        region_masks={n: d[f"region_{n}"] for n in
                                      ("interface", "core", "surface") if f"region_{n}" in d.files})
ds = eng.run_ph_redesign(ctx=ctx, criteria_list=[crit], seed=0)
jax_traj = ds[0].energy_trajectory
ref_traj = ref["designs"][0]["energy_trajectory"]
print(f"steps: jax={len(jax_traj)} torch={len(ref_traj)}")
first = None
for a, b in zip(jax_traj, ref_traj):
    if a["extended_tokens"] != b["extended_tokens"]:
        first = a["step"]
        na = a["extended_tokens"].split(); nb = b["extended_tokens"].split()
        diffs = [(i, y, x) for i, (x, y) in enumerate(zip(na, nb)) if x != y]
        print(f"FIRST divergence at step {first}: {len(diffs)} token(s) -> {diffs[:5]}")
        print(f"  torch H={b['potts_energy']:.4f}  jax H={a['potts_energy']:.4f}  "
              f"dH={a['potts_energy']-b['potts_energy']:+.4f}")
        prev_a, prev_b = jax_traj[first - 1], ref_traj[first - 1]
        print(f"  step {first-1} identical: {prev_a['extended_tokens'] == prev_b['extended_tokens']}"
              f"  (H dH={prev_a['potts_energy']-prev_b['potts_energy']:+.4e})")
        break
if first is None:
    print("trajectories agree over the shared prefix; lengths:",
          len(jax_traj), len(ref_traj))
else:
    tail_a = jax_traj[-1]["extended_tokens"].split()
    tail_b = ref_traj[-1]["extended_tokens"].split()
    print(f"final sequences differ in {sum(x != y for x, y in zip(tail_a, tail_b))} token(s)")
