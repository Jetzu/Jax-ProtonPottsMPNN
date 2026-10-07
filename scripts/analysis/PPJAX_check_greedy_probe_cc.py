# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX.
# Purpose: test the hypothesis for the one diverging reference case - that greedy_energy_block
#          diverges because of the NUMBER of sequential Boltzmann draws it makes, not because the
#          algorithm is mis-ported. If the T=0 (argmin) variant matches and the T>0 variants do not,
#          the mechanism is the sampling.
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np

from ppjax.paths import resolve_checkpoint
from ppjax.criteria import PHDesignCriteria
from ppjax.engine import PottsMPNNPHEngine

REF = Path("tests/data/ref_greedy/PPJAX_greedy_probe.json")
cases = json.loads(REF.read_text())
meta = json.loads(Path("tests/data/ref_pdl1/PPJAX_reference_meta.json").read_text())
d = np.load("tests/data/ref_suite/PPJAX_ctx_pdl1.npz")
eng = PottsMPNNPHEngine.from_checkpoint(resolve_checkpoint(meta.get("ckpt")), extended_vocab="v6")
ctx = eng.build_context({k[3:]: d[k] for k in d.files if k.startswith("ni_")}, "A",
                        token_res_id=d["token_res_id"], token_res_name=d["token_res_name"],
                        token_chain_id=d["token_chain_id"],
                        region_masks={n: d[f"region_{n}"] for n in
                                      ("interface", "core", "surface") if f"region_{n}" in d.files})
print(f"{'case':22s} {'T':>6s} {'designs':>7s}  verdict")
for name, blob in cases.items():
    if "error" in blob:
        print(f"{name:22s} torch errored: {blob['error'][:50]}"); continue
    crit = PHDesignCriteria(**{k: v for k, v in blob["criteria"].items()
                               if k in PHDesignCriteria.__dataclass_fields__})
    ds = eng.run_ph_redesign(ctx=ctx, criteria_list=[crit], seed=0)
    got = {x.design_id(): x for x in ds}
    ref = {r["design_id"]: r for r in blob["designs"]}
    ndiff, ok = 0, set(got) == set(ref)
    for did, r in ref.items():
        g = got.get(did)
        if g is None:
            ok = False; continue
        if " ".join(g.extended_tokens) != r["extended_tokens"]:
            ok = False
            ndiff += sum(a != b for a, b in zip(r["extended_tokens"].split(), g.extended_tokens))
    print(f"{name:22s} {crit.temperature:6.3f} {len(ds):7d}  "
          f"{'IDENTICAL' if ok else f'DIFFERS ({ndiff} tokens)'}")
