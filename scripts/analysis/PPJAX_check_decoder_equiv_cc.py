# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX.
# Purpose: compare the ported decoder (design field, autoregressive sampling, mpnn-backend design
#          methods) against the torch golden outputs.
import sys, json
from pathlib import Path

FAILURES = []
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np

from ppjax.paths import resolve_checkpoint

from ppjax import rng as trng
from ppjax.engine import PottsMPNNPHEngine
from ppjax.criteria import PHDesignCriteria

REFC = Path("tests/data/ref_pdl1")
REFD = Path("tests/data/ref_decoder")
meta = json.loads((REFC / "PPJAX_reference_meta.json").read_text())
d = np.load(REFC / "PPJAX_reference_ctx.npz")
dec = np.load(REFD / "PPJAX_decoder_ref.npz")

eng = PottsMPNNPHEngine.from_checkpoint(resolve_checkpoint(meta.get("ckpt")), extended_vocab="v6")
feats = {k[3:]: d[k] for k in d.files if k.startswith("ni_")}
ctx = eng.build_context(feats, meta["binder_chain"], token_res_id=d["token_res_id"],
                        token_res_name=d["token_res_name"], token_chain_id=d["token_chain_id"],
                        region_masks={n: d[f"region_{n}"] for n in
                                      ("interface", "core", "surface") if f"region_{n}" in d.files})

def cmp(name, got, want, tol=1e-2):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    err = np.abs(got - want)
    ok = err.max() < tol
    if not ok:
        FAILURES.append(f"{name}: max_abs {err.max():.3e} >= {tol}")
    print(f"  {name:22s} max_abs={err.max():.3e}  ref_absmax={np.abs(want).max():.3e}  "
          f"{'ok' if ok else 'FAIL'}")

print("decoder field (conditional_minus_self, teacher forced):")
trng.manual_seed(0); cmp("field_mpnn(native)", ctx.field_mpnn(ctx.S_native), dec["field_mpnn_native"])
trng.manual_seed(0); cmp("field_mpnn(one UNK)", ctx.field_mpnn(dec["S_mpnn_unk_input"]),
                         dec["field_mpnn_unk"])

print("autoregressive sampling (mpnn_sample, 4 samples, seed 0):")
trng.manual_seed(0)
S = ctx.sample_decoder(4)
ref_S = dec["mpnn_sample_seed0"]
n_diff = int((S != ref_S).sum())
if n_diff:
    FAILURES.append(f"mpnn_sample: {n_diff}/{ref_S.size} tokens differ")
print(f"  sampled tokens identical: {n_diff == 0}  ({n_diff}/{ref_S.size} differ)")
for r in range(ref_S.shape[0]):
    nd = int((S[r] != ref_S[r]).sum())
    print(f"    sample {r}: {nd} token(s) differ")

print("mpnn-backend design methods:")
cases = json.loads((REFD / "PPJAX_decoder_ref.json").read_text())
for name, blob in cases.items():
    if "error" in blob:
        print(f"  {name:28s} torch errored: {blob['error'][:60]}")
        continue
    crit = PHDesignCriteria(**{k: v for k, v in blob["criteria"].items()
                               if k in PHDesignCriteria.__dataclass_fields__})
    try:
        ds = eng.run_ph_redesign(ctx=ctx, criteria_list=[crit], seed=0)
    except Exception as e:
        FAILURES.append(f"{name}: JAX raised {type(e).__name__}: {e}")
        print(f"  {name:28s} JAX ERROR {type(e).__name__}: {e}")
        continue
    ref_by = {r["design_id"]: r for r in blob["designs"]}
    got_by = {x.design_id(): x for x in ds}
    ok = set(ref_by) == set(got_by)
    ndiff = 0
    for did, r in ref_by.items():
        g = got_by.get(did)
        if g is None:
            ok = False; continue
        gt = " ".join(g.extended_tokens)
        if gt != r["extended_tokens"]:
            ok = False
            ndiff += sum(a != b for a, b in zip(r["extended_tokens"].split(), g.extended_tokens))
    if not ok:
        FAILURES.append(f"{name}: designs differ ({ndiff} tokens)")
    print(f"  {name:28s} {len(ds)} designs  {'IDENTICAL' if ok else f'DIFFERS ({ndiff} tokens)'}")

if FAILURES:
    print("\nFAILED:")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("\nall decoder checks passed")
