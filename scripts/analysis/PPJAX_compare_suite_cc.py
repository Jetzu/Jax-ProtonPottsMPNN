# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX.
# Purpose: replay every case of the torch reference suite through the JAX engine and report, per
#          case, whether the generated sequences are identical (the port's acceptance criterion).
import os, sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np

from ppjax.engine import PottsMPNNPHEngine
from ppjax.criteria import PHDesignCriteria

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="tests/data/ref_suite")
ap.add_argument("--ckpt", default=os.environ.get("PROTON_CKPT") or str(
    Path(os.environ.get("PROTON_ROOT") or (Path.home() / "ProtonPottsMPNN")).expanduser()
    / "checkpoints" / "potts_v6_afdb_edge_his0.3_acid0.06" / "epoch-0125.ckpt"))
ap.add_argument("--only", default=None, help="substring filter on the case name")
ap.add_argument("--json-out", default=None)
args = ap.parse_args()

SUITE = Path(args.suite)
ref = json.loads((SUITE / "PPJAX_reference_suite.json").read_text())
eng = PottsMPNNPHEngine.from_checkpoint(args.ckpt, extended_vocab="v6")

ctx_cache = {}
def get_ctx(structure):
    if structure not in ctx_cache:
        d = np.load(SUITE / f"PPJAX_ctx_{structure}.npz")
        feats = {k[3:]: d[k] for k in d.files if k.startswith("ni_")}
        chain = "A"
        ctx_cache[structure] = (eng.build_context(
            feats, chain, token_res_id=d["token_res_id"], token_res_name=d["token_res_name"],
            token_chain_id=d["token_chain_id"],
            region_masks={n: d[f"region_{n}"] for n in ("interface", "core", "surface")
                          if f"region_{n}" in d.files}), d)
    return ctx_cache[structure]

rows, n_ok, n_cases = [], 0, 0
for case, blob in ref.items():
    if args.only and args.only not in case:
        continue
    if "error" in blob:
        print(f"{case:38s} torch ERRORED: {blob['error'][:70]} - skipped")
        continue
    structure = blob.get("structure", case.split("/")[0])
    ctx, d = get_ctx(structure)
    crits = [PHDesignCriteria(**{k: v for k, v in c.items()
                                 if k in PHDesignCriteria.__dataclass_fields__})
             for c in blob["criteria"]]
    t0 = time.time()
    try:
        ds = eng.run_ph_redesign(ctx=ctx, criteria_list=crits, seed=blob["seed"],
                                 initial_sequences=blob.get("initial_sequences"))
    except Exception as e:
        print(f"{case:38s} JAX ERROR {type(e).__name__}: {e}")
        rows.append({"case": case, "status": "jax_error", "detail": f"{type(e).__name__}: {e}"})
        n_cases += 1
        continue
    ref_by_id = {r["design_id"]: r for r in blob["designs"]}
    got_by_id = {dd.design_id(): dd for dd in ds}
    same_ids = set(ref_by_id) == set(got_by_id)
    mismatch, maxdH, tok_diff = [], 0.0, 0
    for did, r in ref_by_id.items():
        g = got_by_id.get(did)
        if g is None:
            mismatch.append(f"missing:{did}")
            continue
        gt = " ".join(g.extended_tokens)
        if gt != r["extended_tokens"]:
            n = sum(a != b for a, b in zip(r["extended_tokens"].split(), g.extended_tokens))
            tok_diff += n
            mismatch.append(f"{did}:{n}tok")
        maxdH = max(maxdH, abs(g.final_potts_energy - r["potts_energy"]))
    ok = same_ids and not mismatch
    n_ok += ok
    n_cases += 1
    status = "IDENTICAL" if ok else "DIFFERS"
    print(f"{case:38s} {len(ds):3d} designs  {status:10s} maxdH={maxdH:.2e}  "
          f"{time.time()-t0:6.1f}s" + (f"  {mismatch[:3]}" if mismatch else ""))
    rows.append({"case": case, "status": status, "n_designs": len(ds), "max_abs_dH": maxdH,
                 "token_mismatches": tok_diff, "mismatch": mismatch[:10],
                 "same_design_ids": same_ids})

print(f"\n{n_ok}/{n_cases} cases identical")
if args.json_out:
    Path(args.json_out).write_text(json.dumps(rows, indent=1))
