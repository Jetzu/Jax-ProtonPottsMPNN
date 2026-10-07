# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX.
# Purpose: run the JAX engine on the exported reference backbone with the manuscript block-descent
#          criteria and compare every design against the torch golden outputs.
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np

from ppjax.paths import resolve_checkpoint
from ppjax.engine import PottsMPNNPHEngine
from ppjax.criteria import PHDesignCriteria

REF = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/data/ref_pdl1")
meta = json.loads((REF / "PPJAX_reference_meta.json").read_text())
d = np.load(REF / "PPJAX_reference_ctx.npz")

eng = PottsMPNNPHEngine.from_checkpoint(resolve_checkpoint(meta.get("ckpt")), extended_vocab="v6")
feats = {k[3:]: d[k] for k in d.files if k.startswith("ni_")}
t0 = time.time()
ctx = eng.build_context(
    feats, meta["binder_chain"],
    token_res_id=d["token_res_id"], token_res_name=d["token_res_name"],
    token_chain_id=d["token_chain_id"],
    region_masks={n: d[f"region_{n}"] for n in ("interface", "core", "surface")
                  if f"region_{n}" in d.files})
print(f"context built in {time.time()-t0:.1f}s | L={ctx.L} V={ctx.V} K={ctx.K} "
      f"H_native={ctx.scorer.H_of(ctx.S_native):.4f} (torch {float(d['H_native']):.4f})")

crit = PHDesignCriteria(
    method="block_descent", backend="potts", temperature=0.05, samples_per_site=2,
    block_size=3, combined_lambda=0.3, seed_source="native",
    center_types=["HIS-P", "ASP-P", "GLU-P"],
    dep_map={"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]},
    forbidden_tokens=["HIS-A", "ASP-A", "GLU-A", "UNK"], placement_by="scan_potts",
    placement_region=["all"], repetitive_window_parents=["ARG", "LYS", "HIS", "ASP", "GLU"],
    repetitive_window_radius=2, repetitive_window_weight=1.0, neighbour_k=16,
    max_mutations=20, record_trajectory=True)

t0 = time.time()
ds = eng.run_ph_redesign(ctx=ctx, criteria_list=[crit], seed=0)
print(f"produced {len(ds)} designs in {time.time()-t0:.1f}s")

ref = json.loads((REF / "PPJAX_reference_designs.json").read_text())
ref_by_id = {r["design_id"]: r for r in ref}
ok = True
for dd in ds:
    did = dd.design_id()
    r = ref_by_id.get(did)
    print(f"\n  JAX   {did}")
    print(f"        H={dd.final_potts_energy:.6f} sel={dd.selective_energy:.6f}")
    print(f"        {dd.canonical_sequence}")
    if r is None:
        print("        !! no torch design with this id"); ok = False; continue
    same = r["extended_tokens"] == " ".join(dd.extended_tokens)
    nmis = sum(a != b for a, b in zip(r["extended_tokens"].split(), dd.extended_tokens))
    print(f"  torch H={r['potts_energy']:.6f} sel={r['selective_energy']:.6f}")
    print(f"        tokens identical: {same}" + ("" if same else f"  ({nmis} token(s) differ)"))
    print(f"        dH={dd.final_potts_energy - r['potts_energy']:+.6f}  "
          f"dSel={dd.selective_energy - r['selective_energy']:+.6f}")
    ok = ok and same
print("\nRESULT:", "IDENTICAL" if ok else "DIVERGENT")
