# Origin: cc (Claude Code, 2026-10-02) - mirrors ~/ProtonPottsMPNN/inference/design_ph.py, rewritten
#         against ppjax (the JAX port). Upstream read-only.
# Purpose: the worked example - design one pH-switch binder with block descent, then sweep
#          combined_lambda to trace the stability <-> selectivity Pareto front. Same outputs as the
#          upstream script (designs.fasta / designs_states.fasta / designs.tsv / trajectory.tsv /
#          sweep_designs.tsv), produced by the JAX engine.

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ppjax.criteria import PHDesignCriteria            # noqa: E402
from ppjax.engine import PottsMPNNPHEngine             # noqa: E402
from ppjax.frontend import FeaturisedBackbone, featurise_structure  # noqa: E402

def _upstream_root() -> Path:
    """The unmodified ProtonPottsMPNN checkout. Override with PROTON_ROOT (upstream's own
    convention); defaults to ~/ProtonPottsMPNN."""
    return Path(os.environ.get("PROTON_ROOT") or (Path.home() / "ProtonPottsMPNN")).expanduser()


UPSTREAM = _upstream_root()


def main() -> int:
    ap = argparse.ArgumentParser(description="pH-switch binder design with ppjax")
    ap.add_argument("--checkpoint", default=str(
        UPSTREAM / "checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt"))
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--features", help="featurised backbone .npz")
    src.add_argument("--structure", help="PDB/CIF (needs the upstream mpnn package + HBPLUS_PATH)")
    ap.add_argument("--binder-chain", default="A")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-designs", type=int, default=12, help="lambda values in the sweep")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fb = (FeaturisedBackbone.load(args.features, binder_chain=args.binder_chain)
          if args.features else
          featurise_structure(args.structure, binder_chain=args.binder_chain))

    eng = PottsMPNNPHEngine.from_checkpoint(args.checkpoint, extended_vocab="v6")
    ctx = eng.build_context(fb.features, fb.binder_chain, token_res_id=fb.token_res_id,
                            token_res_name=fb.token_res_name, token_chain_id=fb.token_chain_id,
                            region_masks=fb.region_masks, base_seed=args.seed)
    print(f"engine ready - {len(ctx.chA_free_idx)} free (designable) binder positions")

    crit = PHDesignCriteria(
        method="block_descent", backend="potts", temperature=0.05, samples_per_site=2,
        block_size=3, combined_lambda=0.3, seed_source="native",
        center_types=["HIS-P", "ASP-P", "GLU-P"],
        dep_map={"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]},
        forbidden_tokens=["HIS-A", "ASP-A", "GLU-A", "UNK"], placement_by="scan_potts",
        placement_region=["all"],
        repetitive_window_parents=["ARG", "LYS", "HIS", "ASP", "GLU"],
        repetitive_window_radius=2, repetitive_window_weight=1.0, neighbour_k=16,
        max_mutations=20, record_trajectory=True)

    designs = eng.run_ph_redesign(ctx=ctx, criteria_list=[crit], seed=args.seed)
    best = designs[0]
    print(f"produced {len(designs)} design(s); best H = {best.final_potts_energy:.3f}, "
          f"selective = {best.selective_energy:.3f}")
    print(f"centres: {list(zip(best.center_res_ids, best.center_protonation_types))}")
    print(f"sequence: {best.canonical_sequence}")

    _write_designs(out, designs, "designs")
    with open(out / "trajectory.tsv", "w") as fh:
        fh.write("step\tpotts_energy\tselective_energy\tcanonical_sequence\textended_tokens\n")
        for t in (best.energy_trajectory or []):
            fh.write(f"{t['step']}\t{t['potts_energy']:.6f}\t{t['selective_energy']:.6f}\t"
                     f"{t['canonical_sequence']}\t{t['extended_tokens']}\n")

    lambdas = np.round(np.linspace(0.0, 1.0, args.n_designs), 3)
    sweep_criteria = [replace(crit, combined_lambda=float(l), samples_per_site=1,
                              record_trajectory=False) for l in lambdas]
    sweep = eng.run_ph_redesign(ctx=ctx, criteria_list=sweep_criteria, seed=args.seed)
    rows = sorted(({"combined_lambda": d.combined_lambda,
                    "potts_energy": d.final_potts_energy,
                    "selective_energy": d.selective_energy,
                    "design_id": d.design_id(),
                    "canonical_sequence": d.canonical_sequence,
                    "extended_tokens": " ".join(d.extended_tokens)} for d in sweep),
                  key=lambda r: r["combined_lambda"])
    P = np.array([[r["potts_energy"], r["selective_energy"]] for r in rows])
    dominated = np.zeros(len(P), bool)
    for i in range(len(P)):
        for j in range(len(P)):
            if i != j and (P[j] <= P[i]).all() and (P[j] < P[i]).any():
                dominated[i] = True
                break
    with open(out / "sweep_designs.tsv", "w") as fh:
        cols = ["combined_lambda", "potts_energy", "selective_energy", "pareto_optimal",
                "design_id", "canonical_sequence", "extended_tokens"]
        fh.write("\t".join(cols) + "\n")
        for r, dom in zip(rows, dominated):
            fh.write("\t".join([f"{r['combined_lambda']}", f"{r['potts_energy']:.6f}",
                                f"{r['selective_energy']:.6f}", str(not dom), r["design_id"],
                                r["canonical_sequence"], r["extended_tokens"]]) + "\n")
    print(f"sweep: {len(rows)} designs, {int((~dominated).sum())} Pareto-optimal "
          f"-> {out/'sweep_designs.tsv'}")
    return 0


def _write_designs(out: Path, designs, stem: str) -> None:
    with open(out / f"{stem}.fasta", "w") as fh:
        for d in designs:
            fh.write(f">{d.design_id()} H={d.final_potts_energy:.3f} sel={d.selective_energy}\n"
                     f"{d.canonical_sequence}\n")
    with open(out / f"{stem}_states.fasta", "w") as fh:
        for d in designs:
            fh.write(f">{d.design_id()} H={d.final_potts_energy:.3f} sel={d.selective_energy}\n"
                     f"{' '.join(d.extended_tokens)}\n")
    (out / f"{stem}.json").write_text(
        json.dumps([d.to_metadata() for d in designs], indent=2, default=float))
    with open(out / f"{stem}.tsv", "w") as fh:
        fh.write("design_id\tcombined_lambda\tpotts_energy\tselective_energy\tcenters\t"
                 "canonical_sequence\textended_tokens\n")
        for d in designs:
            centers = ";".join(f"{r}:{t}" for r, t in
                               zip(d.center_res_ids or [], d.center_protonation_types or []))
            fh.write("\t".join([d.design_id(), str(d.combined_lambda),
                                f"{d.final_potts_energy:.6f}", str(d.selective_energy), centers,
                                d.canonical_sequence, " ".join(d.extended_tokens)]) + "\n")


if __name__ == "__main__":
    sys.exit(main())
