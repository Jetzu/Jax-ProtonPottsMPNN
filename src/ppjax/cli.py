# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX.
# Purpose: command-line entry point - featurise a structure (via the upstream front end), run the
#          JAX pH-switch design, and write the same output files as the torch example script.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _criteria_from(args):
    from ppjax.criteria import PHDesignCriteria
    if args.criteria_json:
        return [PHDesignCriteria(**c) for c in json.loads(Path(args.criteria_json).read_text())]
    base = dict(
        method=args.method, backend=args.backend, temperature=args.temperature,
        samples_per_site=args.samples_per_site, block_size=args.block_size,
        seed_source="native", center_types=list(args.center_types),
        dep_map={"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]},
        forbidden_tokens=["HIS-A", "ASP-A", "GLU-A", "UNK"], placement_by=args.placement_by,
        placement_region=["all"],
        repetitive_window_parents=["ARG", "LYS", "HIS", "ASP", "GLU"],
        repetitive_window_radius=2, repetitive_window_weight=args.repetitive_window_weight,
        neighbour_k=args.neighbour_k, max_mutations=args.max_mutations,
        record_trajectory=args.trajectory)
    if args.n_lambda > 1:
        lams = np.round(np.linspace(args.lambda_min, args.lambda_max, args.n_lambda), 3)
        return [PHDesignCriteria(**{**base, "combined_lambda": float(l)}) for l in lams]
    return [PHDesignCriteria(**{**base, "combined_lambda": args.combined_lambda})]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="ppjax", description="pH-switch binder design with the JAX ProtonPottsMPNN port")
    ap.add_argument("--checkpoint", required=True, help="PottsMPNN .ckpt (torch format)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--features", help="featurised backbone .npz (see ppjax.frontend)")
    src.add_argument("--structure", help="PDB/CIF - needs the upstream mpnn package + HBPLUS_PATH")
    ap.add_argument("--binder-chain", default="A")
    ap.add_argument("--extended-vocab", default="v6")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default="block_descent")
    ap.add_argument("--backend", default="potts")
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--block-size", type=int, default=3)
    ap.add_argument("--samples-per-site", type=int, default=2)
    ap.add_argument("--combined-lambda", type=float, default=0.3)
    ap.add_argument("--n-lambda", type=int, default=1, help=">1 sweeps lambda (the Pareto front)")
    ap.add_argument("--lambda-min", type=float, default=0.0)
    ap.add_argument("--lambda-max", type=float, default=1.0)
    ap.add_argument("--placement-by", default="scan_potts")
    ap.add_argument("--center-types", nargs="*", default=["HIS-P", "ASP-P", "GLU-P"])
    ap.add_argument("--neighbour-k", type=int, default=16)
    ap.add_argument("--max-mutations", type=int, default=20)
    ap.add_argument("--repetitive-window-weight", type=float, default=1.0)
    ap.add_argument("--trajectory", action="store_true")
    ap.add_argument("--save-features", default=None, help="also write the featurised .npz here")
    ap.add_argument("--criteria-json", default=None, help="explicit list of criteria dicts")
    args = ap.parse_args(argv)

    from ppjax.engine import PottsMPNNPHEngine
    from ppjax.frontend import FeaturisedBackbone, featurise_structure

    if args.features:
        fb = FeaturisedBackbone.load(args.features, binder_chain=args.binder_chain)
    else:
        fb = featurise_structure(args.structure, binder_chain=args.binder_chain,
                                 extended_vocab=args.extended_vocab)
        if args.save_features:
            fb.save(args.save_features)

    eng = PottsMPNNPHEngine.from_checkpoint(args.checkpoint, extended_vocab=args.extended_vocab)
    ctx = eng.build_context(fb.features, fb.binder_chain, token_res_id=fb.token_res_id,
                            token_res_name=fb.token_res_name, token_chain_id=fb.token_chain_id,
                            region_masks=fb.region_masks, base_seed=args.seed)
    designs = eng.run_ph_redesign(ctx=ctx, criteria_list=_criteria_from(args), seed=args.seed)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "designs.fasta", "w") as fh:
        for d in designs:
            fh.write(f">{d.design_id()} H={d.final_potts_energy:.3f} sel={d.selective_energy}\n"
                     f"{d.canonical_sequence}\n")
    with open(out / "designs_states.fasta", "w") as fh:
        for d in designs:
            fh.write(f">{d.design_id()} H={d.final_potts_energy:.3f} sel={d.selective_energy}\n"
                     f"{' '.join(d.extended_tokens)}\n")
    (out / "designs.json").write_text(
        json.dumps([d.to_metadata() for d in designs], indent=2, default=float))
    with open(out / "designs.tsv", "w") as fh:
        cols = ["design_id", "combined_lambda", "potts_energy", "selective_energy", "centers",
                "canonical_sequence", "extended_tokens"]
        fh.write("\t".join(cols) + "\n")
        for d in designs:
            centers = ";".join(f"{r}:{t}" for r, t in
                               zip(d.center_res_ids or [], d.center_protonation_types or []))
            fh.write("\t".join([
                d.design_id(), str(d.combined_lambda), f"{d.final_potts_energy:.6f}",
                str(d.selective_energy), centers, d.canonical_sequence,
                " ".join(d.extended_tokens)]) + "\n")
    print(f"wrote {len(designs)} designs to {out}")
    for d in designs[:5]:
        print(f"  {d.design_id()}  H={d.final_potts_energy:.3f}  sel={d.selective_energy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
