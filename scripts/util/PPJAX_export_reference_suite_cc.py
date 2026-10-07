# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX; runs the UPSTREAM torch engine read-only.
# Purpose: golden outputs for a BROAD matrix of design configurations (lambda sweep, every
#          Potts-backend method, several seeds, two structures) so the JAX port's equivalence is
#          not established on one happy path.
import os, sys, json, time
os.environ.pop("DEBUG", None); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
import numpy as np, torch
from dataclasses import replace

def _upstream_root() -> Path:
    """The unmodified ProtonPottsMPNN checkout. Override with PROTON_ROOT (upstream's own
    convention); defaults to ~/ProtonPottsMPNN."""
    return Path(os.environ.get("PROTON_ROOT") or (Path.home() / "ProtonPottsMPNN")).expanduser()


UP = _upstream_root()
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
CKPT = str(UP / "checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt")

from mpnn.inference_engines.potts_mpnn_ph import PottsMPNNPHEngine, PHDesignCriteria
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure

engine = PottsMPNNPHEngine(checkpoint_path=CKPT, extended_vocab="v6",
                           out_directory=None, write_fasta=False, write_structures=False)

BASE = dict(method="block_descent", backend="potts", temperature=0.05, samples_per_site=2,
            block_size=3, combined_lambda=0.3, seed_source="native",
            center_types=["HIS-P", "ASP-P", "GLU-P"],
            dep_map={"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]},
            forbidden_tokens=["HIS-A", "ASP-A", "GLU-A", "UNK"], placement_by="scan_potts",
            placement_region=["all"],
            repetitive_window_parents=["ARG", "LYS", "HIS", "ASP", "GLU"],
            repetitive_window_radius=2, repetitive_window_weight=1.0, neighbour_k=16,
            max_mutations=20, record_trajectory=False)

def cases():
    yield "lambda_sweep", 0, [PHDesignCriteria(**{**BASE, "combined_lambda": float(l),
                                                  "samples_per_site": 1})
                              for l in np.round(np.linspace(0.0, 1.0, 12), 3)]
    yield "block_sizes", 0, [PHDesignCriteria(**{**BASE, "block_size": b, "samples_per_site": 1})
                             for b in (1, 2, 4)]
    yield "T0_argmin", 0, [PHDesignCriteria(**{**BASE, "temperature": 0.0, "samples_per_site": 1})]
    yield "sweep_orders", 0, [PHDesignCriteria(**{**BASE, "sweep_order": o, "samples_per_site": 1})
                              for o in ("position", "knn", "energy")]
    yield "zscale_single_mutation", 0, [PHDesignCriteria(
        **{**BASE, "zscale_mode": "single_mutation", "samples_per_site": 1})]
    yield "self_weight", 0, [PHDesignCriteria(**{**BASE, "self_weight": w, "samples_per_site": 1})
                             for w in (0.5, 2.0)]
    yield "adjacent_repeat", 0, [PHDesignCriteria(
        **{**BASE, "adjacent_repeat_weight": 0.5, "samples_per_site": 1})]
    yield "no_repetitive", 0, [PHDesignCriteria(
        **{**BASE, "repetitive_window_weight": 0.0, "samples_per_site": 1})]
    yield "converged_mcmc", 0, [PHDesignCriteria(
        **{**BASE, "method": "converged_mcmc", "samples_per_site": 2})]
    yield "converged_mcmc_combined", 0, [PHDesignCriteria(
        **{**BASE, "method": "converged_mcmc_combined", "samples_per_site": 2})]
    yield "two_phase", 0, [PHDesignCriteria(**{**BASE, "method": "two_phase",
                                               "samples_per_site": 2})]
    yield "greedy_energy_block", 0, [PHDesignCriteria(
        **{**BASE, "method": "greedy_energy_block", "center_count": 0, "center_types": [],
           "infill_scope": "chain", "samples_per_site": 1, "cv_max": 6, "cv_patience": 2,
           "repetitive_window_parents": ["ALL"]})]
    yield "gibbs", 0, [PHDesignCriteria(**{**BASE, "method": "gibbs", "num_designs": 2,
                                           "temperature": 0.05})]
    for sd in (1, 7, 12345):
        yield f"seed{sd}", sd, [PHDesignCriteria(**BASE)]
    yield "explicit_centers", 0, [PHDesignCriteria(
        **{**BASE, "center_types": [], "explicit_centers": [
            {"res_id": 20, "protonation_type": "HIS-P"},
            {"res_id": 55, "protonation_type": "GLU-P"}], "center_count": 2,
           "samples_per_site": 2})]
    yield "placement_random", 0, [PHDesignCriteria(
        **{**BASE, "center_types": [], "placement_by": "random", "center_count": 2,
           "candidate_pool": 6, "n_plan_samples": 8, "max_plans_per_seed": 2,
           "samples_per_site": 1})]
    yield "combo_scan_potts", 0, [PHDesignCriteria(
        **{**BASE, "center_types": [], "center_count": 2, "candidate_pool": 6,
           "n_plan_samples": 8, "max_plans_per_seed": 2, "samples_per_site": 1})]
    yield "regions", 0, [PHDesignCriteria(**{**BASE, "placement_region": [r],
                                             "samples_per_site": 1})
                         for r in ("interface", "core", "surface")]
    yield "neighbour_k0", 0, [PHDesignCriteria(**{**BASE, "neighbour_k": 0, "max_mutations": 12,
                                                  "samples_per_site": 1})]
    yield "inverse_seeds", 0, [PHDesignCriteria(**{**BASE, "seed_source": "inverse",
                                                   "samples_per_site": 1})]

structures = {"pdl1": (str(UP / "inference/examples/pdl1_seed_binder.pdb"), "A")}
fold = UP / "inference/examples/pdl1_design_fold.cif"
if fold.exists():
    structures["fold"] = (str(fold), "A")

results = {}
for sname, (path, chain) in structures.items():
    if path.endswith(".cif"):
        aa = get_structure(CIFFile.read(path), model=1)
    else:
        aa = PDBFile.read(path).get_structure(model=1)
    ctx = engine._build_context(aa, chain)
    binder_len = int(ctx.chainA_all_idx.numel())
    native_seq = ctx.decode_canonical(ctx.S_native, ctx.chainA_all_idx.tolist())
    seeds_for_inverse = [native_seq, native_seq[:-1] + "A"]
    arrays = {f"ni_{k}": v.detach().cpu().numpy()
              for k, v in ctx.network_input["input_features"].items() if torch.is_tensor(v)}
    arrays["token_res_id"] = np.asarray(ctx.token_aa.res_id)
    arrays["token_res_name"] = np.asarray(ctx.token_aa.res_name).astype("U8")
    arrays["token_chain_id"] = np.asarray(ctx.token_aa.chain_id).astype("U8")
    arrays["etab_out"] = ctx.scorer.etab_out.detach().cpu().numpy()
    arrays["E_idx"] = ctx.scorer.E_idx.detach().cpu().numpy()
    arrays["H_native"] = np.asarray(ctx.scorer.H_of(ctx.S_native))
    for n, m in ctx.region_masks.items():
        arrays[f"region_{n}"] = m.detach().cpu().numpy()
    np.savez_compressed(OUT / f"PPJAX_ctx_{sname}.npz", **arrays)
    print(f"[{sname}] featurised L={ctx.L} binder={binder_len} chain={chain}", flush=True)

    for case, seed, crits in cases():
        if sname != "pdl1" and case not in ("lambda_sweep", "converged_mcmc", "greedy_energy_block"):
            continue
        t0 = time.time()
        try:
            init = seeds_for_inverse if case == "inverse_seeds" else None
            ds = engine.run_ph_redesign(atom_array=aa, binder_chain=chain, criteria_list=crits,
                                        seed=seed, initial_sequences=init, n_jobs=1)
            results[f"{sname}/{case}"] = {
                "seed": seed, "structure": sname,
                "initial_sequences": init,
                "criteria": [c.__dict__ for c in crits],
                "designs": [d.to_metadata() for d in ds],
            }
            print(f"[{sname}/{case}] {len(ds)} designs in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            results[f"{sname}/{case}"] = {"error": f"{type(e).__name__}: {e}", "seed": seed,
                                          "criteria": [c.__dict__ for c in crits]}
            print(f"[{sname}/{case}] ERROR {type(e).__name__}: {e}", flush=True)

(OUT / "PPJAX_reference_suite.json").write_text(json.dumps(results, indent=1, default=str))
print("wrote", OUT / "PPJAX_reference_suite.json", len(results), "cases")
