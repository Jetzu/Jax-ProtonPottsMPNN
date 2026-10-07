# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX; runs the UPSTREAM torch engine read-only.
# Purpose: characterise the ONE reference case that diverges (pdl1/greedy_energy_block). Hypothesis:
#          it is not the algorithm but the SAMPLING - that method makes hundreds of sequential
#          27000-way Boltzmann draws, where block_descent makes ~80, so one near-tied draw is enough
#          to send the trajectories apart. Test: the same case at temperature=0 (pure argmin) and
#          with a step cap, which remove / reduce the draws.
import os, sys, json, time
os.environ.pop("DEBUG", None); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
import numpy as np
def _upstream_root() -> Path:
    """The unmodified ProtonPottsMPNN checkout. Override with PROTON_ROOT (upstream's own
    convention); defaults to ~/ProtonPottsMPNN."""
    return Path(os.environ.get("PROTON_ROOT") or (Path.home() / "ProtonPottsMPNN")).expanduser()


UP = _upstream_root()
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
from mpnn.inference_engines.potts_mpnn_ph import PottsMPNNPHEngine, PHDesignCriteria
from biotite.structure.io.pdb import PDBFile

engine = PottsMPNNPHEngine(
    checkpoint_path=str(UP / "checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt"),
    extended_vocab="v6", out_directory=None, write_fasta=False, write_structures=False)
aa = PDBFile.read(str(UP / "inference/examples/pdl1_seed_binder.pdb")).get_structure(model=1)

GREEDY = dict(method="greedy_energy_block", backend="potts", block_size=3, combined_lambda=0.3,
              seed_source="native", center_count=0, center_types=[], infill_scope="chain",
              dep_map={"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]},
              forbidden_tokens=["HIS-A", "ASP-A", "GLU-A", "UNK"], placement_region=["all"],
              samples_per_site=1, repetitive_window_parents=["ALL"],
              repetitive_window_radius=2, repetitive_window_weight=1.0,
              neighbour_k=16, max_mutations=20, record_trajectory=False)
BD = dict(method="block_descent", backend="potts", temperature=0.05, samples_per_site=1,
          block_size=3, combined_lambda=0.3, seed_source="native",
          center_types=["HIS-P", "ASP-P", "GLU-P"],
          dep_map={"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]},
          forbidden_tokens=["HIS-A", "ASP-A", "GLU-A", "UNK"], placement_by="scan_potts",
          placement_region=["all"], repetitive_window_parents=["ARG", "LYS", "HIS", "ASP", "GLU"],
          repetitive_window_radius=2, repetitive_window_weight=1.0, neighbour_k=16,
          max_mutations=20, record_trajectory=False)

cases = {
    "greedy_T0":        PHDesignCriteria(**{**GREEDY, "temperature": 0.0, "cv_max": 6, "cv_patience": 2}),
    "greedy_T005_cap6": PHDesignCriteria(**{**GREEDY, "temperature": 0.05, "cv_max": 6, "cv_patience": 2}),
    "greedy_T005_cap1": PHDesignCriteria(**{**GREEDY, "temperature": 0.05, "cv_max": 1, "cv_patience": 1}),
    "block_descent_T0": PHDesignCriteria(**{**BD, "temperature": 0.0}),
    "block_descent_r30": PHDesignCriteria(**{**BD, "block_max_rounds": 30}),
}
res = {}
for name, crit in cases.items():
    t0 = time.time()
    try:
        ds = engine.run_ph_redesign(atom_array=aa, binder_chain="A", criteria_list=[crit], seed=0,
                                    n_jobs=1)
        res[name] = {"criteria": crit.__dict__, "designs": [d.to_metadata() for d in ds]}
        print(f"[{name}] {len(ds)} designs in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        res[name] = {"error": f"{type(e).__name__}: {e}"}
        print(f"[{name}] ERROR {type(e).__name__}: {e}", flush=True)
(OUT / "PPJAX_greedy_probe.json").write_text(json.dumps(res, indent=1, default=str))
print("done")
