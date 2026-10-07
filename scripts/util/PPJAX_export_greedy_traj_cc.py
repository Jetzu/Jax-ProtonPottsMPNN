# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX; runs the UPSTREAM torch engine read-only.
# Purpose: a STEP-BY-STEP torch trajectory for greedy_energy_block, so the first diverging step can
#          be located exactly instead of inferred from the endpoint.
import os, sys, json
os.environ.pop("DEBUG", None); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
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
crit = PHDesignCriteria(
    method="greedy_energy_block", backend="potts", temperature=0.05, block_size=3,
    combined_lambda=0.3, seed_source="native", center_count=0, center_types=[],
    infill_scope="chain", dep_map={"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]},
    forbidden_tokens=["HIS-A", "ASP-A", "GLU-A", "UNK"], placement_region=["all"],
    samples_per_site=1, repetitive_window_parents=["ALL"], repetitive_window_radius=2,
    repetitive_window_weight=1.0, neighbour_k=16, max_mutations=20,
    cv_max=6, cv_patience=2, record_trajectory=True)
ds = engine.run_ph_redesign(atom_array=aa, binder_chain="A", criteria_list=[crit], seed=0, n_jobs=1)
(OUT / "PPJAX_greedy_traj.json").write_text(
    json.dumps({"criteria": crit.__dict__, "designs": [d.to_metadata() for d in ds]},
               indent=1, default=str))
print("steps:", len(ds[0].energy_trajectory or []))
