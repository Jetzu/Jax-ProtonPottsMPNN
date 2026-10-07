# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX; runs the UPSTREAM torch engine read-only.
# Purpose: golden outputs for the DECODER half of the generation path - the conditional_minus_self
#          design field, autoregressive sampling, and the mpnn-backend design methods.
import os, sys, json, time
os.environ.pop("DEBUG", None); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
import numpy as np, torch

def _upstream_root() -> Path:
    """The unmodified ProtonPottsMPNN checkout. Override with PROTON_ROOT (upstream's own
    convention); defaults to ~/ProtonPottsMPNN."""
    return Path(os.environ.get("PROTON_ROOT") or (Path.home() / "ProtonPottsMPNN")).expanduser()


UP = _upstream_root()
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
CKPT = str(UP / "checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt")
from mpnn.inference_engines.potts_mpnn_ph import PottsMPNNPHEngine, PHDesignCriteria
from biotite.structure.io.pdb import PDBFile

engine = PottsMPNNPHEngine(checkpoint_path=CKPT, extended_vocab="v6",
                           out_directory=None, write_fasta=False, write_structures=False)
aa = PDBFile.read(str(UP / "inference/examples/pdl1_seed_binder.pdb")).get_structure(model=1)
ctx = engine._build_context(aa, "A")

arrays, res = {}, {}
torch.manual_seed(0)
arrays["field_mpnn_native"] = ctx.field_mpnn(ctx.S_native).detach().cpu().numpy()
S2 = ctx.S_native.clone(); S2[int(ctx.chA_free_idx[0])] = ctx.encoding.token_to_idx["UNK"]
torch.manual_seed(0)
arrays["field_mpnn_unk"] = ctx.field_mpnn(S2).detach().cpu().numpy()
arrays["S_mpnn_unk_input"] = S2.cpu().numpy()

# autoregressive sampling (mpnn_sample path), fixed seed
import copy
ni = copy.deepcopy(ctx.network_input)
ni["input_features"]["repeat_sample_num"] = 4
torch.manual_seed(0)
with torch.no_grad():
    S_sampled = engine.model(ni)["decoder_features"]["S_sampled"]
arrays["mpnn_sample_seed0"] = S_sampled.cpu().numpy()
print("mpnn_sample shapes", S_sampled.shape, flush=True)

BASE = dict(backend="mpnn", temperature=0.1, samples_per_site=1, combined_lambda=0.3,
            seed_source="native", center_types=["HIS-P", "ASP-P", "GLU-P"],
            dep_map={"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]},
            forbidden_tokens=["HIS-A", "ASP-A", "GLU-A", "UNK"],
            placement_region=["all"], neighbour_k=8, max_mutations=8, record_trajectory=False)
cases = {
    "autoregressive_potts_sel": PHDesignCriteria(**{**BASE, "method": "autoregressive",
                                                    "placement_by": "scan_potts"}),
    "autoregressive_decoder_sel": PHDesignCriteria(**{**BASE, "method": "autoregressive",
                                                      "selective_source": "decoder",
                                                      "placement_by": "scan_potts"}),
    "scan_mpnn_placement": PHDesignCriteria(**{**BASE, "method": "autoregressive",
                                               "placement_by": "scan_mpnn"}),
    "mpnn_sample": PHDesignCriteria(**{**BASE, "method": "mpnn_sample", "num_designs": 3,
                                       "center_types": []}),
}
for name, crit in cases.items():
    t0 = time.time()
    try:
        ds = engine.run_ph_redesign(atom_array=aa, binder_chain="A", criteria_list=[crit],
                                    seed=0, n_jobs=1)
        res[name] = {"criteria": crit.__dict__, "designs": [d.to_metadata() for d in ds]}
        print(f"[{name}] {len(ds)} designs in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        res[name] = {"error": f"{type(e).__name__}: {e}"}
        print(f"[{name}] ERROR {type(e).__name__}: {e}", flush=True)

np.savez_compressed(OUT / "PPJAX_decoder_ref.npz", **arrays)
(OUT / "PPJAX_decoder_ref.json").write_text(json.dumps(res, indent=1, default=str))
print("done")
