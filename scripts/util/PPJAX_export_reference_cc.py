# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX; runs the UPSTREAM torch ProtonPottsMPNN
#         read-only (~/ProtonPottsMPNN) inside its own .venv.
# Purpose: export the featurised network_input plus every intermediate tensor of the generation path
#          (edge features, encoder states, Potts tables) and the reference design outputs, as .npz /
#          .json golden files for the JAX port's equivalence tests.
import os, sys, json, argparse
os.environ.pop("DEBUG", None)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
import numpy as np
import torch

def _upstream_root() -> Path:
    """The unmodified ProtonPottsMPNN checkout. Override with PROTON_ROOT (upstream's own
    convention); defaults to ~/ProtonPottsMPNN."""
    return Path(os.environ.get("PROTON_ROOT") or (Path.home() / "ProtonPottsMPNN")).expanduser()


UPSTREAM = _upstream_root()

ap = argparse.ArgumentParser()
ap.add_argument("--pdb", default=None)
ap.add_argument("--binder-chain", default=None)
ap.add_argument("--ckpt", default=str(UPSTREAM / "checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt"))
ap.add_argument("--out", required=True)
ap.add_argument("--designs", action="store_true", help="also run the reference design (slow)")
args = ap.parse_args()

META = json.loads((UPSTREAM / "inference/examples/example_meta.json").read_text())
pdb = args.pdb or str(UPSTREAM / "inference/examples" / META["pdb"])
binder_chain = args.binder_chain or META["binder_chain"]
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

from biotite.structure.io.pdb import PDBFile
from mpnn.inference_engines.potts_mpnn_ph import PottsMPNNPHEngine, PHDesignCriteria

engine = PottsMPNNPHEngine(checkpoint_path=args.ckpt, extended_vocab="v6",
                           out_directory=None, write_fasta=False, write_structures=False)
model = engine.model
atom_array = PDBFile.read(pdb).get_structure(model=1)

# ---- capture every intermediate of the encoder-side forward -----------------
cap = {}
gfm = model.graph_featurization_module
_orig_edge_emb = gfm.edge_embedding.forward
def _edge_emb(x):
    cap["E_raw"] = x.detach().cpu().numpy()
    y = _orig_edge_emb(x); cap["E_embed"] = y.detach().cpu().numpy(); return y
gfm.edge_embedding.forward = _edge_emb

_orig_enc = model.encode
def _encode(input_features, graph_features):
    cap["E"] = graph_features["E"].detach().cpu().numpy()
    cap["E_idx"] = graph_features["E_idx"].detach().cpu().numpy()
    ef = _orig_enc(input_features, graph_features)
    cap["h_V"] = ef["h_V"].detach().cpu().numpy()
    cap["h_E"] = ef["h_E"].detach().cpu().numpy()
    return ef
model.encode = _encode

ctx = engine._build_context(atom_array, binder_chain)
model.encode = _orig_enc
gfm.edge_embedding.forward = _orig_edge_emb

ni = ctx.network_input["input_features"]
arrays = {f"ni_{k}": v.detach().cpu().numpy() for k, v in ni.items() if torch.is_tensor(v)}
arrays.update({f"cap_{k}": v for k, v in cap.items()})
arrays["etab_out"] = ctx.scorer.etab_out.detach().cpu().numpy()
arrays["E_idx"] = ctx.scorer.E_idx.detach().cpu().numpy()
arrays["S_native"] = ctx.S_native.detach().cpu().numpy()
arrays["free_mask"] = ctx.free_mask.detach().cpu().numpy()
arrays["chainA"] = ctx.chainA_t.detach().cpu().numpy()
arrays["canonical_map"] = ctx.canonical_map.detach().cpu().numpy()
arrays["token_res_id"] = np.asarray(ctx.token_aa.res_id)
arrays["token_res_name"] = np.asarray(ctx.token_aa.res_name).astype("U8")
arrays["token_chain_id"] = np.asarray(ctx.token_aa.chain_id).astype("U8")
for name, m in ctx.region_masks.items():
    arrays[f"region_{name}"] = m.detach().cpu().numpy()
# a couple of derived scores to pin the scorer down
arrays["cond_energy_native"] = ctx.scorer.cond_energy(ctx.S_native).detach().cpu().numpy()
arrays["H_native"] = np.asarray(ctx.scorer.H_of(ctx.S_native), dtype=np.float64)

np.savez_compressed(out / "PPJAX_reference_ctx.npz", **arrays)
meta = {"pdb": pdb, "binder_chain": binder_chain, "ckpt": args.ckpt,
        "L": int(ctx.L), "V": int(ctx.V), "K": int(ctx.K),
        "unknown_indices": list(map(int, ctx.unknown_indices)),
        "torch_version": torch.__version__}
(out / "PPJAX_reference_meta.json").write_text(json.dumps(meta, indent=1))
print("wrote ctx npz; L", ctx.L, "V", ctx.V, "K", ctx.K, "H_native", float(arrays["H_native"]))

if args.designs:
    crit = PHDesignCriteria(
        method="block_descent", backend="potts", temperature=0.05, samples_per_site=2,
        block_size=3, combined_lambda=0.3, seed_source="native",
        center_types=["HIS-P", "ASP-P", "GLU-P"],
        dep_map={"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]},
        forbidden_tokens=["HIS-A", "ASP-A", "GLU-A", "UNK"], placement_by="scan_potts",
        placement_region=["all"], repetitive_window_parents=["ARG", "LYS", "HIS", "ASP", "GLU"],
        repetitive_window_radius=2, repetitive_window_weight=1.0, neighbour_k=16,
        max_mutations=20, record_trajectory=True)
    ds = engine.run_ph_redesign(atom_array=atom_array, binder_chain=binder_chain,
                                criteria_list=[crit], seed=0)
    (out / "PPJAX_reference_designs.json").write_text(
        json.dumps([d.to_metadata() for d in ds], indent=1, default=float))
    for d in ds:
        print("DESIGN", d.design_id(), f"H={d.final_potts_energy:.6f}", f"sel={d.selective_energy}")
        print("   ", d.canonical_sequence)
