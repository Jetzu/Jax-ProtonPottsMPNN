# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX.
# Purpose: layer-by-layer equivalence of the JAX forward against the torch golden tensors exported
#          by scripts/util/PPJAX_export_reference_cc.py.
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import jax.numpy as jnp

from ppjax.checkpoint import load_state_dict
from ppjax.params import build_params, infer_etab_source, infer_field_source, infer_shapes
from ppjax.model import ModelConfig, forward_potts
from ppjax.features import FeatureConfig
from ppjax.tokens import get_encoding

REF = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/data/ref_pdl1")
CKPT = json.loads((REF / "PPJAX_reference_meta.json").read_text())["ckpt"]
d = np.load(REF / "PPJAX_reference_ctx.npz")

flat = load_state_dict(CKPT)
params = build_params(flat)
shapes = infer_shapes(flat)
enc = get_encoding("v6")
cfg = ModelConfig(vocab_size=shapes["vocab_size"], hidden_dim=shapes["hidden_dim"],
                  num_encoder_layers=shapes["num_encoder_layers"],
                  num_decoder_layers=shapes["num_decoder_layers"],
                  field_source=infer_field_source(flat), etab_source=infer_etab_source(flat),
                  features=FeatureConfig())
print("cfg:", cfg.vocab_size, cfg.field_source, cfg.etab_source, "| enc tokens", enc.n_tokens)

feats = {k[3:]: d[k] for k in d.files if k.startswith("ni_")}
out = forward_potts(params, enc, cfg, enc.unknown_indices(), feats)

def cmp(name, got, want, idx=False):
    got = np.asarray(got); want = np.asarray(want)
    if got.shape != want.shape:
        print(f"  {name:12s} SHAPE MISMATCH {got.shape} vs {want.shape}"); return
    if idx:
        n = int((got != want).sum())
        print(f"  {name:12s} exact-match {'YES' if n == 0 else f'NO ({n} differ)'}")
        return
    err = np.abs(got.astype(np.float64) - want.astype(np.float64))
    den = np.maximum(np.abs(want.astype(np.float64)), 1e-9)
    print(f"  {name:12s} max_abs={err.max():.3e}  max_rel={(err/den).max():.3e}  "
          f"ref_absmax={np.abs(want).max():.3e}")

print("featurisation / encoder:")
cmp("E_idx", out["E_idx"], d["cap_E_idx"], idx=True)
cmp("E_raw", out["E_raw"], d["cap_E_raw"])
cmp("E", out["E"], d["cap_E"])
cmp("h_V", out["h_V"], d["cap_h_V"])
cmp("h_E", out["h_E"], d["cap_h_E"])
print("potts head:")
cmp("etab_out", out["etab_out"], d["etab_out"])
cmp("masks", out["mask_for_loss"], d["ni_mask_for_loss"], idx=True)
