# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX; no upstream.
# Purpose: pin down torch's CPU randn consumption (normal_fill fast path vs the scalar
#          Box-Muller kernel) so ppjax.rng.normal_float32 keeps the stream aligned.
import torch, json
out = {}
for n in (4, 16, 17, 32, 229):
    torch.manual_seed(0); out[f"randn_f32_{n}"] = torch.randn(n, dtype=torch.float32).tolist()
    torch.manual_seed(0); _ = torch.randn(n, dtype=torch.float32)
    out[f"after_randn_{n}"] = torch.rand(3, dtype=torch.float32).tolist()
torch.manual_seed(0); out["uniform_245"] = torch.rand(245, dtype=torch.float32).tolist()
torch.manual_seed(0); out["randn_f64_4"] = torch.randn(4, dtype=torch.float64).tolist()
print(json.dumps(out))
