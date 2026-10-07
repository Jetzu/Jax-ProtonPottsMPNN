# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX; no upstream.
# Purpose: probe torch's CPU RNG (manual_seed / rand / multinomial / randint / randperm) so the
#          numpy re-implementation in ppjax.rng can be validated bit-for-bit.
import json, sys
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
DATA.mkdir(parents=True, exist_ok=True)

out = {}

# 1. raw float32 uniforms after manual_seed
for s in (0, 1, 12345, 2**31, 100003):
    torch.manual_seed(s)
    out[f"rand_f32_{s}"] = torch.rand(8, dtype=torch.float32).tolist()
    torch.manual_seed(s)
    out[f"rand_f64_{s}"] = torch.rand(8, dtype=torch.float64).tolist()

# 2. multinomial, 1-D, num_samples=1 (the engine's call), on a softmax-like vector
rs = np.random.RandomState(7)
for V in (30, 27000):
    logits = torch.tensor(rs.randn(V) * 2.0, dtype=torch.float32)
    p = torch.softmax(logits, -1)
    draws = []
    torch.manual_seed(0)
    for _ in range(12):
        draws.append(int(torch.multinomial(p, 1)))
    out[f"multinomial_V{V}_seed0"] = draws
    out[f"multinomial_V{V}_probs_sha"] = int(abs(float(p.sum())) * 0)  # placeholder
    np.save(DATA / f"PPJAX_rng_probs_V{V}.npy", p.numpy())

# 3. how many uniforms does one multinomial consume? draw rand() after
V = 30
logits = torch.tensor(rs.randn(V) * 2.0, dtype=torch.float32)
p = torch.softmax(logits, -1)
np.save(DATA / "PPJAX_rng_probs_probe.npy", p.numpy())
torch.manual_seed(0)
r_before = torch.rand(4, dtype=torch.float32).tolist()
torch.manual_seed(0)
m = int(torch.multinomial(p, 1))
r_after = torch.rand(4, dtype=torch.float32).tolist()
out["probe_rand_first4_seed0"] = r_before
out["probe_multinomial_seed0"] = m
out["probe_rand_after_multinomial"] = r_after

# 4. randint / randperm
torch.manual_seed(0)
out["randint_17"] = [int(torch.randint(17, (1,))) for _ in range(10)]
torch.manual_seed(0)
out["randperm_10"] = torch.randperm(10).tolist()

# 5. torch.rand with a fresh generator object vs default
g = torch.Generator()
g.manual_seed(0)
out["gen_rand_f32_0"] = torch.rand(8, generator=g, dtype=torch.float32).tolist()

print(json.dumps(out, indent=1))
