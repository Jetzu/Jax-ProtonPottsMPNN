# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX; no upstream.
# Purpose: determine which transformation torch's CPU exponential_ uses (-log(u) vs -log1p(-u))
#          and at which precision, so ppjax.rng.multinomial1 matches torch.multinomial bit-for-bit.
import torch
torch.manual_seed(0); e32 = torch.empty(8, dtype=torch.float32).exponential_(1)
torch.manual_seed(0); u64 = torch.rand(8, dtype=torch.float64)
torch.manual_seed(0); u32 = torch.rand(8, dtype=torch.float32)
torch.manual_seed(0); e64 = torch.empty(8, dtype=torch.float64).exponential_(1)
print("exp32        ", e32.tolist())
print("exp64        ", e64.tolist())
print("-log(u64)    ", (-torch.log(u64)).tolist())
print("-log1p(-u64) ", (-torch.log1p(-u64)).tolist())
print("-log(u32)    ", (-torch.log(u32)).tolist())
print("-log1p(-u32) ", (-torch.log1p(-u32)).tolist())
