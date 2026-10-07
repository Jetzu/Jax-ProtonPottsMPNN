# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX; no upstream.
# Purpose: reproduce (or refute) two red-team objections first-hand - (a) that torch's Welford
#          std/var differs materially from numpy's pairwise reduction at float32, and (b) that
#          torch.argsort's default unstable sort breaks ties differently from numpy's stable sort.
import numpy as np, torch

v = np.load("tests/data/PPJAX_zscale_probe_vals.npy")
t = torch.from_numpy(v)
print("--- std/var on a real block-descent J tensor (n=%d) ---" % v.size)
for name, a, b in [
    ("var", float(t.var(unbiased=False)), float(np.var(v))),
    ("std", float(t.std(unbiased=False)), float(np.std(v))),
]:
    f64 = float(np.var(v.astype(np.float64))) if name == "var" else float(np.std(v.astype(np.float64)))
    print(f"  {name}: torch={a!r} numpy_f32={b!r} numpy_f64={f64!r}")
    print(f"       rel(torch,numpy_f32)={abs(a-b)/abs(f64):.3e}  rel(torch,f64)={abs(a-f64)/abs(f64):.3e}")

# a deltas-like vector with heavy cancellation (what _zscale actually sees)
rs = np.random.RandomState(0)
d = (rs.randn(2400).astype(np.float32) * np.float32(80.0))
td = torch.from_numpy(d)
print("--- std on a cancellation-heavy delta vector (n=2400, sd~80) ---")
print(f"  torch={float(td.std(unbiased=False))!r} numpy_f32={float(np.std(d))!r} "
      f"numpy_f64={float(np.std(d.astype(np.float64)))!r}")

print("--- argsort tie-breaking ---")
for n, nties in ((48, 8), (114, 20), (229, 40)):
    x = rs.randn(n).astype(np.float32)
    x[:nties] = x[0]                       # exact ties
    rs.shuffle(x)
    ti = torch.argsort(torch.from_numpy(x)).numpy()
    ns = np.argsort(x, kind="stable")
    nq = np.argsort(x, kind="quicksort")
    print(f"  n={n:4d} ties={nties:3d}  torch==np_stable: {np.array_equal(ti, ns)}  "
          f"torch==np_quicksort: {np.array_equal(ti, nq)}")
# the 2-D case used for the decoding order
x2 = rs.randn(1, 229).astype(np.float32); x2[0, :30] = x2[0, 0]
ti = torch.argsort(torch.from_numpy(x2), dim=-1).numpy()
print(f"  2-D [1,229] ties=30  torch==np_stable: "
      f"{np.array_equal(ti, np.argsort(x2, axis=-1, kind='stable'))}")
