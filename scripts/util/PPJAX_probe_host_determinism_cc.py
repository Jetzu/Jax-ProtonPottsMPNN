# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX.
# Purpose: is the port's float32 arithmetic HOST-dependent? XLA picks vector widths from the CPU,
#          so the same code can associate a reduction differently on different nodes. Reports the
#          host, its vector ISA, and the Potts Hamiltonian of the native sequence.
import json, platform, socket, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from ppjax.scorer import PottsScorer

d = np.load("tests/data/ref_pdl1/PPJAX_reference_ctx.npz")
sc = PottsScorer.from_tables(d["etab_out"], d["E_idx"])
H = sc.H_of(d["S_native"])
ce = np.asarray(sc.cond_energy(d["S_native"]))
try:
    flags = subprocess.run(["grep", "-m1", "^flags", "/proc/cpuinfo"], capture_output=True,
                           text=True).stdout
    isa = ",".join(sorted({f for f in flags.split() if f.startswith(("avx", "sse4"))}))
except Exception:
    isa = "?"
print(json.dumps({
    "host": socket.gethostname(), "machine": platform.machine(), "isa": isa[:120],
    "H_native_jax": repr(H), "H_native_torch_ref": repr(float(d["H_native"])),
    "dH": H - float(d["H_native"]),
    "cond_energy_checksum": repr(float(ce.sum())),
}))
