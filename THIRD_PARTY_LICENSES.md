# Licences in this repository

This is a **fork of [ProtonPottsMPNN](https://github.com/christian-creator/ProtonPottsMPNN)** in which
the *design/generation path* has been re-implemented in JAX. The original tree is kept here in full —
including the trained model weights — and keeps its own licence. Only the port's own code is
noncommercial.

| material | path | licence |
|---|---|---|
| **The JAX port** | `src/ppjax/`, `scripts/`, `tests/`, `inference/design_ph.py`, `pyproject.toml` | PolyForm Noncommercial 1.0.0 — [`LICENSE-ppjax`](LICENSE-ppjax) |
| **Proton-PottsMPNN** (Jacobsen et al., 2026) | `labeller/`, `scoring/`, `benchmarks/`, `training/`, `checkpoints/` (**the model weights**), `figures/`, the rest of `inference/`, and the pH-design additions to the bundled `mpnn` package | MIT, © 2026 Christian P. Jacobsen — [`LICENSE`](LICENSE) (unchanged) |
| **rc-foundry** (Institute for Protein Design) | [`foundry/`](foundry/) | BSD 3-Clause, © 2025 IPD, University of Washington — [`foundry/LICENSE.md`](foundry/LICENSE.md) (unchanged) |
| **PottsMPNN** (Birnbaum & Keating, PNAS 2026) | the Potts-energy formulation the above extends | as published; see <https://github.com/KeatingLab/PottsMPNN> |
| **This README**, adapted from the original's | `README.md` | MIT, with the original's notice retained |

The noncommercial term applies **only** to the port's own code. It does not and cannot restrict the
original work or the weights, which remain available under MIT from the upstream repository. If you want
the original under MIT alone, take it from there — nothing here changes its terms.

`src/ppjax/` is a re-implementation, not a copy: it contains no upstream source. It reads the same
checkpoint, and `ppjax.frontend` calls the bundled `mpnn` package for structure featurisation, which is
licensed as above.

The two original licence texts are reproduced below for convenience; the authoritative copies are
[`LICENSE`](LICENSE) and [`foundry/LICENSE.md`](foundry/LICENSE.md).

---

## Proton-PottsMPNN — MIT License

```
MIT License

Copyright (c) 2026 Christian P. Jacobsen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## rc-foundry (`foundry/`) — BSD 3-Clause License

```
BSD 3-Clause License

Copyright (c) 2025, Institute for Protein Design, University of Washington

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
