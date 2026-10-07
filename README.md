# pH-sensitive binder design with Proton-PottsMPNN — **JAX**

A **PottsMPNN with an explicit protonation-state alphabet**, for designing **pH-switchable** binders.
Histidine is `HIS-P` (charged, +1) vs `HIS-S` (neutral); acids are `ASP-P`/`GLU-P` (protonated, neutral
COOH) vs `ASP-D`/`GLU-D` (deprotonated, −1). Because the learned Potts energy is protonation-aware, the
design engine can **pin protonated centres** and redesign around them so that binding **switches with pH**.

> **This fork replaces the design engine with a JAX implementation.** Everything else — the trained
> model, the labeller, the benchmarks, the science — is the original work of
> [Jacobsen et al., 2026](https://github.com/christian-creator/ProtonPottsMPNN), kept here unchanged.
> The port's one hard requirement was that it be a **drop-in**, not a lookalike: the same checkpoint and
> the same seed must produce **the same designed sequences, token for token**. It does. The evidence is
> [below](#does-it-actually-match).

The original extends **PottsMPNN** — the Potts-energy inverse-folding model of **Birnbaum & Keating**
([github.com/KeatingLab/PottsMPNN](https://github.com/KeatingLab/PottsMPNN);
[PNAS 2026, 10.1073/pnas.2535494123](https://www.pnas.org/doi/10.1073/pnas.2535494123)) — by adding
explicit protonation-state tokens so that a single energy function scores alternative protonation
assignments on a fixed backbone, which is what makes pH-conditioned design possible.

![Figure 1 — Proton-PottsMPNN and pH-conditioned binder design](figures/Figure_1.png)

> **Figure 1.** **(a)** Local geometric features predict per-residue protonation states, encoded as sequence
> tokens to train Proton-PottsMPNN; backbone node/edge embeddings feed a shared encoder, then an
> autoregressive decoder (token likelihoods) and a Potts head (single-site fields + pairwise couplings).
> **(b)** Protonation-conditioned redesign for a fixed `ASP-P` centre (pink) vs `ASP-D` (blue): neighbours are
> mutated to balance global Potts energy against the selective gap `E_selective = E_P − E_D`, weighted by λ.
> **(c)** Flow cytometry of enriched yeast-display binders incubated with PD-L1 at pH 5.0 vs 7.4.
> *(Figure from the original work.)*

The project has two halves, and this folder is self-contained for both:

1. **Label** — a FLAML labeller assigns a protonation state to every titratable residue of every training
   structure (the supervision the model learns from). **Original, torch/scikit-learn.**
2. **Design** — the trained Potts model drives a block-descent optimiser that places protonated centres and
   redesigns their neighbourhood. **This is the half that is ported to JAX** (`src/ppjax/`).

---

## Does it actually match?

Measured on the shipped PD-L1 example (L = 229, 114 designable binder positions, the v6 30-token
checkpoint in `checkpoints/`) plus a second structure. CPU, torch 2.14.1 vs jax 0.8.1.

| quantity | agreement with the original |
|---|---|
| kNN graph `E_idx`, every mask | **exact** |
| edge features `E_raw` | max abs 2.6e-6 (a float64 reference is no closer to torch, so this is torch's own float32 rounding) |
| Potts tables `etab_out` | max abs 3.7e-5 on a ±37 range (rel. 1e-6) |
| conditional energies `cond_energy` | max abs 1.8e-4 on an ±868 range |
| Hamiltonian `H(S_native)` | ≤ 1.2e-2 of 5.2e4, and host-dependent (see [Limits](#limits--stated-plainly)) |
| decoder field `−log p` | max abs 2.2e-4 on a 0–71 range |
| **designed sequences** | **25/25 reference cases identical** on one machine, 24/25 on another — 2 structures × 8 design methods × both backends × 3 placement strategies × 4 seeds |

Every design method is covered **and asserted against a golden output from the original**:
`block_descent`, `greedy_energy_block`, `converged_mcmc`, `converged_mcmc_combined`, `two_phase`,
`autoregressive` (both `selective_source` variants), `gibbs`, `mpnn_sample`; both backends
(`potts`, `mpnn`); all three placement strategies (`random`, `scan_potts`, `scan_mpnn`);
`explicit_centers`, `center_types` and the combination pool.

### The part that is easy to get wrong

The design engine samples from torch's **global CPU RNG** (`manual_seed`, `multinomial`, `randint`,
`randperm`, `randn`). Reaching for `jax.random` would give a correct-looking sampler on a completely
different stream — and therefore different designs from the same seed. `ppjax.rng` reproduces torch's
generator in numpy, bit for bit. Two of its behaviours are in no documentation and had to be found by
probing torch directly:

- **`torch.multinomial(p, 1)` does not use an inverse CDF.** With `n_sample == 1` it always takes the
  *without-replacement* branch: `argmax(p / q)` with `q ~ Exp(1)` — an exponential race.
- **`Tensor.exponential_()` on CPU draws the float64 uniform** whatever the tensor's dtype (two MT19937
  words per sample), and uses `−log1p(−u)`, where the CUDA kernel uses `−log(u)`.

| call | status |
|---|---|
| `manual_seed`, `rand` (f32/f64), `exponential_`, `multinomial` (1-D and 2-D), `randint`, `randperm` | bit-exact |
| `randn` | stream position exact; values ≤ 1e-6 off (libm — see Limits) |

### Reproduce it yourself

```bash
python scripts/analysis/PPJAX_check_forward_equiv_cc.py       # layer by layer
python scripts/analysis/PPJAX_run_design_equiv_cc.py          # the manuscript design
python scripts/analysis/PPJAX_check_decoder_equiv_cc.py       # decoder + mpnn_sample
python scripts/analysis/PPJAX_compare_suite_cc.py             # the whole reference matrix
python scripts/analysis/PPJAX_locate_greedy_divergence_cc.py  # where the one knife-edge case splits
python scripts/analysis/PPJAX_measure_greedy_flip_cc.py       # and by how little
pytest                                                        # the same checks, as tests
```

The golden files are generated by running **the original engine in this same repo** through
`scripts/util/PPJAX_export_reference*_cc.py` — nothing is patched, and the port imports nothing from it
at run time. The large reference tensors (30–65 MB `.npz`) are not committed; regenerate them with those
scripts. The small JSON/NPY goldens the RNG and vocabulary tests need **are** committed, so `pytest` is
useful immediately and the heavier checks skip cleanly until you generate them.

### What the port adds

- **`jax.jit` and GPU** for the forward pass (featurisation → encoder → Potts head).
- **Differentiability.** `ppjax.contract` evaluates the Potts energy of a *soft* sequence without ever
  materialising the `[L, K, V, V]` tables — **18.5× fewer multiply-accumulates and no 40 MB tensor** — so
  the Potts term can sit inside a design loop as a per-step loss with a gradient.
- **No torch at run time**, including a from-scratch reader for torch's checkpoint format.
- **A second independent implementation is a cross-check on the first.** Writing it surfaced two
  undocumented torch RNG behaviours (above) and a host-dependence of float32 reductions that applies to
  the original too.

---

## Install (once) — with uv

```bash
cd Jax-ProtonPottsMPNN
./install.sh                     # uv venv (Python 3.12) + uv pip install -e ./foundry + extras
source .venv/bin/activate
pip install -e .                 # the JAX port (src/ppjax); pip install -e '.[cuda]' for GPU
```

`install.sh` is the original's: it runs `uv venv --clear` then a single
`uv pip install -e ./foundry -r requirements-extra.txt` (the one resolution keeps both the foundry core —
torch, lightning, atomworks[ml] — and the extras — the FLAML stack, jupyter, propka, and now jax), and
verifies `import mpnn` resolves inside this folder.

- **Python 3.12** is required (`mpnn`/`foundry` pin `>=3.12,<3.13`); the port itself needs only ≥ 3.11.
- **HBPLUS** is an external C binary (not pip-installable) used by the **labeller** and the **fold
  scoring** to read H-bond geometry. Point `HBPLUS_PATH` at your build.
  **Correction to the original README:** it is also needed to *featurise* a backbone for design —
  `prepare_potts_input` runs the bond annotation, and without `HBPLUS_PATH` it fails. You pay that once
  (see example 1); every design after it is torch- and HBPLUS-free.
- Let the install finish uninterrupted — a killed `uv pip install` can leave the venv half-written. If
  imports fail oddly, repair in place with
  `uv pip install --python .venv/bin/python --reinstall -e ./foundry -r requirements-extra.txt`.

| part | needs mpnn/torch | needs HBPLUS | needs FLAML stack | needs jax |
|------|:---:|:---:|:---:|:---:|
| **label a PDB** (`labeller/`) | ✅ | ✅ | ✅ | — |
| **featurise a backbone** (`ppjax.frontend`, once) | ✅ | ✅ | ✅ | — |
| **design a binder** (`inference/`, `src/ppjax/`) | — | — | — | ✅ |
| **score a fold** (`scoring/`) | ✅ | ✅ (pH-bonds) | — | — |
| **benchmarks** (`benchmarks/`) | ✅ | — | — | — |
| **train** (`training/`, reference) | ✅ | ✅ | — | — |

---

## Two examples

### 1. Label / featurise a PDB with the protonation pipeline

Assign a protonation state to every His/Asp/Glu in a structure — the labels the Potts model is trained on
and the design engine consumes. This runs the **transformation pipeline**
`prepare_potts_input(..., extended_vocab="v6")`: it strips hydrogens, runs HBPLUS, applies the FLAML
labeller, and attaches a per-residue `protonation_label` token (`-P` protonated · `-S` neutral His ·
`-D` deprotonated acid · `-A` ambiguous). **This half is not ported** — it is a C binary and a
gradient-boosted tree ensemble, neither of which has a JAX counterpart, and re-implementing them would
*break* numerical identity rather than preserve it.

```bash
HBPLUS_PATH=/path/to/hbplus python labeller/label_pdb.py     # -> labeller/outputs/protonation_labels.csv
```

The port consumes its output. Run it **once** per backbone and cache the result:

```python
from ppjax.frontend import featurise_structure, FeaturisedBackbone

featurise_structure("inference/examples/pdl1_seed_binder.pdb", binder_chain="A").save("pdl1.npz")
fb = FeaturisedBackbone.load("pdl1.npz")          # from here on: no torch, no HBPLUS
fb.token_res_name[:5], fb.features["S"].shape     # per-residue tokens, the encoded sequence
```

```text
chain  res_id res_name token
    A      32      HIS  HIS-S     # neutral at rest
    A      57      GLU  GLU-D     # deprotonated
    B      52      HIS  HIS-S
    …                             # 21 titratable residues, all neutral/deprotonated
```

Note the read-out: the apo seed binder carries **no** strongly-protonated residue at rest — which is
exactly why *design* (below) **pins** `HIS-P`/`ASP-P`/`GLU-P` centres deliberately rather than reading
them off the input.

### 2. Design a pH-switch binder

`PottsMPNNPHEngine` (now in `src/ppjax/engine.py`) places protonated centres and redesigns their
neighbourhood with **Potts-head block descent** — the same optimiser as the original, step for step.

```bash
python inference/design_ph.py                        # -> inference/outputs/
python inference/design_ph.py --features pdl1.npz    # reuse the cached featurisation
```

```python
from ppjax import PottsMPNNPHEngine, PHDesignCriteria, FeaturisedBackbone

eng = PottsMPNNPHEngine.from_checkpoint(CKPT, extended_vocab="v6")   # 30-token v6 model
fb  = FeaturisedBackbone.load("pdl1.npz")
ctx = eng.build_context(fb.features, fb.binder_chain,
                        token_res_id=fb.token_res_id, token_res_name=fb.token_res_name,
                        token_chain_id=fb.token_chain_id, region_masks=fb.region_masks)

crit = PHDesignCriteria(
    method="block_descent", backend="potts",          # the internal-campaign optimiser
    combined_lambda=0.3,                               # Eq (6): O = (1−λ)·zscore(H_stab) + λ·zscore(Σ sel)
    center_types=["HIS-P", "ASP-P", "GLU-P"],          # composition to place …
    placement_by="scan_potts",                         # … placement chooses the positions
    dep_map={"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]},  # v6 has no HID/HIE
    block_size=3, temperature=0.05, neighbour_k=16, max_mutations=20,
    seed_source="native", record_trajectory=True,
)
design_set = eng.run_ph_redesign(ctx=ctx, criteria_list=[crit], seed=0)
```

`PHDesignCriteria` is field-for-field and validation-for-validation identical to the original's, so
existing configs transfer unchanged.

**Many designs → the Pareto front.** `run_ph_redesign` takes a *list* of criteria, so N designs is just a
`combined_lambda` sweep from 0 (pure stability) to 1 (pure selectivity):

```python
from dataclasses import replace
import numpy as np

sweep = [replace(crit, combined_lambda=float(l), samples_per_site=1, record_trajectory=False)
         for l in np.round(np.linspace(0.0, 1.0, 12), 3)]
designs = eng.run_ph_redesign(ctx=ctx, criteria_list=sweep, seed=0)
```

The original's `n_jobs` CPU fork pool is not reproduced: it only changes the order results come back in,
and the serial path it is defined to equal is what runs here.

**The Potts energy as a design loss** — what the port adds over the original:

```python
import jax
from ppjax.contract import soft_potts_energy

out  = eng.forward(fb.features)                      # one jitted pass
E, g = jax.value_and_grad(lambda P: soft_potts_energy(
    eng.params, eng.cfg, out["h_V"], out["h_E"], out["E_idx"],
    out["residue_mask"], P))(P)                      # P: a SOFT sequence [L, V]
```

**Fold the designs.** `inference/fold_rf3.py` (the original, unported) folds a design with the packaged
RF3 engine; RF3 **code** ships here but the **weights (~3 GB) do not** — set `RF3_CKPT=/abs/rf3_*.ckpt`
and run on a **GPU**. `scoring/` then scores charge clashes and pH-sensitive H-bonds at the pinned centres.

**Every design is saved with its protonation states.** Outputs in `inference/outputs/`:

| file | what |
|------|------|
| `designs.fasta` | 1-letter canonical sequence (RF3-foldable) |
| `designs_states.fasta` | the **3-letter + protonation-state** sequence (`… ASP-P … HIS-P …`) |
| `designs.tsv` / `designs.json` | both sequence forms + energies + pinned centres |
| `trajectory.tsv` | the binder sequence (1-letter **and** 3-letter/protonation) **at every optimisation step** |
| `sweep_designs.tsv` | the N sweep designs (λ, stability, selectivity, both sequences, Pareto flag) |

`inference/outputs/` ships the **original's** demo outputs, including its `placement_scan.png`,
`optimisation_trajectory.png` and `pareto_front.png` (the plotting notebook is not ported). Running
`inference/design_ph.py` overwrites the text outputs in place, exactly as the original script does.

> Point it at your own backbone with `--pdb` / `--binder-chain` (or `inference/examples/example_meta.json`).
> The checkpoint's vocabulary **must** be `"v6"` or the 30-token weight load fails.

---

## Layout

| folder | what it holds | ported? |
|--------|---------------|---------|
| [`src/ppjax/`](src/ppjax/) | **the JAX port**: `rng` · `tokens` · `checkpoint` · `params` · `nn` · `layers` · `features` · `model` · `scorer` · `criteria` · `engine` · `decoder` · `contract` · `compat` · `paths` · `frontend` · `cli` | **new** |
| [`scripts/`](scripts/) | `run/` the worked example · `analysis/` the equivalence checks and divergence diagnostics · `util/` torch-side probes and golden-reference exporters | **new** |
| [`tests/`](tests/) | pytest mirror of the equivalence checks, plus guards for what the port refuses | **new** |
| [`foundry/`](foundry/) | a full verbatim copy of the `ph/foundry` monorepo — still the source of the **featurisation front end** (`prepare_potts_input`), the **FLAML labeller** (`transforms/ev6/`) and everything the unported halves use | original |
| [`labeller/`](labeller/) | the FLAML protonation labeller: `label_pdb.py`, the train path `01…05_*.py`, `pr_curve.py` (AUPR), and the trained models | original |
| [`inference/`](inference/) | `design_ph.py` **(now the JAX runner)** + `design_ph.ipynb` / `design_placement_scan.py` / `fold_rf3.py` (original, torch) + `examples/` | mixed |
| [`scoring/`](scoring/) | fold read-outs: `charge_clash` (geometry) + `annotate` (pH-sensitive H-bonds / salt bridges via HBPLUS+PLIP) | original |
| [`benchmarks/`](benchmarks/) | PKAD (pKa), MegaScale/FireProt (stability), binding AP, within-backbone, `placement_by_class.py` — all data in-folder | original |
| [`training/`](training/) | PottsMPNN + H-bond-head SLURM launchers — reference (the trainer itself is `mpnn.train`) | original |
| [`checkpoints/`](checkpoints/) | the v6 design checkpoint (21 MB) — read by both implementations | original |

### Where the core code lives

| component | original (torch) | here (JAX) |
|---|---|---|
| graph featurisation (kNN, atomwise RBF, chain-aware positional embedding) | [`graph_embeddings.py`](foundry/models/mpnn/src/mpnn/model/layers/graph_embeddings.py) | [`features.py`](src/ppjax/features.py) |
| message passing (`EncLayer` / `DecLayer` / PWFF) | [`message_passing.py`](foundry/models/mpnn/src/mpnn/model/layers/message_passing.py) | [`layers.py`](src/ppjax/layers.py) |
| masks + encoder | [`mpnn.py`](foundry/models/mpnn/src/mpnn/model/mpnn.py) | [`model.py`](src/ppjax/model.py) |
| causality masks, teacher forcing, autoregressive decode | [`mpnn.py`](foundry/models/mpnn/src/mpnn/model/mpnn.py) | [`decoder.py`](src/ppjax/decoder.py) |
| Potts head, energy helpers, Gibbs | [`pottsmpnn.py`](foundry/models/mpnn/src/mpnn/model/pottsmpnn.py) | [`model.py`](src/ppjax/model.py) · [`scorer.py`](src/ppjax/scorer.py) |
| design engine (placement + block-descent pH-redesign) | [`potts_mpnn_ph.py`](foundry/models/mpnn/src/mpnn/inference_engines/potts_mpnn_ph.py) | [`engine.py`](src/ppjax/engine.py) · [`criteria.py`](src/ppjax/criteria.py) |
| transformation pipeline (structure → model input) | [`potts_inference.py`](foundry/models/mpnn/src/mpnn/potts_inference.py) | **not ported** — used through [`frontend.py`](src/ppjax/frontend.py) |
| checkpoint loading | `torch.load` | [`checkpoint.py`](src/ppjax/checkpoint.py) (reads the zip format directly) |
| torch CPU RNG | `torch.*` | [`rng.py`](src/ppjax/rng.py) |
| *(new)* soft-sequence Potts energy for a design loss | — | [`contract.py`](src/ppjax/contract.py) |

### Everything else, one command each

| I want to… | run | out |
|------------|-----|-----|
| **label a PDB** | `HBPLUS_PATH=… python labeller/label_pdb.py` | `labeller/outputs/protonation_labels.csv` |
| **featurise a backbone** (once) | `HBPLUS_PATH=… python -c "from ppjax.frontend import featurise_structure; featurise_structure('x.pdb','A').save('x.npz')"` | `x.npz` |
| **design a binder** | `python inference/design_ph.py --features x.npz` | `designs*.fasta` / `.tsv` / `trajectory.tsv` (with protonation states) |
| **sweep the Pareto front** | `python inference/design_ph.py --n-designs 20` | `sweep_designs.tsv` |
| **check the port against the original** | `python scripts/analysis/PPJAX_compare_suite_cc.py` | per-case IDENTICAL / DIFFERS |
| **run the tests** | `pytest` | — |
| **retrain the labeller** | `python labeller/05_train_one.py HIS features 1200` | `labeller/models/automl_feature_HIS/` |
| **labeller AUPR** | `python labeller/pr_curve.py` | `labeller/{his,acid}_pr.png` (AP HIS 0.70, acids 0.33) |
| **score a fold** | `HBPLUS_PATH=… python scoring/score_example.py` | pH-bond / charge-clash counts |
| **pKa benchmark** | `PROTON_ROOT=$PWD python -m mpnn.scripts.eval_pkad --checkpoints checkpoints/…/epoch-0125.ckpt` | `pkad_*.csv` + scatter |
| **stability benchmark** | `EV6_OUT_SUBDIR=his0.3_acid0.06 python benchmarks/stability_benchmark.py` | ΔΔG CSVs (GPU recommended) |
| **binding AP** | `python benchmarks/binding_ap.py` | `benchmarks/results/summary_global_ap.png` |

Benchmark scripts derive the package root from their location (override with `PROTON_ROOT=/abs/path`);
the port's scripts use the same convention, plus `PROTON_CKPT` to point straight at a `.ckpt`.

---

## Limits — stated plainly

- **Float arithmetic is float32, not bitwise.** XLA and ATen associate reductions differently, so energies
  agree to ~1e-6 relative. That is enough for every discrete decision in the reference matrix except one,
  and that one is understood (below). The claim is "same sequences across the measured matrix, with the
  single knife-edge named" — not a proof of equivalence.
- **Results are host-dependent at the same level.** XLA picks its reduction tiling from the CPU, so
  `H(S_native)` comes out bit-identical to the original on one machine and 1.17e-2 off (of 5.2e4) on three
  others, with `cond_energy` identical on all four. Any "bit-exact" number measured on a single machine is
  that machine's result, not a property of the port.
- **Near-ties at a discrete selection are where the port actually diverges** — not round-off in the
  energies. The measured instance: `greedy_energy_block` re-ranks all designable positions by their
  single-residue improvement at *every* step and takes the top `block_size`. On the PD-L1 example the 3rd
  and 4th improvements differ by **3.05e-05** at step 12, below the float32 agreement, so the two engines
  pick a different third block member and the trajectories separate — having been identical through step
  11 (ΔH = 0.0000). At `temperature=0` that method is identical, and `block_descent` is identical at both
  `T=0` and `T=0.05`, because it draws its block from `E_idx` order — integers — and is structurally
  immune. The same case comes out identical on a different machine with no code change, which is the
  cleanest confirmation of the mechanism one can ask for. `ppjax.compat` emits a `PPJAXCompatWarning` at
  exactly that cut rather than diverging quietly.
- **Exact ties are a second such place.** `torch.argsort` and `torch.topk` use an *unstable* sort whose tie
  order matches neither numpy's stable nor its quicksort ordering. When two placement scores or two kNN
  distances are bitwise equal, the two implementations may pick differently; `ppjax.compat` warns there too.
- **`randn` values differ by ≤ 1e-6** (a libm `cosf`/`sinf`/`logf` difference; neither torch's nor numpy's
  is correctly rounded, and forcing glibc via `ctypes` does not close it). The stream *position* is exact.
  This affects the sampled decoding order under `causality_pattern="auto_regressive"` and the perturbation
  when `structure_noise > 0`; `conditional_minus_self` — the design field — is invariant to the decoding
  order by construction. Measured: the 916 autoregressively sampled tokens of `mpnn_sample` at seed 0 are
  identical.
- **`symmetry_equivalence_group` is not ported and raises.** The original draws the decoding-order noise
  per symmetry *group* (`randn((B, G))`), not per residue, so the whole RNG stream would shift. Nothing in
  this pipeline sets it; the port refuses rather than diverging quietly.
- **Training is out of scope** — no loss, no trainer, no backward-pass equivalence claim. (`contract.py`
  is differentiable, but it is a design-time loss, not a training path.)

---

## License

Two licences apply, and they do not overlap.

| material | licence |
|---|---|
| **The JAX port** — `src/ppjax/`, `scripts/`, `tests/`, `inference/design_ph.py`, `pyproject.toml` | **PolyForm Noncommercial 1.0.0** — [`LICENSE-ppjax`](LICENSE-ppjax) |
| **Proton-PottsMPNN** — `labeller/`, `inference/` (the rest), `scoring/`, `benchmarks/`, `training/`, `checkpoints/` **including the model weights**, `figures/`, and the pH-design additions to `mpnn` | **MIT**, © 2026 Christian P. Jacobsen — [`LICENSE`](LICENSE), unchanged |
| **rc-foundry** — [`foundry/`](foundry/) | **BSD 3-Clause**, © 2025 Institute for Protein Design, University of Washington — [`foundry/LICENSE.md`](foundry/LICENSE.md), unchanged |
| **This README**, adapted from the original's | MIT, with the original's notice retained |

The noncommercial term applies **only** to the port's own code. It does not and cannot restrict the
original work or the model weights, which remain available under MIT from
[the upstream repository](https://github.com/christian-creator/ProtonPottsMPNN). See
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for the full texts.

## Citation

If you use this in academic work, cite the original manuscript — the science, the model and the weights
are theirs:

> Jacobsen et al. (2026), *pH-sensitive binder design with Proton-PottsMPNN*.
> Birnbaum & Keating, *PNAS* 2026, [10.1073/pnas.2535494123](https://www.pnas.org/doi/10.1073/pnas.2535494123).

You may additionally cite this repository for the JAX port.
