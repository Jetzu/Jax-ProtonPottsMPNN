# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX.
# Purpose: the acceptance criterion - the ported engine must GENERATE the same sequences as the
#          torch original for the same seed, across the reference design matrix.
import json
from pathlib import Path

import numpy as np
import pytest

from ppjax.paths import resolve_checkpoint

from ppjax.criteria import PHDesignCriteria

DATA = Path(__file__).resolve().parent / "data"

MANUSCRIPT_CRITERIA = dict(
    method="block_descent", backend="potts", temperature=0.05, samples_per_site=2,
    block_size=3, combined_lambda=0.3, seed_source="native",
    center_types=["HIS-P", "ASP-P", "GLU-P"],
    dep_map={"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]},
    forbidden_tokens=["HIS-A", "ASP-A", "GLU-A", "UNK"], placement_by="scan_potts",
    placement_region=["all"], repetitive_window_parents=["ARG", "LYS", "HIS", "ASP", "GLU"],
    repetitive_window_radius=2, repetitive_window_weight=1.0, neighbour_k=16,
    max_mutations=20, record_trajectory=True)


@pytest.fixture(scope="module")
def ref_designs(ref_dir):
    path = ref_dir / "PPJAX_reference_designs.json"
    if not path.exists():
        pytest.skip("reference designs not exported")
    return json.loads(path.read_text())


def test_manuscript_designs_are_identical(engine, context, ref_designs):
    designs = engine.run_ph_redesign(ctx=context,
                                     criteria_list=[PHDesignCriteria(**MANUSCRIPT_CRITERIA)],
                                     seed=0)
    got = {d.design_id(): d for d in designs}
    ref = {r["design_id"]: r for r in ref_designs}
    assert set(got) == set(ref), "design ids differ (placement or scheme label changed)"
    for did, r in ref.items():
        assert " ".join(got[did].extended_tokens) == r["extended_tokens"], did
        assert abs(got[did].final_potts_energy - r["potts_energy"]) < 1e-1, did


def test_trajectory_matches(engine, context, ref_designs):
    """Not just the endpoint: every recorded block-descent step must be the same sequence."""
    designs = engine.run_ph_redesign(ctx=context,
                                     criteria_list=[PHDesignCriteria(**MANUSCRIPT_CRITERIA)],
                                     seed=0)
    ref = {r["design_id"]: r for r in ref_designs}
    for d in designs:
        r_traj = ref[d.design_id()]["energy_trajectory"]
        assert len(d.energy_trajectory) == len(r_traj)
        for a, b in zip(d.energy_trajectory, r_traj):
            assert a["canonical_sequence"] == b["canonical_sequence"], (d.design_id(), a["step"])


def test_the_seed_actually_drives_the_sampler(engine, context):
    """Different seeds must give different sequences, and the same seed must be reproducible.
    Without this, every 'identical to torch' result could be an artefact of a dead RNG."""
    crit = PHDesignCriteria(**{**MANUSCRIPT_CRITERIA, "record_trajectory": False})
    runs = {s: [d.canonical_sequence for d in
                engine.run_ph_redesign(ctx=context, criteria_list=[crit], seed=s)]
            for s in (0, 1, 7)}
    assert all(len(v) == 2 for v in runs.values())
    assert runs[0] != runs[1] and runs[0] != runs[7], "seeds produced identical sequences"
    again = [d.canonical_sequence for d in
             engine.run_ph_redesign(ctx=context, criteria_list=[crit], seed=0)]
    assert again == runs[0], "the same seed was not reproducible"


# Cases known to sit on a knife edge. These are NOT skipped: they run and are expected to match,
# but a mismatch is reported as xfail-with-cause instead of a hard failure, because the decision
# genuinely can go either way on a given machine.
#
# Measured for pdl1/greedy_energy_block: that method re-ranks all 114 designable positions by
# single-residue improvement at EVERY step and takes the top block_size. At step 12 the 3rd and
# 4th improvements differ by 3.05e-05 - below the port's float32 agreement with torch - so which
# position joins the block is decided by noise. Trajectories are identical through step 11
# (dH = 0.0000). Observed DIFFERENT on t38cn027 and IDENTICAL on t38cn030, with no code change:
# XLA associates the reduction differently per CPU.
# scripts/analysis/PPJAX_{locate_greedy_divergence,measure_greedy_flip,check_greedy_probe}_cc.py
KNIFE_EDGE = {
    "pdl1/greedy_energy_block":
        "greedy_energy_block's per-step block cut is decided by a 3.05e-05 improvement gap at "
        "step 12, below the float32 agreement with torch; host-dependent",
}


def _suite():
    path = DATA / "ref_suite" / "PPJAX_reference_suite.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@pytest.mark.parametrize("case", sorted(_suite()) or ["<no suite>"])
def test_reference_suite_case(case, ref_meta):
    suite = _suite()
    if not suite:
        pytest.skip("reference suite not exported")
    blob = suite[case]
    if "error" in blob:
        # A reference the ORIGINAL could not produce is not evidence of equivalence; make it
        # visibly non-green rather than an indistinguishable skip.
        pytest.xfail(f"torch reference errored for this case: {blob['error'][:120]}")
    from ppjax.engine import PottsMPNNPHEngine
    structure = blob.get("structure", case.split("/")[0])
    npz = DATA / "ref_suite" / f"PPJAX_ctx_{structure}.npz"
    if not npz.exists():
        pytest.skip(f"missing {npz}")
    d = np.load(npz)
    eng = PottsMPNNPHEngine.from_checkpoint(resolve_checkpoint(ref_meta.get("ckpt")), extended_vocab="v6")
    ctx = eng.build_context(
        {k[3:]: d[k] for k in d.files if k.startswith("ni_")}, "A",
        token_res_id=d["token_res_id"], token_res_name=d["token_res_name"],
        token_chain_id=d["token_chain_id"],
        region_masks={n: d[f"region_{n}"] for n in ("interface", "core", "surface")
                      if f"region_{n}" in d.files})
    crits = [PHDesignCriteria(**{k: v for k, v in c.items()
                                 if k in PHDesignCriteria.__dataclass_fields__})
             for c in blob["criteria"]]
    ds = eng.run_ph_redesign(ctx=ctx, criteria_list=crits, seed=blob["seed"],
                             initial_sequences=blob.get("initial_sequences"))
    got = {x.design_id(): x for x in ds}
    ref = {r["design_id"]: r for r in blob["designs"]}
    mismatch = (set(got) != set(ref)) or any(
        " ".join(got[did].extended_tokens) != r["extended_tokens"]
        for did, r in ref.items() if did in got)
    if mismatch and case in KNIFE_EDGE:
        pytest.xfail(f"{case}: {KNIFE_EDGE[case]}")
    assert set(got) == set(ref), f"{case}: design ids differ"
    for did, r in ref.items():
        assert " ".join(got[did].extended_tokens) == r["extended_tokens"], f"{case}/{did}"
