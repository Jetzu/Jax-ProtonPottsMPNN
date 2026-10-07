# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX after the N001 red-team flagged that
#         src/ppjax/decoder.py had zero assertion coverage.
# Purpose: assert the decoder half of the generation path against the torch golden outputs -
#          the design field, autoregressive sampling, and every mpnn-backend design method.
import json
from pathlib import Path

import numpy as np
import pytest

from ppjax import rng as trng
from ppjax.criteria import PHDesignCriteria

DATA = Path(__file__).resolve().parent / "data" / "ref_decoder"


@pytest.fixture(scope="module")
def dec_npz():
    p = DATA / "PPJAX_decoder_ref.npz"
    if not p.exists():
        pytest.skip("decoder reference not exported "
                    "(scripts/util/PPJAX_export_decoder_reference_cc.py)")
    return np.load(p)


@pytest.fixture(scope="module")
def dec_json():
    p = DATA / "PPJAX_decoder_ref.json"
    if not p.exists():
        pytest.skip("decoder reference not exported")
    return json.loads(p.read_text())


def test_decoder_is_built(context):
    assert context.decoder is not None, "no decoder: backend='mpnn' would be unusable"


def test_design_field_matches_torch(context, dec_npz):
    trng.manual_seed(0)
    got = context.field_mpnn(context.S_native)
    ref = dec_npz["field_mpnn_native"]
    assert got.shape == ref.shape
    assert np.abs(got.astype(np.float64) - ref.astype(np.float64)).max() < 1e-2


def test_design_field_with_a_masked_position(context, dec_npz):
    trng.manual_seed(0)
    got = context.field_mpnn(dec_npz["S_mpnn_unk_input"])
    assert np.abs(got.astype(np.float64)
                  - dec_npz["field_mpnn_unk"].astype(np.float64)).max() < 1e-2


def test_autoregressive_sampling_is_token_identical(context, dec_npz):
    """229 sequential multinomial draws per sample on top of a sampled decoding order - the most
    RNG-sensitive path in the whole port."""
    trng.manual_seed(0)
    got = context.sample_decoder(4)
    ref = dec_npz["mpnn_sample_seed0"]
    assert got.shape == ref.shape
    assert np.array_equal(got, ref), f"{int((got != ref).sum())}/{ref.size} tokens differ"


@pytest.mark.parametrize("case", ["autoregressive_potts_sel", "autoregressive_decoder_sel",
                                  "scan_mpnn_placement", "mpnn_sample"])
def test_mpnn_backend_designs_are_identical(engine, context, dec_json, case):
    blob = dec_json[case]
    if "error" in blob:
        pytest.xfail(f"torch reference errored: {blob['error'][:120]}")
    crit = PHDesignCriteria(**{k: v for k, v in blob["criteria"].items()
                               if k in PHDesignCriteria.__dataclass_fields__})
    ds = engine.run_ph_redesign(ctx=context, criteria_list=[crit], seed=0)
    got = {d.design_id(): d for d in ds}
    ref = {r["design_id"]: r for r in blob["designs"]}
    assert set(got) == set(ref), f"{case}: design ids differ"
    for did, r in ref.items():
        assert " ".join(got[did].extended_tokens) == r["extended_tokens"], f"{case}/{did}"
