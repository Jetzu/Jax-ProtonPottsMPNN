# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX.
# Purpose: the torch-compatible RNG must reproduce torch's CPU stream exactly - this is what makes
#          the ported generation bit-identical for a given seed.
import json
from pathlib import Path

import numpy as np
import pytest

from ppjax.rng import TorchCPUGenerator

DATA = Path(__file__).resolve().parent / "data"

# Measured with torch 2.14.1 on CPU by scripts/util/PPJAX_probe_torch_rng_cc.py.
TORCH_RAND_F32_SEED0 = [0.49625658988952637, 0.7682217955589294, 0.08847743272781372,
                        0.13203048706054688, 0.30742281675338745, 0.6340786814689636,
                        0.4900934100151062, 0.8964447379112244]
TORCH_RAND_F64_SEED0 = [0.9700530018065531, 0.707819864399788, 0.45938294312745087,
                        0.9207476841219603, 0.6450241201227648, 0.7911478921803037,
                        0.17860617520075095, 0.3511076243939284]
TORCH_RANDINT17_SEED0 = [7, 3, 12, 16, 2, 0, 16, 10, 6, 5]
TORCH_RANDPERM10_SEED0 = [4, 1, 7, 5, 3, 9, 0, 8, 6, 2]
TORCH_MULTINOMIAL_V30_SEED0 = [20, 28, 23, 23, 8, 0, 0, 20, 17, 8, 8, 20]
TORCH_MULTINOMIAL_V27000_SEED0 = [25043, 10801, 7341, 5737, 536, 10811, 23519, 8796, 16740,
                                  20287, 22526, 8854]
TORCH_RAND_AFTER_MULTINOMIAL = [0.7262365221977234, 0.7010802030563354, 0.2038237452507019,
                                0.6510535478591919]


def test_uniform_float32_matches_torch():
    g = TorchCPUGenerator(0)
    assert g.uniform_float32(8).tolist() == TORCH_RAND_F32_SEED0


def test_uniform_float64_matches_torch():
    g = TorchCPUGenerator(0)
    assert g.uniform_float64(8).tolist() == TORCH_RAND_F64_SEED0


def test_randint_matches_torch():
    g = TorchCPUGenerator(0)
    assert [int(g.randint(17, 1)[0]) for _ in range(10)] == TORCH_RANDINT17_SEED0


def test_randperm_matches_torch():
    g = TorchCPUGenerator(0)
    assert g.randperm(10).tolist() == TORCH_RANDPERM10_SEED0


@pytest.mark.parametrize("v,expected", [(30, TORCH_MULTINOMIAL_V30_SEED0),
                                        (27000, TORCH_MULTINOMIAL_V27000_SEED0)])
def test_multinomial_matches_torch(v, expected):
    """Covers 87 MT19937 twists at V=27000 - the block-descent pick size for block_size=3."""
    probs = np.load(DATA / f"PPJAX_rng_probs_V{v}.npy")
    g = TorchCPUGenerator(0)
    assert [g.multinomial1(probs) for _ in range(12)] == expected


def test_multinomial_consumes_the_right_number_of_words():
    probs = np.load(DATA / "PPJAX_rng_probs_probe.npy")
    g = TorchCPUGenerator(0)
    assert g.multinomial1(probs) == 20
    assert g.uniform_float32(4).tolist() == TORCH_RAND_AFTER_MULTINOMIAL


def test_normal_keeps_the_stream_aligned():
    """randn VALUES differ from torch by <=1e-6 (libm), but the stream position must match."""
    probe = DATA / "PPJAX_torch_normal_probe.json"
    if not probe.exists():
        pytest.skip("normal probe not exported")
    d = json.loads(probe.read_text())
    g = TorchCPUGenerator(0)
    for n in (4, 16, 17, 32, 229):
        g.manual_seed(0)
        got = g.normal_float32(n)
        want = np.asarray(d[f"randn_f32_{n}"], dtype=np.float32)
        assert np.abs(got - want).max() <= 1e-5
        assert g.uniform_float32(3).tolist() == d[f"after_randn_{n}"]
    g.manual_seed(0)
    assert np.array_equal(g.normal_float32(4), np.asarray(d["randn_f32_4"], np.float32))
