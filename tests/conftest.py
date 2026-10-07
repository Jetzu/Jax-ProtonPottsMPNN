# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX.
# Purpose: shared fixtures - the golden-reference paths and a cached engine/context.
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))      # before any ppjax import
DATA = ROOT / "tests" / "data"

from ppjax.paths import resolve_checkpoint  # noqa: E402  (needs the sys.path line above)


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"golden reference {path} not exported "
                    f"(run scripts/util/PPJAX_export_reference_cc.py in the torch env)")
    return path


@pytest.fixture(scope="session")
def ref_dir():
    return _require(DATA / "ref_pdl1")


@pytest.fixture(scope="session")
def ref_meta(ref_dir):
    return json.loads((ref_dir / "PPJAX_reference_meta.json").read_text())


@pytest.fixture(scope="session")
def ref_npz(ref_dir):
    return np.load(_require(ref_dir / "PPJAX_reference_ctx.npz"))


@pytest.fixture(scope="session")
def engine(ref_meta):
    from ppjax.engine import PottsMPNNPHEngine
    return PottsMPNNPHEngine.from_checkpoint(resolve_checkpoint(ref_meta.get("ckpt")), extended_vocab="v6")


@pytest.fixture(scope="session")
def features(ref_npz):
    return {k[3:]: ref_npz[k] for k in ref_npz.files if k.startswith("ni_")}


@pytest.fixture(scope="session")
def context(engine, features, ref_npz, ref_meta):
    return engine.build_context(
        features, ref_meta["binder_chain"], token_res_id=ref_npz["token_res_id"],
        token_res_name=ref_npz["token_res_name"], token_chain_id=ref_npz["token_chain_id"],
        region_masks={n: ref_npz[f"region_{n}"] for n in ("interface", "core", "surface")
                      if f"region_{n}" in ref_npz.files})
