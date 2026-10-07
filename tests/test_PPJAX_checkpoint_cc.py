# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX.
# Purpose: the torch-free checkpoint reader must recover the state dict byte-for-byte and the
#          architecture flags the loader infers from it.
import numpy as np
import pytest

from ppjax.paths import resolve_checkpoint

from ppjax.checkpoint import load_state_dict
from ppjax.params import build_params, infer_etab_source, infer_field_source, infer_shapes


@pytest.fixture(scope="module")
def flat(ref_meta):
    return load_state_dict(resolve_checkpoint(ref_meta.get("ckpt")))


def test_reads_every_parameter(flat):
    assert len(flat) == 120
    assert all(isinstance(v, np.ndarray) for v in flat.values())
    assert flat["W_s.weight"].shape == (30, 128)
    assert flat["etab_out.weight"].shape == (900, 128)


def test_no_torch_import_needed():
    """ppjax must stay importable without torch; the reader is the only thing that could need it."""
    import ppjax.checkpoint as mod
    assert "torch" not in mod.__dict__


def test_infers_architecture(flat):
    assert infer_etab_source(flat) == "edge"
    assert infer_field_source(flat) == "self_edge"
    s = infer_shapes(flat)
    assert (s["vocab_size"], s["hidden_dim"]) == (30, 128)
    assert s["num_encoder_layers"] == 3 and s["num_decoder_layers"] == 3


def test_nests_into_the_pytree(flat):
    p = build_params(flat)
    assert isinstance(p["encoder_layers"], list) and len(p["encoder_layers"]) == 3
    assert p["encoder_layers"][0]["W1"]["weight"].shape == (128, 384)
    assert p["graph_featurization_module"]["edge_embedding"]["weight"].shape == (128, 416)
