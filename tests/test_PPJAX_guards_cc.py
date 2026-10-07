# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX after the N001 red-team.
# Purpose: the port must REFUSE inputs it cannot reproduce, and must WARN where torch's unstable
#          tie-breaking makes a difference possible. Silence on either would be the failure mode
#          this project exists to avoid.
import warnings

import numpy as np
import pytest

from ppjax.compat import PPJAXCompatWarning, check_knn_tie_boundary, warn_on_ties


def test_symmetry_groups_raise_rather_than_diverge(engine, features):
    """Upstream draws the decoding-order noise per GROUP, not per residue, so the RNG stream
    would diverge silently. The port must refuse."""
    feats = dict(features)
    feats["symmetry_equivalence_group"] = np.zeros_like(np.asarray(features["S"]))
    with pytest.raises(NotImplementedError, match="symmetry_equivalence_group"):
        engine.forward(feats)


def test_structure_noise_is_applied_not_dropped(engine, features):
    """A silently ignored structure_noise would give noiseless features AND a misaligned stream."""
    from ppjax.model import apply_structure_noise
    from ppjax import rng as trng
    X = np.asarray(features["X"])
    assert apply_structure_noise(X, 0.0) is X
    trng.manual_seed(0)
    noised = apply_structure_noise(X, 0.1)
    assert noised.shape == X.shape
    assert not np.array_equal(noised, X)
    assert 0.05 < float(np.std(noised - X)) < 0.2


def test_structure_noise_consumes_the_stream_in_place(features):
    """The normals must come out of the SAME generator, before any later draw."""
    from ppjax.model import apply_structure_noise
    from ppjax import rng as trng
    X = np.asarray(features["X"])[:, :4]
    trng.manual_seed(0)
    apply_structure_noise(X, 0.1)
    after_noise = trng.default_generator().uniform_float32(2).tolist()
    trng.manual_seed(0)
    fresh = trng.default_generator().uniform_float32(2).tolist()
    assert after_noise != fresh, "structure noise drew nothing from the generator"


def test_tie_warning_fires_and_stays_quiet_otherwise():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_on_ties(np.array([1.0, 2.0, 2.0, 3.0], np.float32), "unit test")
        assert len(w) == 1 and issubclass(w[0].category, PPJAXCompatWarning)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_on_ties(np.array([1.0, 2.0, 3.0], np.float32), "unit test")
        assert len(w) == 0


def test_knn_tie_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert check_knn_tie_boundary(np.array([[1.0, 1.0], [2.0, 3.0]], np.float32)) == 1
        assert len(w) == 1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert check_knn_tie_boundary(np.array([[1.0, 2.0]], np.float32)) == 0
        assert len(w) == 0


def test_the_shipped_structure_has_no_knn_ties(engine, features):
    """If this ever fires, the exactness of E_idx is no longer guaranteed for this input."""
    out = engine.forward(features)
    assert check_knn_tie_boundary(np.asarray(out["knn_boundary"])) == 0


def test_the_shipped_graph_has_no_self_loops(context):
    """The one-row shortcut in _selective_energy_sum is an identity only without self-loops."""
    assert context.scorer.has_self_loops is False


def test_a_saved_featurisation_round_trips(tmp_path, features, ref_npz):
    """Regression: `save` omits the optional arrays a normal featurisation does not have
    (symmetry_equivalence_group, and temperature for some callers); `load` must not then demand
    them. Caught by running the documented two-environment workflow, not by the unit tests."""
    from ppjax.frontend import FeaturisedBackbone
    fb = FeaturisedBackbone(
        features={k: v for k, v in features.items()
                  if k in ("X", "X_m", "S", "R_idx", "chain_labels", "residue_mask",
                           "designed_residue_mask")},
        token_res_id=ref_npz["token_res_id"], token_res_name=ref_npz["token_res_name"],
        token_chain_id=ref_npz["token_chain_id"], region_masks={}, binder_chain="A")
    p = tmp_path / "rt.npz"
    fb.save(str(p))
    back = FeaturisedBackbone.load(str(p))
    assert back.binder_chain == "A"
    assert back.features["S"].shape == fb.features["S"].shape
    assert "symmetry_equivalence_group" not in back.features


def test_load_still_refuses_a_truly_incomplete_file(tmp_path):
    import numpy as np
    from ppjax.frontend import FeaturisedBackbone
    p = tmp_path / "bad.npz"
    np.savez(p, ni_S=np.zeros((1, 4), np.int64), token_res_id=np.arange(4),
             token_res_name=np.array(["ALA"] * 4), token_chain_id=np.array(["A"] * 4))
    with pytest.raises(KeyError, match="missing featurisation arrays"):
        FeaturisedBackbone.load(str(p))
