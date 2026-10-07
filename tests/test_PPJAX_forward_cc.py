# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX.
# Purpose: layer-by-layer equivalence of the JAX forward against the torch golden tensors. The kNN
#          graph and every mask must be EXACT; float tensors are compared at float32 round-off.
import numpy as np
import pytest


def _maxabs(a, b):
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())


@pytest.fixture(scope="module")
def forward(engine, features):
    return engine.forward(features)


def test_knn_graph_is_exact(forward, ref_npz):
    """E_idx is an index tensor - any difference changes the model, not just its rounding."""
    assert np.array_equal(np.asarray(forward["E_idx"]), ref_npz["cap_E_idx"])


def test_masks_are_exact(forward, ref_npz):
    for key, ref in (("residue_mask", "ni_residue_mask"),
                     ("known_residue_mask", "ni_known_residue_mask"),
                     ("designed_residue_mask", "ni_designed_residue_mask"),
                     ("mask_for_loss", "ni_mask_for_loss")):
        assert np.array_equal(np.asarray(forward[key]), ref_npz[ref]), key


def test_edge_features(forward, ref_npz):
    assert _maxabs(forward["E_raw"], ref_npz["cap_E_raw"]) < 1e-4
    assert _maxabs(forward["E"], ref_npz["cap_E"]) < 1e-4


def test_encoder_states(forward, ref_npz):
    assert _maxabs(forward["h_V"], ref_npz["cap_h_V"]) < 1e-4
    assert _maxabs(forward["h_E"], ref_npz["cap_h_E"]) < 1e-3


def test_potts_tables(forward, ref_npz):
    """The Potts tables drive every design decision; ~4e-5 on a +-37 range is float32 noise."""
    etab = np.asarray(forward["etab_out"])
    ref = ref_npz["etab_out"]
    assert etab.shape == ref.shape
    assert _maxabs(etab, ref) < 1e-3
    assert _maxabs(etab, ref) / max(float(np.abs(ref).max()), 1e-9) < 1e-5


def test_self_edge_is_diagonal(forward):
    """field_source='self_edge' zeroes the off-diagonal of slot 0 - consumers read the field
    straight off that diagonal, so a non-zero off-diagonal would silently corrupt every energy."""
    slot0 = np.asarray(forward["etab_out"])[0, :, 0]
    V = slot0.shape[-1]
    off = slot0 * (1 - np.eye(V, dtype=slot0.dtype))
    assert np.abs(off).max() == 0.0


def test_scorer_matches_reference(context, ref_npz):
    assert _maxabs(context.scorer.cond_energy(context.S_native),
                   ref_npz["cond_energy_native"]) < 1e-2
    assert abs(context.scorer.H_of(context.S_native) - float(ref_npz["H_native"])) < 1e-1


def test_cond_energy_at_matches_cond_energy(context):
    """The fast per-position scorer must agree with the full table it shortcuts."""
    full = np.asarray(context.scorer.cond_energy(context.S_native))
    for i in (0, 7, 113, context.L - 1):
        assert _maxabs(context.scorer.cond_energy_at(context.S_native, i), full[i]) < 1e-2


def test_potts_delta_is_consistent_with_the_hamiltonian(context):
    """Changing one residue must move H by exactly the conditional-energy difference - the
    identity the whole block-descent objective is built on."""
    S = context.S_native.copy()
    pos = int(context.chA_free_idx[3])
    row = np.asarray(context.scorer.cond_energy_at(S, pos))
    H0 = context.scorer.H_of(S)
    for a in (0, 5, 19):
        S2 = S.copy(); S2[pos] = a
        assert abs((context.scorer.H_of(S2) - H0) - (row[a] - row[int(S[pos])])) < 5e-2
