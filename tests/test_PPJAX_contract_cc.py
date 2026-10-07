# Origin: cc (2026-10-02, from the BCNat session)
# Purpose: assert that ppjax.contract's factored soft-sequence energy equals the materialised
#          compute_potts_context path, for a one-hot and for a random soft sequence, on the
#          shipped reference context - the correctness gate for using it as a design loss.
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ppjax import model as M
from ppjax.contract import potts_edge_energies, soft_potts_energy


def _materialised(params, cfg, h_V, h_E, E_idx, residue_mask, sequence):
    """The energy the obvious way: build the tables, then contract them."""
    etab, idx, _ = M.compute_potts_context(params, cfg, h_V, h_E, E_idx, residue_mask)
    batch = jnp.arange(sequence.shape[0])[:, None, None]
    neighbour = sequence[batch, idx]
    return jnp.einsum("nia,nikav,nikv->nik", sequence, etab, neighbour)


@pytest.fixture(scope="module")
def forward(engine, features):
    params, cfg = engine.params, engine.cfg
    unknown = tuple(getattr(engine, "unknown_indices", ()) or ())
    out = M.forward_potts(params, engine.encoding, cfg, unknown, features)
    return params, cfg, out


def _sequences(length, vocab, seed=0):
    key = jax.random.PRNGKey(seed)
    hot_key, soft_key = jax.random.split(key)
    hard = jax.nn.one_hot(jax.random.randint(hot_key, (1, length), 0, vocab), vocab)
    soft = jax.nn.softmax(jax.random.normal(soft_key, (1, length, vocab)), axis=-1)
    return hard, soft


def test_factored_edge_energies_match_the_materialised_tables(forward):
    params, cfg, out = forward
    h_V, h_E, E_idx, mask = out["h_V"], out["h_E"], out["E_idx"], out["residue_mask"]
    length, vocab = h_E.shape[1], int(cfg.vocab_size)
    for name, sequence in zip(("one-hot", "soft"), _sequences(length, vocab)):
        reference = _materialised(params, cfg, h_V, h_E, E_idx, mask, sequence)
        factored = potts_edge_energies(params, cfg, h_V, h_E, E_idx, mask, sequence)
        scale = float(jnp.abs(reference).max())
        assert factored.shape == reference.shape
        assert np.allclose(np.asarray(factored), np.asarray(reference), rtol=0, atol=2e-3 * max(scale, 1.0)), \
            f"{name}: max abs diff {float(jnp.abs(factored - reference).max()):.3e} on a {scale:.3e} scale"


def test_total_energy_matches_and_row_restriction_is_a_partial_sum(forward):
    params, cfg, out = forward
    h_V, h_E, E_idx, mask = out["h_V"], out["h_E"], out["E_idx"], out["residue_mask"]
    length, vocab = h_E.shape[1], int(cfg.vocab_size)
    _, sequence = _sequences(length, vocab, seed=3)
    reference = _materialised(params, cfg, h_V, h_E, E_idx, mask, sequence).sum()
    total = soft_potts_energy(params, cfg, h_V, h_E, E_idx, mask, sequence)
    assert np.allclose(float(total), float(reference), rtol=2e-5, atol=1.0)

    rows = jnp.zeros((1, length)).at[:, : length // 2].set(1.0)
    part = soft_potts_energy(params, cfg, h_V, h_E, E_idx, mask, sequence, rows=rows)
    rest = soft_potts_energy(params, cfg, h_V, h_E, E_idx, mask, sequence, rows=1.0 - rows)
    assert np.allclose(float(part + rest), float(total), rtol=2e-5, atol=1.0)


def test_it_is_differentiable_in_the_sequence(forward):
    params, cfg, out = forward
    h_V, h_E, E_idx, mask = out["h_V"], out["h_E"], out["E_idx"], out["residue_mask"]
    length, vocab = h_E.shape[1], int(cfg.vocab_size)

    def energy(logits):
        sequence = jax.nn.softmax(logits, axis=-1)
        return soft_potts_energy(params, cfg, h_V, h_E, E_idx, mask, sequence)

    logits = jax.random.normal(jax.random.PRNGKey(7), (1, length, vocab))
    gradient = jax.grad(energy)(logits)
    assert gradient.shape == logits.shape
    assert bool(jnp.isfinite(gradient).all())
    assert float(jnp.abs(gradient).max()) > 0.0


def test_an_unsupported_head_is_refused_rather_than_misscored(forward):
    params, cfg, out = forward
    h_V, h_E, E_idx, mask = out["h_V"], out["h_E"], out["E_idx"], out["residue_mask"]
    length, vocab = h_E.shape[1], int(cfg.vocab_size)
    _, sequence = _sequences(length, vocab)
    other = type(cfg)(**{**cfg.__dict__, "etab_source": "node_edge_node"}) if hasattr(cfg, "__dict__") else None
    if other is None:
        pytest.skip("config is not a plain dataclass")
    with pytest.raises(NotImplementedError):
        potts_edge_energies(params, other, h_V, h_E, E_idx, mask, sequence)
