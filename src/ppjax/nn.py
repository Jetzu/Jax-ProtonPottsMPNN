# Origin: cc (Claude Code, 2026-10-02) - ported from torch.nn primitives used by ProteinMPNN.
# Purpose: the handful of JAX primitives the port needs, matching torch's conventions exactly
#          (Linear stores [out, in]; LayerNorm eps=1e-5 over the last axis; GELU is the EXACT
#          erf form, not the tanh approximation).

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp

Params = Dict[str, Any]

LAYERNORM_EPS = 1e-5


def linear(p: Params, x: jnp.ndarray) -> jnp.ndarray:
    """``torch.nn.Linear``: ``x @ W.T + b`` with ``W`` stored as ``[out_features, in_features]``."""
    y = x @ p["weight"].T
    if "bias" in p:
        y = y + p["bias"]
    return y


def layer_norm(p: Params, x: jnp.ndarray, eps: float = LAYERNORM_EPS) -> jnp.ndarray:
    """``torch.nn.LayerNorm(H)``: normalise the last axis with the BIASED variance, then affine."""
    mu = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mu), axis=-1, keepdims=True)
    return (x - mu) * jax.lax.rsqrt(var + eps) * p["weight"] + p["bias"]


def gelu(x: jnp.ndarray) -> jnp.ndarray:
    """``torch.nn.GELU()`` default = exact erf form."""
    return jax.nn.gelu(x, approximate=False)


def embedding(p: Params, idx: jnp.ndarray) -> jnp.ndarray:
    """``torch.nn.Embedding``: row lookup into ``[num_embeddings, dim]``."""
    return jnp.take(p["weight"], idx, axis=0)


def one_hot(idx: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    return jax.nn.one_hot(idx, num_classes, dtype=jnp.float32)
