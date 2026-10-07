# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX; mirrors the torch module tree of
#         mpnn.model.pottsmpnn.PottsMPNN / mpnn.model.mpnn.ProteinMPNN.
# Purpose: turn a flat torch state_dict (name -> numpy array) into the nested jax pytree the
#          ported modules read, and infer the architecture flags the checkpoint implies.

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp
import numpy as np


def nest(flat: Dict[str, np.ndarray], to_jax: bool = True) -> Dict[str, Any]:
    """``{'a.b.c': arr}`` -> ``{'a': {'b': {'c': arr}}}``; integer keys stay strings."""
    out: Dict[str, Any] = {}
    for key, arr in flat.items():
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = jnp.asarray(arr) if to_jax else arr
    return out


def _as_list(d: Dict[str, Any]) -> Any:
    """Collapse ``{'0': ..., '1': ...}`` levels into lists so layer stacks index naturally."""
    if isinstance(d, dict):
        if d and all(k.isdigit() for k in d):
            return [_as_list(d[k]) for k in sorted(d, key=int)]
        return {k: _as_list(v) for k, v in d.items()}
    return d


def build_params(flat: Dict[str, np.ndarray]) -> Dict[str, Any]:
    return _as_list(nest(flat))


def infer_etab_source(flat: Dict[str, np.ndarray], hidden_dim: int = 128) -> str:
    """``PottsMPNN.infer_etab_source``: the head's in-features are H ('edge') or 3H
    ('node_edge_node'). Handles both the bare-Linear and MLP key spellings."""
    for key in ("etab_out.weight", "etab_out.0.weight"):
        if key in flat:
            in_dim = int(flat[key].shape[1])
            break
    else:
        raise KeyError("checkpoint has no 'etab_out.weight' / 'etab_out.0.weight'; not a PottsMPNN?")
    mult, remainder = divmod(in_dim, hidden_dim)
    if remainder or mult not in (1, 3):
        raise ValueError(f"etab_out in-features {in_dim} is not 1x or 3x hidden_dim={hidden_dim}")
    return "node_edge_node" if mult == 3 else "edge"


def infer_field_source(flat: Dict[str, np.ndarray]) -> str:
    """'node' when the checkpoint carries a dedicated single-body head, else 'self_edge'."""
    return "node" if any(k.startswith("node_field.") for k in flat) else "self_edge"


def infer_shapes(flat: Dict[str, np.ndarray]) -> Dict[str, int]:
    return {
        "vocab_size": int(flat["W_s.weight"].shape[0]),
        "hidden_dim": int(flat["W_s.weight"].shape[1]),
        "num_encoder_layers": 1 + max(
            (int(k.split(".")[1]) for k in flat if k.startswith("encoder_layers.")), default=-1),
        "num_decoder_layers": 1 + max(
            (int(k.split(".")[1]) for k in flat if k.startswith("decoder_layers.")), default=-1),
        "potts_vocab_size": int(round(float(flat["etab_out.weight"].shape[0]) ** 0.5))
        if "etab_out.weight" in flat else int(flat["W_s.weight"].shape[0]),
    }
