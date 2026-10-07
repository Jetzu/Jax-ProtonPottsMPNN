# Origin: cc (Claude Code, 2026-10-02) - ported from
#         ~/ProtonPottsMPNN/foundry/models/mpnn/src/mpnn/model/layers/message_passing.py and
#         position_wise_feed_forward.py (upstream read-only).
# Purpose: the ProteinMPNN graph gathers and the encoder/decoder message-passing layers, in JAX.
#          Dropout is identity here: the port is inference-only and the torch engine runs model.eval().

from __future__ import annotations

import jax.numpy as jnp

from ppjax.nn import Params, gelu, layer_norm, linear

ENC_SCALE = 30.0
DEC_SCALE = 30.0


def gather_edges(edge_features: jnp.ndarray, neighbor_idx: jnp.ndarray) -> jnp.ndarray:
    """[B,L,L,H] + [B,L,K] -> [B,L,K,H]; ``out[b,i,k] = edge_features[b,i,E_idx[b,i,k]]``."""
    return jnp.take_along_axis(edge_features, neighbor_idx[..., None], axis=2)


def gather_nodes(node_features: jnp.ndarray, neighbor_idx: jnp.ndarray) -> jnp.ndarray:
    """[B,L1,H] + [B,L2,K] -> [B,L2,K,H]; ``out[b,i,k] = node_features[b, E_idx[b,i,k]]``."""
    B, L2, K = neighbor_idx.shape
    flat = neighbor_idx.reshape(B, L2 * K)
    out = jnp.take_along_axis(node_features, flat[..., None], axis=1)
    return out.reshape(B, L2, K, node_features.shape[-1])


def cat_neighbors_nodes(h_nodes: jnp.ndarray, h_neighbors: jnp.ndarray,
                        E_idx: jnp.ndarray) -> jnp.ndarray:
    """``concat(h_neighbors, h_nodes[E_idx])`` on the last axis (edge features FIRST)."""
    return jnp.concatenate([h_neighbors, gather_nodes(h_nodes, E_idx)], axis=-1)


def position_wise_feed_forward(p: Params, h_V: jnp.ndarray) -> jnp.ndarray:
    return linear(p["W_out"], gelu(linear(p["W_in"], h_V)))


def enc_layer(p: Params, h_V, h_E, E_idx, mask_V=None, mask_E=None):
    """One ``EncLayer``: node update (scaled sum over neighbours) then edge update.

    The edge update deliberately re-reads the ORIGINAL ``h_E`` while concatenating the UPDATED
    ``h_V`` -- the same order as the torch layer.
    """
    h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
    h_V_expand = jnp.broadcast_to(h_V[:, :, None, :], h_EV.shape[:-1] + (h_V.shape[-1],))
    h_EV = jnp.concatenate([h_V_expand, h_EV], axis=-1)

    h_message = linear(p["W3"], gelu(linear(p["W2"], gelu(linear(p["W1"], h_EV)))))
    if mask_E is not None:
        h_message = mask_E[..., None] * h_message
    dh = jnp.sum(h_message, axis=-2) / ENC_SCALE
    h_V = layer_norm(p["norm1"], h_V + dh)
    h_V = layer_norm(p["norm2"], h_V + position_wise_feed_forward(p["dense"], h_V))
    if mask_V is not None:
        h_V = mask_V[..., None] * h_V

    h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
    h_V_expand = jnp.broadcast_to(h_V[:, :, None, :], h_EV.shape[:-1] + (h_V.shape[-1],))
    h_EV = jnp.concatenate([h_V_expand, h_EV], axis=-1)
    h_message = linear(p["W13"], gelu(linear(p["W12"], gelu(linear(p["W11"], h_EV)))))
    h_E = layer_norm(p["norm3"], h_E + h_message)
    return h_V, h_E


def dec_layer(p: Params, h_V, h_E, mask_V=None, mask_E=None):
    """One ``DecLayer``: node-only update. ``h_E`` must already carry the destination node (and
    sequence) features concatenated, exactly as the torch decoder prepares it."""
    h_V_expand = jnp.broadcast_to(h_V[..., None, :], h_E.shape[:-1] + (h_V.shape[-1],))
    h_EV = jnp.concatenate([h_V_expand, h_E], axis=-1)
    h_message = linear(p["W3"], gelu(linear(p["W2"], gelu(linear(p["W1"], h_EV)))))
    if mask_E is not None:
        h_message = mask_E[..., None] * h_message
    dh = jnp.sum(h_message, axis=-2) / DEC_SCALE
    h_V = layer_norm(p["norm1"], h_V + dh)
    h_V = layer_norm(p["norm2"], h_V + position_wise_feed_forward(p["dense"], h_V))
    if mask_V is not None:
        h_V = mask_V[..., None] * h_V
    return h_V
