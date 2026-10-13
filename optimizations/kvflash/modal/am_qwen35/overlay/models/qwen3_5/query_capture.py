# models/qwen3_5/query_capture.py
"""Query capture helpers for hybrid Qwen3.5 models: only the full-attention layers have queries."""
from typing import List, Optional

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, apply_rotary_pos_emb


def get_decoder_layers(model):
    """The decoder layer list of a plain causal LM (`model.model.layers`) or a Qwen3.5 conditional-generation
    wrapper (`model.model.language_model.layers`)."""
    inner = model.model
    return inner.language_model.layers if hasattr(inner, "language_model") else inner.layers


def attention_layer_indices(model, upto: Optional[int] = None) -> List[int]:
    """Indices of the layers that own a `self_attn` (the 16 full-attention layers of Qwen3.8-27B).
    `upto` keeps only indices < upto, like the `layers_to_process` bound of the repo's query hooks."""
    layers = get_decoder_layers(model)
    n = len(layers) if upto is None else min(upto, len(layers))
    return [i for i in range(n) if hasattr(layers[i], "self_attn")]


def project_queries(module, hidden_states, position_embeddings) -> Optional[torch.Tensor]:
    """Post-norm, post-RoPE queries (B, S, H, D) of a Qwen3.5 full-attention module, or None for other modules.

    q_proj of Qwen3.5 emits (query, output gate) interleaved per head; the gate is dropped here.
    """
    if not isinstance(module, Qwen3_5Attention):
        return None
    input_shape = hidden_states.shape[:-1]
    q, _gate = torch.chunk(module.q_proj(hidden_states).view(*input_shape, -1, module.head_dim * 2), 2, dim=-1)
    q = module.q_norm(q.reshape(*input_shape, -1, module.head_dim)).transpose(1, 2)  # (B, H, S, D)
    if position_embeddings is not None:
        cos, sin = position_embeddings
        q, _ = apply_rotary_pos_emb(q, q, cos, sin)
    return q.transpose(1, 2)
