# models/qwen3_5/modeling_qwen3_5.py
"""Attention Matching adapter for the Qwen3.5 / Qwen3.8 hybrid (3 Gated DeltaNet : 1 gated full attention).

Written against transformers 5.19 (`transformers/models/qwen3_5/modeling_qwen3_5.py`). It mirrors what
`models/qwen3/modeling_qwen3.py:223-260` does for Qwen3, and touches only the full-attention path:

  * `Qwen3_5CompactableAttention` is `Qwen3_5Attention` plus the beta bias added to the first t mask columns
    (t = compacted prefix length of that layer). It is added after the K/V cache update and before the
    attention kernel, so the sigmoid output gate (applied after the kernel) is untouched.
  * `Qwen3_5CompactableTextModel` is `Qwen3_5TextModel` with a branch for `CompactedPrefixCache`: RoPE positions
    continue from the ORIGINAL sequence length (`rope_base`), the full-attention mask is built here over the
    longest layer, and the Gated DeltaNet layers run unchanged on the conv/recurrent state the cache carries.
    With any other cache it defers to the unmodified parent forward.

No parameters are added, so an already loaded HF model is converted in place with `adapt_model` (class swap) and
can be restored with `restore_model`; this also lets one set of weights serve both the reference and the adapter.
"""
from typing import Callable, Optional

import torch
from transformers.masking_utils import create_recurrent_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5TextModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from ..cache import CompactedPrefixCache
from .masks import add_beta_to_mask, causal_additive_mask


class Qwen3_5CompactableAttention(Qwen3_5Attention):
    """Full-attention layer with the AM bias. Identical to the parent unless the cache is a CompactedPrefixCache."""

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if isinstance(past_key_values, CompactedPrefixCache):
            # AM bias: beta enters the additive mask of the compacted prefix columns.
            attention_mask = add_beta_to_mask(
                attention_mask,
                past_key_values.beta_for_layer(self.layer_idx),
                self.num_key_value_groups,
                key_states.shape[2],
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)  # output gate: untouched
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3_5CompactableTextModel(Qwen3_5TextModel):
    """Text decoder with a CompactedPrefixCache branch (see module docstring)."""

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,  # accepted for the repo's call sites (chunked_prefill); positions come from the cache
        max_layer=None,  # query capture only: stop after this layer (the logits of such a call are not meaningful)
        **kwargs,
    ):
        if not isinstance(past_key_values, CompactedPrefixCache):
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, q_len = inputs_embeds.shape[:2]
        device = inputs_embeds.device

        # Longest full-attention layer (prefix + tokens appended so far). RoPE continues from the original length.
        past_len = past_key_values.get_seq_length()
        if position_ids is None:
            pos = torch.arange(q_len, device=device)[None, :] + past_len
            pad_counts = past_key_values.pad_counts()
            if pad_counts is not None:
                pos = torch.clamp(pos - pad_counts.to(device)[:, None] + past_key_values.rope_base(), min=0)
            else:
                pos = (pos + past_key_values.rope_base()).expand(batch_size, -1)
            position_ids = pos
        rope_ids = position_ids[None].expand(3, -1, -1)  # text-only: t/h/w streams share the position
        position_embeddings = self.rotary_emb(inputs_embeds, rope_ids)

        full_mask = causal_additive_mask(past_len, q_len, inputs_embeds.dtype, device, batch_size, attention_mask)
        linear_mask = create_recurrent_attention_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        hidden_states = inputs_embeds
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if max_layer is not None and i > max_layer:
                break
            is_full = self.config.layer_types[i] == "full_attention"
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=full_mask if is_full else linear_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


def text_model(model):
    """The Qwen3_5TextModel inside a ConditionalGeneration (`.model.language_model`) or CausalLM (`.model`) wrapper."""
    inner = model.model
    return inner.language_model if hasattr(inner, "language_model") else inner


_TEXT_CONFIG_KEYS = (
    "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "hidden_size",
    "layer_types", "sliding_window", "rms_norm_eps", "vocab_size",
)


def adapt_model(model):
    """Swap the classes of an already loaded HF Qwen3.5 model in place. Returns the model."""
    lm = text_model(model)
    if not isinstance(lm, Qwen3_5TextModel):
        raise TypeError(f"expected a Qwen3_5TextModel, got {type(lm)}")
    lm.__class__ = Qwen3_5CompactableTextModel
    for layer in lm.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.__class__ = Qwen3_5CompactableAttention
    # The repo reads these from `model.config`; on the composite config they live in `text_config`.
    text_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    for key in _TEXT_CONFIG_KEYS:
        if not hasattr(model.config, key) and hasattr(text_cfg, key):
            setattr(model.config, key, getattr(text_cfg, key))
    return model


def restore_model(model):
    """Undo `adapt_model` (the unmodified HF classes come back)."""
    lm = text_model(model)
    lm.__class__ = Qwen3_5TextModel
    for layer in lm.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.__class__ = Qwen3_5Attention
    return model
