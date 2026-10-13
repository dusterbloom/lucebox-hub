# models/qwen3_5/masks.py
"""Pure mask helpers for the Qwen3.5 Attention-Matching adapter (torch only, no transformers import).

Everything here is a function of tensors and ints so it can be unit tested without a model.
"""
from typing import Optional

import torch
from torch import Tensor


def repeat_kv_for_beta(beta: Tensor, n_rep: int) -> Tensor:
    """(B, KV, t) -> (B, KV * n_rep, t); same head order as HF repeat_kv."""
    batch, num_kv_heads, slen = beta.shape
    if n_rep == 1:
        return beta
    beta = beta[:, :, None, :].expand(batch, num_kv_heads, n_rep, slen)
    return beta.reshape(batch, num_kv_heads * n_rep, slen)


def causal_additive_mask(
    prefix_len: int,
    q_len: int,
    dtype: torch.dtype,
    device,
    batch_size: int = 1,
    attention_mask_2d: Optional[Tensor] = None,
) -> Tensor:
    """Additive causal mask (B, 1, q_len, prefix_len + q_len) for a cache that already holds `prefix_len` tokens.

    Every query sees the whole prefix and is causal inside the new block. 0 where allowed, finfo(dtype).min
    elsewhere (the same convention as the HF eager mask). `attention_mask_2d` (B, prefix_len + q_len) or
    (B, q_len) carries padding zeros; a (B, q_len) mask is taken to describe only the new tokens.
    """
    kv_len = prefix_len + q_len
    i = torch.arange(q_len, device=device)[:, None]
    j = torch.arange(kv_len, device=device)[None, :]
    allowed = (j < prefix_len) | ((j - prefix_len) <= i)  # (q, kv)
    allowed = allowed[None, None].expand(batch_size, 1, q_len, kv_len)
    if attention_mask_2d is not None:
        pad = attention_mask_2d.to(device=device).bool()
        if pad.shape[-1] == q_len and prefix_len > 0:
            pad = torch.cat([pad.new_ones(pad.shape[0], prefix_len), pad], dim=-1)
        if pad.shape[-1] != kv_len:
            raise ValueError(f"attention_mask width {pad.shape[-1]} is neither q_len {q_len} nor kv_len {kv_len}")
        allowed = allowed & pad[:, None, None, :]
    mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)


def add_beta_to_mask(
    mask: Tensor,
    beta_kv: Tensor,
    n_rep: int,
    kv_len: int,
) -> Tensor:
    """Add the AM bias to one layer's mask.

    mask:    additive float mask (B, 1 or A, q, >= kv_len); the last `kv_len` columns belong to this layer
             (nonuniform caches build the mask at the longest layer).
    beta_kv: (B, KV, t) bias of the compacted prefix, applied to the first t columns of this layer.
    Returns a new (B, A, q, kv_len) tensor; the input is not modified.
    """
    beta_heads = repeat_kv_for_beta(beta_kv, n_rep)  # (B, A, t)
    batch, num_heads, prefix_len = beta_heads.shape
    out = mask[..., -kv_len:]
    if out.shape[1] == 1 and num_heads > 1:
        out = out.expand(batch, num_heads, out.shape[2], kv_len)
    out = out.clone()
    out[:, :, :, :prefix_len] += beta_heads.unsqueeze(2).to(out.dtype)
    return out
