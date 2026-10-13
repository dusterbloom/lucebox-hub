# models/qwen3_5/cache_convert.py
"""Conversions between the HF 5.x hybrid DynamicCache and the repo's (C1, beta, C2) tuples."""
from typing import Optional, Set, Tuple

import torch

from ..cache import CompactedPrefixCache

Triple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def linear_layer_set(layer_types) -> Set[int]:
    return {i for i, t in enumerate(layer_types) if t == "linear_attention"}


def dynamic_cache_to_triples(cache, layer_types) -> Tuple[Tuple[Triple, ...], Set[int]]:
    """HF DynamicCache(config=...) of a hybrid model -> ((C1, beta, C2) per layer, linear layer indices).

    Full-attention layers: C1/C2 are the cached K/V, beta = 0 (uncompacted prefix).
    Gated DeltaNet layers: C1 = conv state (B, conv_dim, kernel), C2 = recurrent state (B, heads, dk, dv),
    beta = empty placeholder. These are carried through unchanged by CompactedPrefixCache.
    """
    triples, linear = [], set()
    for i, (layer, kind) in enumerate(zip(cache.layers, layer_types)):
        if kind == "linear_attention":
            conv, rec = layer.conv_states[0], layer.recurrent_states[0]
            triples.append((conv, conv.new_zeros(conv.shape[0], 1, 0), rec))
            linear.add(i)
        else:
            k, v = layer.keys, layer.values
            triples.append((k, k.new_zeros(k.shape[:3]), v))
    return tuple(triples), linear


def dynamic_cache_to_kv_tuples(cache, layer_types):
    """Legacy ((K, V), ...) view used by the compaction code; DeltaNet layers get empty placeholders."""
    ref = next(layer for layer, kind in zip(cache.layers, layer_types) if kind != "linear_attention")
    out = []
    for layer, kind in zip(cache.layers, layer_types):
        if kind == "linear_attention":
            out.append((ref.keys.new_zeros(ref.keys.shape[0], ref.keys.shape[1], 0, ref.keys.shape[3]),
                        ref.values.new_zeros(ref.values.shape[0], ref.values.shape[1], 0, ref.values.shape[3])))
        else:
            out.append((layer.keys, layer.values))
    return tuple(out)


def inject_linear_state(compacted, cache, layer_types) -> Tuple[Triple, ...]:
    """Replace the placeholder entries the compaction code returns for DeltaNet layers with the real
    conv/recurrent state of `cache` (cloned)."""
    out = list(compacted)
    for i, kind in enumerate(layer_types):
        if kind == "linear_attention":
            conv, rec = cache.layers[i].conv_states[0], cache.layers[i].recurrent_states[0]
            out[i] = (conv.clone(), conv.new_zeros(conv.shape[0], 1, 0), rec.clone())
    return tuple(out)


def compacted_cache_from_triples(
    triples, linear_layer_indices, original_seq_len: Optional[int]
) -> CompactedPrefixCache:
    """Fresh CompactedPrefixCache (everything cloned, so decoding never mutates `triples`)."""
    return CompactedPrefixCache(
        triples,
        original_seq_len=original_seq_len,
        clone=True,
        linear_layer_indices=linear_layer_indices or None,
    )


@torch.no_grad()
def greedy_decode(model, cache, first_logits, n_new: int, keep_logits: bool = False, stop_ids=()):
    """Greedy decode `n_new` tokens from `cache`, starting from the logits of the last prefilled position.

    The decode loop is the same for the HF reference (DynamicCache) and the adapter (CompactedPrefixCache);
    only the cache and the class of the model differ. Returns (token_ids list, logits (n, V) bf16 on CPU or None).
    Token k is the argmax of step-k logits; step 0 is `first_logits`.
    """
    tokens, rows = [], []
    logits = first_logits
    for step in range(n_new):
        row = logits[0, -1]
        nxt = int(row.argmax())
        tokens.append(nxt)
        if keep_logits:
            rows.append(row.detach().to("cpu"))
        if nxt in stop_ids or step == n_new - 1:
            break
        ids = torch.tensor([[nxt]], device=first_logits.device)
        logits = model(input_ids=ids, past_key_values=cache, use_cache=True).logits
    return tokens, (torch.stack(rows) if keep_logits else None)
