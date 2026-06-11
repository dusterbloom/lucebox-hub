"""Exact block-mass and cross-layer label oracle for captured Qwen3.5 Q/K."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OracleResult:
    label_mass: torch.Tensor
    positive: torch.Tensor
    vote_count: torch.Tensor
    cold_mass: torch.Tensor


def layer_block_attention_mass(
    q_post_rope: torch.Tensor,
    k_post_rope: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    block_size: int = 64,
    sink_tokens: int = 64,
    recent_tokens: int = 8192,
    boundary_position: int,
) -> torch.Tensor:
    """Return exact cold-block mass as [future_tokens, candidate_blocks].

    Q is [future, q_heads, head_dim], K is [keys, kv_heads, head_dim]. The
    softmax denominator includes every causally visible key, while output bins
    cover sealed cold blocks only.
    """

    if q_post_rope.ndim != 3 or k_post_rope.ndim != 3:
        raise ValueError("Q and K captures must be rank-3 tensors")
    if q_post_rope.shape[-1] != k_post_rope.shape[-1]:
        raise ValueError("Q and K head dimensions do not match")
    if q_post_rope.shape[0] != query_positions.numel():
        raise ValueError("query position count does not match Q")
    if k_post_rope.shape[0] != key_positions.numel():
        raise ValueError("key position count does not match K")
    if q_post_rope.shape[1] % k_post_rope.shape[1] != 0:
        raise ValueError("query heads must be divisible by KV heads")
    if block_size <= 0 or sink_tokens < 0 or recent_tokens < 0:
        raise ValueError("oracle token geometry is invalid")
    if boundary_position < 0:
        raise ValueError("boundary position must be non-negative")
    if not torch.isfinite(q_post_rope).all() or not torch.isfinite(k_post_rope).all():
        raise ValueError("Q/K captures contain non-finite values")

    cold_end = max(sink_tokens, boundary_position - recent_tokens)
    first_block = (sink_tokens + block_size - 1) // block_size
    block_count = cold_end // block_size
    candidate_blocks = max(0, block_count - first_block)
    output = torch.zeros(
        (q_post_rope.shape[0], candidate_blocks),
        dtype=torch.float32,
        device=q_post_rope.device,
    )
    if candidate_blocks == 0:
        return output

    group_size = q_post_rope.shape[1] // k_post_rope.shape[1]
    scale = q_post_rope.shape[-1] ** -0.5
    for row, query_position in enumerate(query_positions.tolist()):
        visible = key_positions <= query_position
        if not bool(visible.any()):
            raise ValueError("a future query has no causally visible key")
        visible_keys = k_post_rope[visible].float().permute(1, 0, 2)
        visible_positions = key_positions[visible]
        grouped_query = q_post_rope[row].float().reshape(
            k_post_rope.shape[1], group_size, q_post_rope.shape[-1]
        )
        logits = torch.einsum(
            "hgd,hkd->hgk", grouped_query, visible_keys
        )
        probability = torch.softmax(logits * scale, dim=-1).mean(dim=(0, 1))

        cold = (visible_positions >= first_block * block_size) & (
            visible_positions < block_count * block_size
        )
        if bool(cold.any()):
            bins = visible_positions[cold] // block_size - first_block
            output[row].scatter_add_(0, bins.long(), probability[cold])
    return output


def cross_layer_oracle(
    layer_mass: torch.Tensor,
    *,
    top_p: float = 0.6,
    minimum_cold_mass: float = 0.02,
    minimum_layer_votes: int = 3,
    mean_weight: float = 0.8,
) -> OracleResult:
    """Combine [layers, future, blocks] mass using LSA-style voting."""

    if layer_mass.ndim != 3:
        raise ValueError("layer mass must have shape [layers, future, blocks]")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if minimum_cold_mass < 0 or minimum_layer_votes <= 0:
        raise ValueError("oracle thresholds are invalid")
    if not 0 <= mean_weight <= 1:
        raise ValueError("mean_weight must be in [0, 1]")
    if not torch.isfinite(layer_mass).all() or bool((layer_mass < 0).any()):
        raise ValueError("layer mass must be finite and non-negative")

    layers, future, blocks = layer_mass.shape
    votes = torch.zeros((future, blocks), dtype=torch.int16, device=layer_mass.device)
    cold_mass = layer_mass.sum(dim=-1)
    for layer in range(layers):
        for token in range(future):
            total = cold_mass[layer, token]
            if float(total) < minimum_cold_mass or blocks == 0:
                continue
            normalized = layer_mass[layer, token] / total
            values, indices = torch.sort(normalized, descending=True)
            count = int(
                torch.searchsorted(
                    torch.cumsum(values, dim=0),
                    torch.tensor(top_p, device=values.device),
                    right=False,
                )
            ) + 1
            votes[token, indices[: min(count, blocks)]] += 1

    positive_by_token = votes >= minimum_layer_votes
    positive = positive_by_token.any(dim=0)
    mean_mass = layer_mass.mean(dim=(0, 1)) if layers and future else torch.zeros(blocks)
    max_mass = (
        layer_mass.amax(dim=(0, 1)) if layers and future else torch.zeros(blocks)
    )
    soft_mass = mean_weight * mean_mass + (1 - mean_weight) * max_mass
    label_mass = torch.where(positive, soft_mass.clamp(0, 1), torch.zeros_like(soft_mass))
    return OracleResult(
        label_mass=label_mass,
        positive=positive,
        vote_count=votes.amax(dim=0) if future else torch.zeros(blocks, dtype=torch.int16),
        cold_mass=cold_mass,
    )
