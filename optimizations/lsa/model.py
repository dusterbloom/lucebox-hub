"""Compact Qwen3.5 query encoder and training objective."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CompactQwen35Encoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 5120,
        rank: int = 256,
        kv_heads: int = 4,
        head_dim: int = 256,
        score_temperature: float = 0.1,
        decision_threshold: float = 0.02,
        logit_scale: float = 12.0,
    ) -> None:
        super().__init__()
        if min(hidden_size, rank, kv_heads, head_dim) <= 0:
            raise ValueError("encoder dimensions must be positive")
        if score_temperature <= 0 or logit_scale <= 0:
            raise ValueError("score temperature and logit scale must be positive")
        self.hidden_size = hidden_size
        self.rank = rank
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.score_temperature = score_temperature
        self.decision_threshold = decision_threshold
        self.logit_scale = logit_scale
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, kv_heads * head_dim, bias=False)

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        query = self.up(F.silu(self.down(hidden)))
        query = query.unflatten(-1, (self.kv_heads, self.head_dim))
        return F.normalize(query, dim=-1)

    def forward(self, hidden: torch.Tensor, block_keys: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 1 or hidden.shape[0] != self.hidden_size:
            raise ValueError("hidden must have shape [hidden_size]")
        if block_keys.ndim != 3 or block_keys.shape[1:] != (
            self.kv_heads,
            self.head_dim,
        ):
            raise ValueError("block_keys must have shape [blocks, kv_heads, head_dim]")
        query = self.encode(hidden)
        keys = F.normalize(block_keys, dim=-1)
        per_head = torch.einsum("hd,bhd->bh", query, keys)
        pooled = self.score_temperature * (
            torch.logsumexp(per_head / self.score_temperature, dim=-1)
            - math.log(self.kv_heads)
        )
        return (pooled - self.decision_threshold) * self.logit_scale

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def focal_mass_loss(
    logits: torch.Tensor,
    target_mass: torch.Tensor,
    *,
    gamma: float = 2.0,
    positive_weight: float = 3.0,
) -> torch.Tensor:
    target = target_mass.clamp(0, 1)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    pt = target * probability + (1 - target) * (1 - probability)
    class_weight = 1 + target * (positive_weight - 1)
    return (class_weight * (1 - pt).pow(gamma) * bce).mean()
