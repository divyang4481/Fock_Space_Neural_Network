"""Dynamic linear layer supporting neuron growth and pruning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn


def _init_weight(in_features: int, out_features: int) -> nn.Parameter:
    weight = nn.Parameter(torch.empty(out_features, in_features))
    nn.init.kaiming_uniform_(weight, a=5 ** 0.5)
    return weight


def _init_bias(out_features: int) -> nn.Parameter:
    bias = nn.Parameter(torch.empty(out_features))
    fan_in = 1
    bound = 1 / fan_in ** 0.5
    nn.init.uniform_(bias, -bound, bound)
    return bias


@dataclass
class PruneStats:
    removed: int
    remaining: int


class DynamicLinear(nn.Module):
    """Linear layer that can grow and prune output neurons."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = _init_weight(in_features, out_features)
        self.bias = _init_bias(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

    # --- Growth & pruning -------------------------------------------------
    def grow(self, n: int) -> None:
        """Add ``n`` output neurons with random initialisation."""
        if n <= 0:
            return
        new_weight = _init_weight(self.in_features, n)
        new_bias = _init_bias(n)
        self.weight = nn.Parameter(torch.cat([self.weight, new_weight], dim=0))
        self.bias = nn.Parameter(torch.cat([self.bias, new_bias], dim=0))
        self.out_features += n

    def prune(self, indices: Iterable[int]) -> PruneStats:
        """Remove neurons by index.

        Args:
            indices: iterable of neuron indices to remove.
        Returns:
            PruneStats summarising the operation.
        """
        idx = sorted(set(i for i in indices if 0 <= i < self.out_features))
        if not idx:
            return PruneStats(removed=0, remaining=self.out_features)
        mask = torch.ones(self.out_features, dtype=torch.bool)
        mask[idx] = False
        self.weight = nn.Parameter(self.weight[mask])
        self.bias = nn.Parameter(self.bias[mask])
        self.out_features = self.weight.shape[0]
        return PruneStats(removed=len(idx), remaining=self.out_features)

    def l1_importance(self) -> torch.Tensor:
        """Compute L1 importance for each neuron."""
        return self.weight.abs().sum(dim=1)


__all__ = ["DynamicLinear", "PruneStats"]
