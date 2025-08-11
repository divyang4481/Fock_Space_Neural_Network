"""FSNN multi-layer perceptron with dynamic width."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
from torch import nn

from .dynamic_linear import DynamicLinear


@dataclass
class GrowthConfig:
    step: int = 8
    patience: int = 2
    delta: float = 1e-3


@dataclass
class PruneConfig:
    pct: float = 0.2


class FSNNMLP(nn.Module):
    """Simple FSNN model with a single hidden dynamic layer."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        growth: GrowthConfig | None = None,
        prune: PruneConfig | None = None,
        complexity_lambda: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = DynamicLinear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.growth = growth or GrowthConfig()
        self.prune_cfg = prune or PruneConfig()
        self.complexity_lambda = complexity_lambda
        self.loss_history: List[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    # ------------------------------------------------------------------
    def complexity_penalty(self) -> torch.Tensor:
        return self.complexity_lambda * self.fc1.out_features

    def update_controller(self, val_loss: float) -> Tuple[int, int]:
        """Decide whether to grow or prune based on validation loss.

        Returns tuple ``(grown, pruned)`` with numbers of neurons added/removed.
        """
        grown = 0
        pruned = 0
        self.loss_history.append(val_loss)
        if len(self.loss_history) >= self.growth.patience:
            window = self.loss_history[-self.growth.patience :]
            if max(window) - min(window) < self.growth.delta:
                self.fc1.grow(self.growth.step)
                grown = self.growth.step
                self.loss_history.clear()
        # Prune after growth decision using L1 importance
        if self.prune_cfg.pct > 0 and self.fc1.out_features > 1:
            k = max(1, int(self.prune_cfg.pct * self.fc1.out_features))
            importance = self.fc1.l1_importance()
            idx = importance.argsort()[:k].tolist()
            stats = self.fc1.prune(idx)
            pruned = stats.removed
        return grown, pruned


__all__ = ["FSNNMLP", "GrowthConfig", "PruneConfig"]
