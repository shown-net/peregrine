from __future__ import annotations

import torch
from torch import nn

from collections.abc import Iterable


class MultiHeadPeregrineModel(nn.Module):
    """The paper's shared two-layer trunk with one regression head per label."""

    def __init__(self, input_size: int, hidden_dims: Iterable[int], label_columns: Iterable[str]) -> None:
        super().__init__()
        first, second = tuple(hidden_dims)
        self.label_columns = tuple(label_columns)
        if not self.label_columns:
            raise ValueError("model must define at least one label")
        self.trunk = nn.Sequential(
            nn.Linear(input_size, first), nn.ReLU(),
            nn.Linear(first, second), nn.ReLU(),
        )
        self.heads = nn.ModuleDict({
            label: nn.Linear(second, 1) for label in self.label_columns
        })

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.trunk(features)
        return torch.cat([self.heads[label](encoded) for label in self.label_columns], dim=1)
