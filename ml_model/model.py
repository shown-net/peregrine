"""Canonical neural building blocks for the CPU performance surrogate."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn


POSITIVE_HEAD = "positive"
BOUNDED_HEAD = "bounded"
ZERO_INFLATED_HEAD = "zero_inflated"
HEAD_KINDS = frozenset((POSITIVE_HEAD, BOUNDED_HEAD, ZERO_INFLATED_HEAD))


class PositiveHead(nn.Module):
    """One latent regression output decoded in the positive physical domain."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.output = nn.Linear(width, 1)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.output(encoded)


class BoundedHead(nn.Module):
    """One latent regression output decoded to the unit interval."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.output = nn.Linear(width, 1)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.output(encoded)


class ZeroInflatedHead(nn.Module):
    """Event probability and conditional positive magnitude for sparse metrics."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.event = nn.Linear(width, 1)
        self.magnitude = nn.Linear(width, 1)

    def forward(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.event(encoded), self.magnitude(encoded)


class SurrogateNetwork(nn.Module):
    """Shared MLP encoder with the fixed, canonical metric heads."""

    def __init__(
        self,
        input_size: int,
        hidden_dims: Iterable[int],
        head_kinds: Mapping[str, str],
    ) -> None:
        super().__init__()
        first, second = tuple(hidden_dims)
        if not head_kinds:
            raise ValueError("surrogate must define at least one target")
        unknown = sorted(set(head_kinds.values()) - HEAD_KINDS)
        if unknown:
            raise ValueError(f"unknown surrogate head kinds: {unknown}")
        self.trunk = nn.Sequential(
            nn.Linear(input_size, first), nn.ReLU(), nn.Linear(first, second), nn.ReLU()
        )
        self.head_kinds = dict(head_kinds)
        self.heads = nn.ModuleDict({
            metric: _head(kind, second) for metric, kind in self.head_kinds.items()
        })

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        encoded = self.trunk(features)
        return {metric: head(encoded) for metric, head in self.heads.items()}


def _head(kind: str, width: int) -> nn.Module:
    if kind == POSITIVE_HEAD:
        return PositiveHead(width)
    if kind == BOUNDED_HEAD:
        return BoundedHead(width)
    if kind == ZERO_INFLATED_HEAD:
        return ZeroInflatedHead(width)
    raise ValueError(f"unknown surrogate head kind: {kind}")
