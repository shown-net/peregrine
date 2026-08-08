"""Small, schema-ordered surrogate networks for scalar and raw-stat tasks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn

POSITIVE_HEAD = "positive"
BOUNDED_HEAD = "bounded"
ZERO_INFLATED_HEAD = "zero_inflated"
CONSTANT_ZERO_HEAD = "constant_zero"
HEAD_KINDS = frozenset((POSITIVE_HEAD, BOUNDED_HEAD, ZERO_INFLATED_HEAD, CONSTANT_ZERO_HEAD))


class PositiveHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.output = nn.Linear(width, 1)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.output(encoded)


class BoundedHead(PositiveHead):
    pass


class ZeroInflatedHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.event, self.magnitude = nn.Linear(width, 1), nn.Linear(width, 1)

    def forward(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.event(encoded), self.magnitude(encoded)


class ConstantZeroHead(nn.Module):
    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return encoded.new_zeros((len(encoded), 1))


def _head(kind: str, width: int) -> nn.Module:
    if kind == CONSTANT_ZERO_HEAD:
        return ConstantZeroHead()
    return {POSITIVE_HEAD: PositiveHead, BOUNDED_HEAD: BoundedHead, ZERO_INFLATED_HEAD: ZeroInflatedHead}[kind](width)


class SurrogateNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_dims: Iterable[int], head_kinds: Mapping[str, str], *, dropout: float = 0.0) -> None:
        super().__init__()
        first, second = tuple(hidden_dims)
        if not head_kinds or set(head_kinds.values()) - HEAD_KINDS:
            raise ValueError("surrogate target heads are invalid")
        self.trunk = nn.Sequential(nn.Linear(input_size, first), nn.GELU(), nn.Dropout(dropout), nn.Linear(first, second), nn.GELU())
        self.heads = nn.ModuleDict({metric: _head(kind, second) for metric, kind in head_kinds.items()})

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        encoded = self.trunk(features)
        return {metric: head(encoded) for metric, head in self.heads.items()}
