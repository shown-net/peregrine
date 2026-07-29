from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class RunConfig:
    config_id: str
    parameter_values: Mapping[str, int | float | str]
    gem5_args: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameter_values",
            MappingProxyType(dict(self.parameter_values)),
        )
        object.__setattr__(self, "gem5_args", tuple(self.gem5_args))
