from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


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


def resolve_parameter_values(
    *,
    gem5_args: tuple[str, ...],
    parameter_names: tuple[str, ...],
    parameter_defs: Mapping[str, Any],
) -> dict[str, int | float | str]:
    by_flag: dict[str, str] = {}
    for arg in gem5_args:
        flag = _flag_name(arg)
        if flag in by_flag:
            raise ValueError(f"duplicate CPU gem5 arg: {flag}")
        if "=" in arg:
            by_flag[flag] = arg.split("=", 1)[1]
    missing = [
        parameter_defs[name].gem5_arg
        for name in parameter_names
        if parameter_defs[name].gem5_arg not in by_flag
        and parameter_defs[name].param_type != "categorical"
    ]
    if missing:
        raise ValueError(f"collection plan is missing design-space gem5 overrides: {missing}")
    return {
        name: (
            _parse_gem5_value(
                parameter_defs[name],
                by_flag[parameter_defs[name].gem5_arg],
            )
            if parameter_defs[name].gem5_arg in by_flag
            else parameter_defs[name].default
        )
        for name in parameter_names
    }


def _flag_name(arg: str) -> str:
    if not arg.startswith("--"):
        raise ValueError(f"gem5 arg must be --flag or --flag=value: {arg}")
    return arg.split("=", 1)[0]


def gem5_flag_for(parameter: Any, value: Any) -> str:
    if not parameter.gem5_arg:
        raise ValueError(f"design parameter has no gem5 arg: {parameter.name}")
    if parameter.param_type == "categorical":
        rendered = str(value)
        if rendered not in parameter.values:
            raise ValueError(
                f"design-space categorical value is outside its domain: "
                f"{parameter.name}={rendered}"
            )
    elif parameter.gem5_unit == "KiB":
        rendered = _render_cache_size_kb(int(value))
    elif parameter.param_type == "int":
        rendered = str(int(value))
    else:
        rendered = str(float(value))
    return f"{parameter.gem5_arg}={rendered}"


def _parse_gem5_value(parameter: Any, raw: str) -> int | float | str:
    if parameter.param_type == "categorical":
        value = raw.strip()
        if value not in parameter.values:
            raise ValueError(
                f"design-space categorical value is outside its domain: "
                f"{parameter.gem5_arg}={raw}"
            )
        return value
    if parameter.gem5_unit == "KiB":
        normalized = raw.strip()
        if normalized.endswith("KiB"):
            value = int(normalized[:-3])
        elif normalized.endswith("MiB"):
            value = int(normalized[:-3]) * 1024
        else:
            raise ValueError(
                f"design-space cache baseline must use KiB or MiB: {parameter.gem5_arg}={raw}"
            )
    else:
        value = float(raw) if parameter.param_type == "float" else int(raw)
    if value not in parameter.values:
        raise ValueError(
            f"design-space numeric value is outside its domain: "
            f"{parameter.gem5_arg}={raw}"
        )
    return value


def _render_cache_size_kb(value: int) -> str:
    return f"{value // 1024}MiB" if value % 1024 == 0 else f"{value}KiB"
