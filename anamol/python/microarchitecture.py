from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ParameterDef:
    name: str
    param_type: str
    gem5_arg: str | None
    gem5_unit: str
    baseline: int | float | str
    values: tuple[int | float | str, ...]
    searchable: bool
    analysis_key: str | None


@dataclass(frozen=True)
class MechanismDef:
    name: str
    params: tuple[str, ...]
    latency_dependent: tuple[str, ...]


@dataclass(frozen=True)
class MicroarchitectureConfig:
    profile_id: str
    path: Path
    parameters: tuple[ParameterDef, ...]
    mechanisms: tuple[MechanismDef, ...]
    ordered_constraints: tuple[tuple[str, ...], ...]
    sum_constraints: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sum_constraints", MappingProxyType(dict(self.sum_constraints)))

    @property
    def parameters_by_name(self) -> Mapping[str, ParameterDef]:
        return MappingProxyType({parameter.name: parameter for parameter in self.parameters})

    @property
    def design_parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters if parameter.searchable)

    @property
    def numeric_design_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name in self.design_parameter_names if self.parameters_by_name[name].param_type != "categorical")

    @property
    def categorical_design_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name in self.design_parameter_names if self.parameters_by_name[name].param_type == "categorical")

    @property
    def baseline_values(self) -> Mapping[str, int | float | str]:
        return MappingProxyType({parameter.name: parameter.baseline for parameter in self.parameters})

    @property
    def resource_parameter_names(self) -> frozenset[str]:
        return frozenset(name for mechanism in self.mechanisms for name in mechanism.params)

    def parameter_values(self, overrides: Mapping[str, int | float | str] | None = None) -> Mapping[str, int | float | str]:
        overrides = overrides or {}
        canonical_names = set(self.parameters_by_name)
        if set(overrides) == canonical_names:
            values = dict(overrides)
        else:
            unknown = sorted(set(overrides) - set(self.design_parameter_names))
            if unknown:
                raise ValueError(f"microarchitecture overrides are not searchable parameters: {unknown}")
            values = dict(self.baseline_values)
            values.update(overrides)
        self.validate_values(values)
        return MappingProxyType(values)

    def validate_values(self, values: Mapping[str, int | float | str]) -> None:
        if set(values) != set(self.parameters_by_name):
            raise ValueError("microarchitecture values do not cover the canonical parameter set")
        for parameter in self.parameters:
            value = values[parameter.name]
            if parameter.searchable and value not in parameter.values:
                raise ValueError(f"microarchitecture value is outside its domain: {parameter.name}={value}")
            if parameter.param_type == "int" and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"microarchitecture integer must be positive: {parameter.name}={value}")
        for chain in self.ordered_constraints:
            numeric = [int(values[name]) for name in chain]
            if numeric != sorted(numeric):
                raise ValueError(f"microarchitecture ordering constraint failed: {' <= '.join(chain)}")
        for target, sources in self.sum_constraints.items():
            if int(values[target]) < sum(int(values[source]) for source in sources):
                raise ValueError(f"microarchitecture capacity constraint failed: {target}")

    def gem5_args(self, overrides: Mapping[str, int | float | str] | None = None) -> tuple[str, ...]:
        values = self.parameter_values(overrides)
        args = []
        for parameter in self.parameters:
            if parameter.gem5_arg:
                args.append(f"{parameter.gem5_arg}={_render(parameter, values[parameter.name])}")
        return tuple(args)

    def parse_gem5_args(self, args: tuple[str, ...]) -> Mapping[str, int | float | str]:
        by_flag: dict[str, str] = {}
        for arg in args:
            if not arg.startswith("--") or "=" not in arg:
                continue
            flag, raw = arg.split("=", 1)
            if flag in by_flag:
                raise ValueError(f"duplicate microarchitecture gem5 argument: {flag}")
            by_flag[flag] = raw
        overrides: dict[str, int | float | str] = {}
        for parameter in self.parameters:
            if parameter.searchable:
                if not parameter.gem5_arg or parameter.gem5_arg not in by_flag:
                    raise ValueError(f"collection sample is missing canonical gem5 override: {parameter.name}")
                overrides[parameter.name] = _parse(parameter, by_flag[parameter.gem5_arg])
        return self.parameter_values(overrides)

    def analytical_values(self, overrides: Mapping[str, int | float | str] | None = None) -> Mapping[str, int | float | str]:
        values = self.parameter_values(overrides)
        return MappingProxyType({
            parameter.analysis_key or parameter.name: values[parameter.name]
            for parameter in self.parameters
            if parameter.analysis_key is not None or parameter.name in self.resource_parameter_names
            or parameter.name in {"branch_predictor", "l1i_data_latency", "l1d_data_latency", "l2_data_latency", "hnf_data_latency"}
        })

    def config_id(self, overrides: Mapping[str, int | float | str] | None = None) -> str:
        payload = json.dumps(dict(self.parameter_values(overrides)), sort_keys=True, separators=(",", ":"))
        return f"config_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _render(parameter: ParameterDef, value: int | float | str) -> str:
    if parameter.gem5_unit == "KiB":
        return f"{int(value) // 1024}MiB" if int(value) % 1024 == 0 else f"{int(value)}KiB"
    if parameter.gem5_unit == "GiB":
        return f"{int(value)}GiB"
    if parameter.gem5_unit == "GHz":
        return f"{int(value)}GHz"
    return str(value)


def _parse(parameter: ParameterDef, raw: str) -> int | float | str:
    if parameter.param_type == "categorical":
        return str(raw)
    normalized = raw.strip()
    if parameter.gem5_unit == "KiB":
        if normalized.endswith("KiB"):
            return int(normalized[:-3])
        if normalized.endswith("MiB"):
            return int(normalized[:-3]) * 1024
        raise ValueError(f"gem5 argument has invalid KiB value: {parameter.gem5_arg}={raw}")
    if parameter.gem5_unit == "GiB":
        if not normalized.endswith("GiB"):
            raise ValueError(f"gem5 argument has invalid GiB value: {parameter.gem5_arg}={raw}")
        return int(normalized[:-3])
    if parameter.gem5_unit == "GHz":
        if not normalized.endswith("GHz"):
            raise ValueError(f"gem5 argument has invalid GHz value: {parameter.gem5_arg}={raw}")
        return int(normalized[:-3])
    return int(normalized)


def load_microarchitecture_config(path: str | Path) -> MicroarchitectureConfig:
    resolved = Path(path).resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or not isinstance(payload.get("parameters"), dict):
        raise ValueError(f"microarchitecture config must define parameters: {resolved}")
    parameters = tuple(_parameter(name, node) for name, node in payload["parameters"].items())
    names = {parameter.name for parameter in parameters}
    if len(names) != len(parameters):
        raise ValueError("microarchitecture config has duplicate parameter names")
    mechanisms = tuple(
        MechanismDef(str(node["name"]), tuple(str(name) for name in node["params"]), tuple(str(item) for item in node.get("latency_dependent", ())))
        for node in payload.get("mechanisms", ())
    )
    missing = sorted({name for mechanism in mechanisms for name in mechanism.params} - names)
    if missing:
        raise ValueError(f"microarchitecture mechanisms reference unknown parameters: {missing}")
    constraints = payload.get("constraints") or {}
    ordered = tuple(tuple(str(name) for name in chain) for chain in constraints.get("ordered", ()))
    sums = {str(target): tuple(str(name) for name in sources) for target, sources in (constraints.get("at_least_sum") or {}).items()}
    constraint_names = {name for chain in ordered for name in chain} | set(sums) | {name for sources in sums.values() for name in sources}
    unknown_constraints = sorted(constraint_names - names)
    if unknown_constraints:
        raise ValueError(f"microarchitecture constraints reference unknown parameters: {unknown_constraints}")
    config = MicroarchitectureConfig(str(payload["profile_id"]), resolved, parameters, mechanisms, ordered, sums)
    config.validate_values(config.baseline_values)
    return config


def _parameter(name: str, node: Any) -> ParameterDef:
    if not isinstance(node, dict) or not isinstance(node.get("simulation"), dict):
        raise ValueError(f"microarchitecture parameter must define simulation: {name}")
    sim = node["simulation"]
    param_type = str(node["type"])
    baseline = _typed(sim["baseline"], param_type)
    offsets = sim.get("search_offsets")
    candidates = sim.get("candidates")
    if offsets is not None and candidates is not None:
        raise ValueError(f"microarchitecture parameter has two search definitions: {name}")
    if offsets is not None:
        if param_type == "categorical":
            raise ValueError(f"categorical microarchitecture parameter requires candidates: {name}")
        values = tuple(_typed(int(baseline) + int(offset), param_type) for offset in offsets)
    elif candidates is not None:
        values = tuple(_typed(value, param_type) for value in candidates)
    else:
        values = (baseline,)
    if not values or len(values) != len(set(values)) or baseline not in values:
        raise ValueError(f"microarchitecture search domain must be unique and include baseline: {name}")
    if param_type == "int" and any(int(value) <= 0 for value in values):
        raise ValueError(f"microarchitecture integer domain must be positive: {name}")
    return ParameterDef(name, param_type, None if sim.get("gem5_arg") is None else str(sim["gem5_arg"]), str(sim.get("gem5_unit", "")), baseline, values, len(values) > 1, None if sim.get("analysis") is None and sim.get("analysis_key") is None else str(sim.get("analysis_key") or name))


def _typed(value: Any, param_type: str) -> int | float | str:
    if param_type == "categorical":
        return str(value)
    if param_type == "int":
        return int(value)
    raise ValueError(f"unsupported microarchitecture parameter type: {param_type}")
