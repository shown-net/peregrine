from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


PARAMETER_SOURCES = ("host_fact", "simulator_default", "design")
DESIGN_SOURCE = "design"
GEM5_EMITS = ("always", "sampled", "never")
SUPPORTED_COMPONENT_MODELS = frozenset(
    (
        "rob_capacity_latency_bound",
        "load_queue_capacity_latency_bound",
        "store_queue_capacity_latency_bound",
        "issue_width_count_bound",
        "load_store_port_combined_bound",
        "width_bound",
        "icache_fill_slots_bound",
    )
)
ANALYSIS_CACHE_INPUTS = frozenset(
    (
        "line_bytes",
        "l1i_size",
        "l1d_size",
        "l2_size",
        "l1_associativity",
        "l2_associativity",
        "l1i_data_latency",
        "l1d_data_latency",
        "l2_data_latency",
        "dram_latency_cycles",
    )
)
CACHE_ANNOTATED_MODELS = frozenset(
    (
        "rob_capacity_latency_bound",
        "load_queue_capacity_latency_bound",
        "store_queue_capacity_latency_bound",
        "icache_fill_slots_bound",
    )
)


@dataclass(frozen=True)
class ParameterDef:
    name: str
    source: str
    param_type: str
    baseline: int | float | str
    values: tuple[int | float | str, ...]
    real: int | float | str | None
    gem5_flag: str | None
    gem5_unit: str
    gem5_emit: str | None

    @property
    def searchable(self) -> bool:
        return self.source == DESIGN_SOURCE


@dataclass(frozen=True)
class AnalysisInputDef:
    name: str
    parameter: str


@dataclass(frozen=True)
class ComponentDef:
    name: str
    model: str
    params: tuple[str, ...]


MechanismDef = ComponentDef


@dataclass(frozen=True)
class AnalysisModel:
    inputs: Mapping[str, str]
    components: tuple[ComponentDef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))

    @property
    def input_defs(self) -> tuple[AnalysisInputDef, ...]:
        return tuple(AnalysisInputDef(name, parameter) for name, parameter in self.inputs.items())

    @property
    def mechanisms(self) -> tuple[ComponentDef, ...]:
        return self.components


@dataclass(frozen=True)
class MicroarchitectureConfig:
    profile_id: str
    path: Path
    parameters: tuple[ParameterDef, ...]
    analysis: AnalysisModel
    ordered_constraints: tuple[tuple[str, ...], ...]
    sum_constraints: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sum_constraints", MappingProxyType(dict(self.sum_constraints)))

    @property
    def parameters_by_name(self) -> Mapping[str, ParameterDef]:
        return MappingProxyType({parameter.name: parameter for parameter in self.parameters})

    @property
    def mechanisms(self) -> tuple[ComponentDef, ...]:
        return self.analysis.components

    @property
    def components(self) -> tuple[ComponentDef, ...]:
        return self.analysis.components

    @property
    def design_parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters if parameter.source == DESIGN_SOURCE)

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
    def gem5_argument_parameters(self) -> tuple[ParameterDef, ...]:
        return tuple(
            parameter
            for parameter in self.parameters
            if parameter.gem5_flag
        )

    @property
    def sampled_gem5_parameters(self) -> tuple[ParameterDef, ...]:
        return tuple(
            parameter
            for parameter in self.parameters
            if parameter.gem5_flag and parameter.gem5_emit == "sampled"
        )

    @property
    def resource_parameter_names(self) -> frozenset[str]:
        return frozenset(name for component in self.components for name in component.params)

    def parameter_values(self, overrides: Mapping[str, int | float | str] | None = None) -> Mapping[str, int | float | str]:
        overrides = overrides or {}
        canonical_names = set(self.parameters_by_name)
        if set(overrides) == canonical_names:
            values = dict(overrides)
        else:
            unknown = sorted(set(overrides) - set(self.design_parameter_names))
            if unknown:
                raise ValueError(f"microarchitecture overrides are not design parameters: {unknown}")
            values = dict(self.baseline_values)
            values.update(overrides)
        self.validate_values(values)
        return MappingProxyType(values)

    def validate_values(self, values: Mapping[str, int | float | str]) -> None:
        if set(values) != set(self.parameters_by_name):
            raise ValueError("microarchitecture values do not cover the canonical parameter set")
        for parameter in self.parameters:
            value = values[parameter.name]
            if parameter.source == DESIGN_SOURCE and value not in parameter.values:
                raise ValueError(f"microarchitecture value is outside its domain: {parameter.name}={value}")
            if parameter.param_type == "int" and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"microarchitecture integer must be positive: {parameter.name}={value}")
            if parameter.param_type == "float" and (not isinstance(value, int | float) or float(value) <= 0.0):
                raise ValueError(f"microarchitecture float must be positive: {parameter.name}={value}")
        for chain in self.ordered_constraints:
            numeric = [int(values[name]) for name in chain]
            if numeric != sorted(numeric):
                raise ValueError(f"microarchitecture ordering constraint failed: {' <= '.join(chain)}")
        for target, sources in self.sum_constraints.items():
            if int(values[target]) < sum(int(values[source]) for source in sources):
                raise ValueError(f"microarchitecture capacity constraint failed: {target}")

    def gem5_args(self, overrides: Mapping[str, int | float | str] | None = None) -> tuple[str, ...]:
        values = self.parameter_values(overrides)
        return tuple(
            f"{parameter.gem5_flag}={_render(parameter, values[parameter.name])}"
            for parameter in self.gem5_argument_parameters
        )

    def parse_gem5_args(self, args: tuple[str, ...]) -> Mapping[str, int | float | str]:
        by_flag: dict[str, str] = {}
        for arg in args:
            if not arg.startswith("--") or "=" not in arg:
                continue
            flag, raw = arg.split("=", 1)
            if flag in by_flag:
                raise ValueError(f"duplicate microarchitecture gem5 argument: {flag}")
            by_flag[flag] = raw
        values = dict(self.baseline_values)
        for parameter in self.gem5_argument_parameters:
            if parameter.gem5_flag is None or parameter.gem5_flag not in by_flag:
                raise ValueError(f"collection sample is missing canonical gem5 override: {parameter.name}")
            values[parameter.name] = _parse(parameter, by_flag[parameter.gem5_flag])
        return self.parameter_values(values)

    def analytical_values(self, overrides: Mapping[str, int | float | str] | None = None) -> Mapping[str, int | float | str]:
        values = self.parameter_values(overrides)
        analysis_values = {
            input_name: values[parameter_name]
            for input_name, parameter_name in self.analysis.inputs.items()
        }
        analysis_values.update(
            {
                parameter_name: values[parameter_name]
                for parameter_name in self.resource_parameter_names
            }
        )
        return MappingProxyType(analysis_values)

    def config_id(self, overrides: Mapping[str, int | float | str] | None = None) -> str:
        payload = json.dumps(dict(self.parameter_values(overrides)), sort_keys=True, separators=(",", ":"))
        return f"config_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _render(parameter: ParameterDef, value: int | float | str) -> str:
    if parameter.gem5_unit == "KiB":
        return f"{int(value) // 1024}MiB" if int(value) % 1024 == 0 else f"{int(value)}KiB"
    if parameter.gem5_unit == "GiB":
        return f"{int(value)}GiB"
    if parameter.gem5_unit == "GHz":
        return f"{float(value):g}GHz"
    return str(value)


def _parse(parameter: ParameterDef, raw: str) -> int | float | str:
    if parameter.param_type == "categorical":
        value = str(raw)
        if value not in parameter.values:
            raise ValueError(f"microarchitecture value is outside its domain: {parameter.name}={value}")
        return value
    normalized = raw.strip()
    if parameter.gem5_unit == "KiB":
        if normalized.endswith("KiB"):
            return int(normalized[:-3])
        if normalized.endswith("MiB"):
            return int(normalized[:-3]) * 1024
        raise ValueError(f"gem5 argument has invalid KiB value: {parameter.gem5_flag}={raw}")
    if parameter.gem5_unit == "GiB":
        if not normalized.endswith("GiB"):
            raise ValueError(f"gem5 argument has invalid GiB value: {parameter.gem5_flag}={raw}")
        return int(normalized[:-3])
    if parameter.gem5_unit == "GHz":
        if not normalized.endswith("GHz"):
            raise ValueError(f"gem5 argument has invalid GHz value: {parameter.gem5_flag}={raw}")
        return float(normalized[:-3]) if parameter.param_type == "float" else int(normalized[:-3])
    return int(normalized)


def load_microarchitecture_config(path: str | Path) -> MicroarchitectureConfig:
    resolved = Path(path).resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"microarchitecture config must be a mapping: {resolved}")
    legacy_keys = sorted(set(payload) & {"parameter_groups", "analysis_bindings"})
    if legacy_keys:
        raise ValueError(f"microarchitecture config uses retired keys: {legacy_keys}")
    if not isinstance(payload.get("parameters"), dict):
        raise ValueError(f"microarchitecture config must define parameters: {resolved}")
    parameter_items = tuple(
        _parameter(str(name), node)
        for name, node in payload["parameters"].items()
    )
    names = {parameter.name for parameter in parameter_items}
    if len(names) != len(parameter_items):
        raise ValueError("microarchitecture config has duplicate parameter names")
    analysis = _analysis_model(payload.get("analysis"))
    missing_component_params = sorted({name for component in analysis.components for name in component.params} - names)
    if missing_component_params:
        raise ValueError(f"microarchitecture components reference unknown parameters: {missing_component_params}")
    missing_inputs = sorted(set(analysis.inputs.values()) - names)
    if missing_inputs:
        raise ValueError(f"microarchitecture analysis inputs reference unknown parameters: {missing_inputs}")
    cache_models = {component.model for component in analysis.components} & CACHE_ANNOTATED_MODELS
    if cache_models:
        missing_cache_inputs = sorted(ANALYSIS_CACHE_INPUTS - set(analysis.inputs))
        if missing_cache_inputs:
            raise ValueError(f"microarchitecture analysis inputs missing cache annotations: {missing_cache_inputs}")
    constraints = payload.get("constraints") or {}
    ordered = tuple(tuple(str(name) for name in chain) for chain in constraints.get("ordered", ()))
    sums = {str(target): tuple(str(name) for name in sources) for target, sources in (constraints.get("at_least_sum") or {}).items()}
    constraint_names = {name for chain in ordered for name in chain} | set(sums) | {name for sources in sums.values() for name in sources}
    unknown_constraints = sorted(constraint_names - names)
    if unknown_constraints:
        raise ValueError(f"microarchitecture constraints reference unknown parameters: {unknown_constraints}")
    config = MicroarchitectureConfig(str(payload["profile_id"]), resolved, parameter_items, analysis, ordered, sums)
    config.validate_values(config.baseline_values)
    return config


def _analysis_model(raw: Any) -> AnalysisModel:
    if not isinstance(raw, dict):
        raise ValueError("microarchitecture config must define analysis")
    input_nodes = raw.get("inputs")
    if not isinstance(input_nodes, dict):
        raise ValueError("analysis.inputs must be a mapping")
    inputs = {str(name): str(parameter) for name, parameter in input_nodes.items()}
    component_nodes = raw.get("components")
    if not isinstance(component_nodes, list) or not component_nodes:
        raise ValueError("analysis.components must be a non-empty list")
    components = []
    seen = set()
    for node in component_nodes:
        if not isinstance(node, dict):
            raise ValueError("analysis component must be a mapping")
        retired = sorted(set(node) & {"formula", "gem5", "latency_inputs"})
        if retired:
            raise ValueError(f"analysis component uses retired keys: {retired}")
        name = str(node["name"])
        if name in seen:
            raise ValueError(f"duplicate analysis component: {name}")
        seen.add(name)
        model = str(node.get("model") or "").strip()
        if model not in SUPPORTED_COMPONENT_MODELS:
            raise ValueError(f"unknown analysis component model: {name}={model}")
        params = node.get("params")
        if not isinstance(params, list) or not params:
            raise ValueError(f"analysis component params must be a non-empty list: {name}")
        components.append(ComponentDef(name=name, model=model, params=tuple(str(item) for item in params)))
    return AnalysisModel(inputs, tuple(components))


def _parameter(name: str, node: Any) -> ParameterDef:
    if not isinstance(node, dict):
        raise ValueError(f"microarchitecture parameter must be a mapping: {name}")
    retired = sorted(set(node) & {"simulation", "analysis", "analysis_key"})
    if retired:
        raise ValueError(f"microarchitecture parameter uses retired keys: {name}: {retired}")
    source = str(node.get("source") or "")
    if source not in PARAMETER_SOURCES:
        raise ValueError(f"unknown microarchitecture parameter source: {name}={source}")
    param_type = str(node["type"])
    baseline = _typed(node["baseline"], param_type)
    real = None if "real" not in node else _typed(node["real"], param_type)
    domain = node.get("domain") or {}
    if not isinstance(domain, dict):
        raise ValueError(f"microarchitecture parameter domain must be a mapping: {name}")
    offsets = domain.get("offsets")
    candidates = domain.get("candidates")
    if offsets is not None and candidates is not None:
        raise ValueError(f"microarchitecture parameter has two search definitions: {name}")
    has_search_domain = offsets is not None or candidates is not None
    if source == DESIGN_SOURCE and not has_search_domain:
        raise ValueError(f"design microarchitecture parameter must define a domain: {name}")
    if source != DESIGN_SOURCE and has_search_domain:
        raise ValueError(f"non-design microarchitecture parameter must not define a domain: {name}")
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
    if source == DESIGN_SOURCE and len(values) < 2:
        raise ValueError(f"design microarchitecture parameter domain must contain alternatives: {name}")
    if param_type == "int" and any(int(value) <= 0 for value in values):
        raise ValueError(f"microarchitecture integer domain must be positive: {name}")
    if param_type == "float" and any(float(value) <= 0.0 for value in values):
        raise ValueError(f"microarchitecture float domain must be positive: {name}")
    gem5 = node.get("gem5")
    gem5_flag = None
    gem5_unit = ""
    gem5_emit = None
    if gem5 is not None:
        if not isinstance(gem5, dict):
            raise ValueError(f"microarchitecture gem5 mapping must be a mapping: {name}")
        gem5_flag = str(gem5["flag"])
        gem5_unit = str(gem5.get("unit", ""))
        gem5_emit = str(gem5.get("emit", ""))
        if gem5_emit not in GEM5_EMITS:
            raise ValueError(f"unknown microarchitecture gem5 emit policy: {name}={gem5_emit}")
        if source == DESIGN_SOURCE and gem5_emit != "sampled":
            raise ValueError(f"design microarchitecture parameter must use gem5 emit=sampled: {name}")
        if source != DESIGN_SOURCE and gem5_emit == "sampled":
            raise ValueError(f"non-design microarchitecture parameter cannot use gem5 emit=sampled: {name}")
    elif source == DESIGN_SOURCE:
        raise ValueError(f"design microarchitecture parameter must define gem5 mapping: {name}")
    return ParameterDef(name, source, param_type, baseline, values, real, gem5_flag, gem5_unit, gem5_emit)


def _typed(value: Any, param_type: str) -> int | float | str:
    if param_type == "categorical":
        return str(value)
    if param_type == "int":
        return int(value)
    if param_type == "float":
        return float(value)
    raise ValueError(f"unsupported microarchitecture parameter type: {param_type}")
