from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np

from .distribution_features import distribution_feature_columns
from .microarchitecture import ComponentDef


@dataclass(frozen=True)
class FeatureBatch:
    values: np.ndarray


def mechanism_bindings(mechanisms: tuple[ComponentDef, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "name": mechanism.name,
            "model": mechanism.model,
            "params": mechanism.params,
        }
        for mechanism in mechanisms
    )


def analytical_feature_columns(mechanisms: tuple[ComponentDef, ...]) -> tuple[str, ...]:
    return tuple(
        column
        for mechanism in mechanisms
        for column in distribution_feature_columns(f"dynamic_{mechanism.name}")
    )


def trace_section_counts(
    trace_path: str | Path,
) -> tuple[int, ...]:
    trace = Path(trace_path).resolve()
    if trace.suffixes[-2:] != [".pb", ".zst"]:
        raise ValueError(f"Anamol requires a protobuf trace: {trace}")
    return (int(_analysis_module().trace_instruction_count(str(trace))),)


def iter_anamol_feature_batches(
    *,
    trace_path: str | Path,
    configs: tuple[Mapping[str, int | float | str], ...],
    window_size: int,
    mechanisms: tuple[ComponentDef, ...],
) -> Iterator[FeatureBatch]:
    if not configs:
        raise ValueError("Anamol configs must not be empty")
    trace = Path(trace_path).resolve()
    if trace.suffixes[-2:] != [".pb", ".zst"]:
        raise ValueError(f"Anamol requires a protobuf trace: {trace}")
    bindings = list(mechanism_bindings(mechanisms))
    module = _analysis_module()
    expected_columns = len(analytical_feature_columns(mechanisms))
    if int(module.feature_count_for_bindings(bindings)) != expected_columns:
        raise ValueError("Anamol extension feature count differs from configured columns")
    values = module.analyze_trace(
        str(trace),
        int(window_size),
        [dict(config) for config in configs],
        bindings,
    )
    values = np.asarray(values, dtype=np.float64)
    expected_shape = (len(configs), expected_columns)
    if values.shape != expected_shape:
        raise ValueError("Anamol feature matrix dimensions differ from the configured contract")
    if not np.isfinite(values).all():
        raise ValueError("Anamol feature matrix contains non-finite features")
    yield FeatureBatch(values=values)


def _analysis_module():
    try:
        from anamol import _analysis
    except ImportError as error:
        raise RuntimeError(
            "Anamol Python extension is not built. Run "
            "`make -C /data00/xuhaoen/peregrine/anamol python-extension`."
        ) from error
    return _analysis
