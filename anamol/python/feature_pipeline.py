from __future__ import annotations

from pathlib import Path

import numpy as np

from .distribution_features import distribution_feature_columns
from .microarchitecture import ComponentDef


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


def analyze_full_roi_windows(
    *,
    trace_path: str | Path,
    configs: tuple[Mapping[str, int | float | str], ...],
    full_roi_window_size: int,
    analysis_window_size: int,
    window_count: int,
    mechanisms: tuple[ComponentDef, ...],
    config_threads: int = 1,
) -> np.ndarray:
    if not configs:
        raise ValueError("Anamol configs must not be empty")
    if full_roi_window_size <= 0:
        raise ValueError("full-ROI window size must be positive")
    if analysis_window_size <= 0:
        raise ValueError("analysis window size must be positive")
    if window_count < 2:
        raise ValueError("full-ROI analysis needs one warm-up and one labeled window")
    if config_threads < 1:
        raise ValueError("Anamol config thread budget must be positive")
    trace = Path(trace_path).resolve()
    if trace.suffixes[-2:] != [".pb", ".zst"]:
        raise ValueError(f"Anamol requires a protobuf trace: {trace}")
    bindings = list(mechanism_bindings(mechanisms))
    module = _analysis_module()
    expected_columns = len(analytical_feature_columns(mechanisms))
    if int(module.feature_count_for_bindings(bindings)) != expected_columns:
        raise ValueError("Anamol extension feature count differs from configured columns")
    values = module.analyze_trace_windows(
        str(trace),
        int(full_roi_window_size),
        int(analysis_window_size),
        int(window_count),
        [dict(config) for config in configs],
        bindings,
        int(config_threads),
    )
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != len(configs) or values.shape[2] != expected_columns:
        raise ValueError("Anamol window feature tensor dimensions differ from the configured contract")
    values = np.transpose(values, (1, 0, 2))
    if not np.isfinite(values).all():
        raise ValueError("Anamol window feature tensor contains non-finite features")
    return values


def _analysis_module():
    try:
        from anamol import _analysis
    except ImportError as error:
        raise RuntimeError(
            "Anamol Python extension is not built. Run "
            "`make -C anamol python-extension` from the Peregrine repository."
        ) from error
    return _analysis
