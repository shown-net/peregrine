from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import re
import shlex
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .design_space import PeregrineConfig, reciprocal_feature_columns, workload_context_feature_columns
from .feature_pipeline import analyze_full_roi_windows, analytical_feature_columns
from .gem5_stats import read_label_values
from .run_config import RunConfig


IDENTITY_COLUMNS = ("workload_id", "region_id", "config_id")
FULL_ROI_IDENTITY_COLUMNS = ("workload_id", "window_index", "config_id")
TRACE_FILE = "peregrine.trace.pb.zst"
SAMPLING_MANIFEST = "sampling_manifest.json"
PARQUET_BATCH_ROWS = 8192
FINAL_CONFIG_ID = re.compile(r"config_[0-9a-f]{16}$")
FULL_ROI_INTERVAL_INSTS = 100_000


@dataclass(frozen=True)
class RawSample:
    workload_id: str
    region_id: str
    config: RunConfig
    labels: tuple[float, ...]
    trace_path: Path


@dataclass(frozen=True)
class FullRoiConfigSample:
    config_id: str
    config: RunConfig
    stats_path: Path
    labels: np.ndarray


@dataclass(frozen=True)
class ConfigStatsSample:
    """One canonical full-ROI gem5 config artifact shared by model builders."""

    config_id: str
    config: RunConfig | None
    stats_path: Path
    window_count: int


@dataclass(frozen=True)
class ConfigStatsWorkload:
    """The raw config-stats evidence for one workload."""

    workload_id: str
    reference_trace_path: Path
    samples: tuple[ConfigStatsSample, ...]


@dataclass(frozen=True)
class DatasetTableContext:
    analytical_columns: tuple[str, ...]
    workload_context_columns: tuple[str, ...]
    categorical_values: dict[str, tuple[str, ...]]
    reciprocal_parameters: tuple[str, ...]


def build_full_roi_dataset_shards(
    *,
    config: PeregrineConfig,
    raw_root: str | Path,
    output_dir: str | Path,
    workload_ids: tuple[str, ...] | None = None,
    workers: int = 24,
    full_roi_window_size: int = FULL_ROI_INTERVAL_INSTS,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("dataset workers must be positive")
    raw = Path(raw_root).resolve()
    workloads_root = raw / "workloads"
    if not workloads_root.is_dir():
        raise FileNotFoundError(f"missing full-ROI workloads root: {workloads_root}")
    canonical_workloads = tuple(workload_ids) if workload_ids is not None else tuple(
        sorted(path.name for path in workloads_root.iterdir() if path.is_dir())
    )
    if not canonical_workloads or len(set(canonical_workloads)) != len(canonical_workloads):
        raise ValueError("dataset workload selection is empty or duplicated")
    out = Path(output_dir)
    base_root = out / "base"
    components_root = out / "components"
    aggregated_root = out / "aggregated"
    for root in (base_root, components_root, aggregated_root):
        root.mkdir(parents=True, exist_ok=True)
        for partial in root.rglob("*.partial"):
            partial.unlink()
    expected_shards = {f"{_shard_stem(workload_id)}.parquet" for workload_id in canonical_workloads}
    for root in (base_root, aggregated_root):
        for stale in root.glob("*.parquet"):
            if stale.name not in expected_shards:
                stale.unlink()
    for mechanism in config.microarchitecture.mechanisms:
        component_root = components_root / mechanism.name
        component_root.mkdir(exist_ok=True)
        for stale in component_root.glob("*.parquet"):
            if stale.name not in expected_shards:
                stale.unlink()
    reports: list[dict[str, int | str]] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(canonical_workloads))) as executor:
        futures = [
            executor.submit(
                _build_full_roi_workload_artifacts,
                config=config,
                raw_root=raw,
                workload_id=workload_id,
                output_root=out,
                full_roi_window_size=full_roi_window_size,
            )
            for workload_id in canonical_workloads
        ]
        for future in futures:
            reports.append(future.result())
    return {
        "dataset": str(out), "base": str(base_root), "components": str(components_root),
        "aggregated": str(aggregated_root), "shards": sorted(expected_shards), "workloads": reports,
    }


def rebuild_full_roi_component_shards(
    *, config: PeregrineConfig, output_dir: str | Path, component_names: tuple[str, ...],
    workload_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Re-aggregate selected component shards after their independent regeneration.

    Component generators own trace analysis.  This operation deliberately only
    combines existing artifacts, so changing an aggregation rule never starts a
    trace scan.
    """
    if not component_names or len(component_names) != len(set(component_names)):
        raise ValueError("component rebuild selection is empty or duplicated")
    known = {mechanism.name for mechanism in config.microarchitecture.mechanisms}
    if not set(component_names) <= known:
        raise ValueError("component rebuild selection is not in the configured microarchitecture")
    root = Path(output_dir)
    base_root = root / "base"
    selected = tuple(workload_ids) if workload_ids is not None else tuple(
        sorted(path.stem for path in base_root.glob("*.parquet"))
    )
    if not selected:
        raise ValueError("component rebuild has no base workload shards")
    reports = [
        _aggregate_full_roi_workload_artifacts(config=config, workload_id=workload_id, output_root=root)
        for workload_id in selected
    ]
    return {"aggregated": str(root / "aggregated"), "components": component_names, "workloads": reports}


def regenerate_full_roi_component_shards(
    *, config: PeregrineConfig, raw_root: str | Path, output_dir: str | Path,
    component_names: tuple[str, ...], workload_ids: tuple[str, ...] | None = None,
    full_roi_window_size: int = FULL_ROI_INTERVAL_INSTS,
) -> dict[str, Any]:
    """Re-run only selected causal components, then cheaply aggregate them."""
    root = Path(output_dir)
    selected = tuple(workload_ids) if workload_ids is not None else tuple(
        sorted(path.stem for path in (root / "base").glob("*.parquet"))
    )
    mechanisms = tuple(
        mechanism for mechanism in config.microarchitecture.mechanisms if mechanism.name in component_names
    )
    if not mechanisms or len(mechanisms) != len(component_names):
        raise ValueError("component regeneration selection is not in the configured microarchitecture")
    for workload_id in selected:
        _regenerate_full_roi_workload_components(
            config=config, raw_root=Path(raw_root), output_root=root, workload_id=workload_id,
            mechanisms=mechanisms, full_roi_window_size=full_roi_window_size,
        )
    return rebuild_full_roi_component_shards(
        config=config, output_dir=root, component_names=component_names, workload_ids=selected,
    )


def load_config_stats_workload(
    *, config: PeregrineConfig, raw_root: str | Path, workload_id: str,
    parse_configs: bool = True,
) -> ConfigStatsWorkload:
    """Load the single canonical raw config-stats contract for a workload.

    Both the surrogate dataset builder and the fixed-N2 calibrator use this
    loader.  It owns raw layout validation so consumers do not rescan and
    reinterpret the same gem5 artifacts independently.
    """
    root = Path(raw_root)
    configs_root = root / "workloads" / workload_id / "configs"
    if not configs_root.is_dir():
        raise FileNotFoundError(f"missing config-stats artifacts: {configs_root}")
    reference_id = config.microarchitecture.config_id({})
    reference_trace_path = configs_root / reference_id / TRACE_FILE
    if not reference_trace_path.is_file() or reference_trace_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing config-stats reference trace: {reference_trace_path}")
    samples: list[ConfigStatsSample] = []
    expected_windows: int | None = None
    for config_root in sorted(
        (path for path in configs_root.iterdir() if path.is_dir()),
        key=lambda path: (path.name != reference_id, path.name),
    ):
        stats_path = config_root / "stats.h5"
        if not stats_path.is_file():
            raise FileNotFoundError(f"missing config-stats statistics: {stats_path}")
        labels = read_label_values(stats_path, config.labels)
        window_count = int(labels.shape[0])
        if window_count <= 0:
            raise ValueError(f"config-stats artifact has no complete windows: {stats_path}")
        if expected_windows is None:
            expected_windows = window_count
        elif window_count != expected_windows:
            raise ValueError(
                f"config-stats window counts differ: expected {expected_windows}, "
                f"got {window_count} at {stats_path}"
            )
        run_config = None
        if parse_configs:
            run_config = (
                config.run_config_from_args(reference_id, config.microarchitecture.gem5_args())
                if config_root.name == reference_id
                else _run_config_from_log(config_root, config)
            )
            if run_config.config_id != config_root.name:
                raise ValueError(f"config-stats identity differs from artifact directory: {config_root}")
        samples.append(ConfigStatsSample(config_root.name, run_config, stats_path, window_count))
    if not samples:
        raise ValueError(f"config-stats workload has no configuration artifacts: {workload_id}")
    return ConfigStatsWorkload(workload_id, reference_trace_path, tuple(samples))


def _load_manifest_samples(
    raw_root: Path,
    *,
    manifest_path: Path,
    config: PeregrineConfig,
) -> dict[str, tuple[RawSample, ...]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing CPU collection sampling manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"sampling manifest must be a mapping: {manifest_path}")
    if str(payload.get("trace_file")) != TRACE_FILE:
        raise ValueError(f"sampling manifest trace file differs from dataset contract: {manifest_path}")
    workloads = payload.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError(f"sampling manifest contains no workloads: {manifest_path}")
    by_workload: dict[str, list[RawSample]] = {}
    for workload_node in workloads:
        if not isinstance(workload_node, dict):
            raise ValueError("sampling manifest workload must be a mapping")
        workload_id = str(workload_node["workload_id"])
        regions = workload_node.get("regions")
        if not isinstance(regions, list) or not regions:
            raise ValueError(f"sampling manifest contains no regions: {workload_id}")
        for region_node in regions:
            if not isinstance(region_node, dict):
                raise ValueError("sampling manifest region must be a mapping")
            region_id = str(region_node["region_id"])
            samples = region_node.get("samples")
            if not isinstance(samples, list) or not samples:
                raise ValueError(f"sampling manifest contains no samples: {workload_id}/{region_id}")
            roles = [str(sample.get("role")) for sample in samples if isinstance(sample, dict)]
            if roles.count("trace_producer") != 1:
                raise ValueError(f"sampling manifest region requires one trace producer: {workload_id}/{region_id}")
            for sample_node in samples:
                if not isinstance(sample_node, dict):
                    raise ValueError("sampling manifest sample must be a mapping")
                trace_path = _manifest_path(raw_root, sample_node["trace_path"])
                stats_path = _manifest_path(raw_root, sample_node["stats_path"])
                parameter_values = sample_node.get("parameter_values")
                if not isinstance(parameter_values, dict):
                    raise ValueError(f"sampling manifest sample has no parameter values: {workload_id}/{region_id}")
                values = config.microarchitecture.parameter_values(
                    {str(name): value for name, value in parameter_values.items()}
                )
                run_config = RunConfig(
                    str(sample_node["config_id"]),
                    values,
                    config.microarchitecture.gem5_args(values),
                )
                try:
                    labels = read_label_values(stats_path, config.labels)
                except (OSError, KeyError, TypeError, ValueError):
                    by_workload.setdefault(workload_id, []).append(
                        RawSample(workload_id, region_id, run_config, (), trace_path)
                    )
                    continue
                if labels.shape != (1, len(config.labels.labels)) or not np.isfinite(labels).all():
                    by_workload.setdefault(workload_id, []).append(
                        RawSample(workload_id, region_id, run_config, (), trace_path)
                    )
                    continue
                by_workload.setdefault(workload_id, []).append(
                    RawSample(
                        workload_id,
                        region_id,
                        run_config,
                        tuple(float(value) for value in labels[0]),
                        trace_path,
                    )
                )
    return {
        workload_id: tuple(sorted(samples, key=lambda item: (item.region_id, item.config.config_id)))
        for workload_id, samples in by_workload.items()
    }


def _manifest_path(raw_root: Path, raw_path: object) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"sampling manifest path must be relative to raw root: {path}")
    return raw_root / path


def _manifest_valid_samples(
    samples: tuple[RawSample, ...],
    *,
    config: PeregrineConfig,
) -> tuple[tuple[RawSample, ...], int]:
    valid: list[RawSample] = []
    skipped = 0
    for sample in samples:
        if not sample.trace_path.is_file() or sample.trace_path.stat().st_size == 0:
            skipped += 1
            continue
        if len(sample.labels) != len(config.labels.labels) or not np.isfinite(np.asarray(sample.labels, dtype=np.float64)).all():
            skipped += 1
            continue
        valid.append(sample)
    return tuple(valid), skipped


def _discover_raw_samples(
    workload_root: Path,
    *,
    workload_id: str,
    config: PeregrineConfig,
) -> tuple[tuple[RawSample, ...], int]:
    regions = workload_root / "regions"
    if not regions.is_dir():
        raise FileNotFoundError(f"missing planned region directory: {regions}")
    region_ids = tuple(sorted(path.name for path in regions.iterdir() if path.is_dir()))
    if not region_ids:
        raise ValueError(f"collection plan contains no region samples: {workload_root}")
    samples: list[RawSample] = []
    skipped = 0
    for region_id in region_ids:
        region_root = regions / region_id
        if not region_root.is_dir():
            continue
        trace_path = region_root / "trace" / TRACE_FILE
        if not trace_path.is_file() or trace_path.stat().st_size == 0:
            skipped += 1
            continue
        trace_stats = region_root / "trace" / "baseline_stats.h5"
        if trace_stats.is_file():
            try:
                baseline = config.run_config_from_args(
                    config.microarchitecture.config_id({}),
                    config.microarchitecture.gem5_args(),
                )
                labels = read_label_values(trace_stats, config.labels)
            except (OSError, TypeError, ValueError):
                skipped += 1
            else:
                if labels.shape != (1, len(config.labels.labels)) or not np.isfinite(labels).all():
                    skipped += 1
                else:
                    samples.append(
                        RawSample(
                            workload_id,
                            region_id,
                            baseline,
                            tuple(float(value) for value in labels[0]),
                            trace_path,
                        )
                    )
        config_parent = region_root / "configs"
        if not config_parent.is_dir():
            continue
        for config_root in sorted(path for path in config_parent.iterdir() if path.is_dir()):
            if not FINAL_CONFIG_ID.fullmatch(config_root.name):
                continue
            if (config_root / TRACE_FILE).exists():
                skipped += 1
                continue
            run_config = _run_config_from_log(config_root, config)
            try:
                labels = read_label_values(config_root / "stats.h5", config.labels)
            except (OSError, TypeError, ValueError):
                skipped += 1
                continue
            if labels.shape != (1, len(config.labels.labels)) or not np.isfinite(labels).all():
                skipped += 1
                continue
            samples.append(
                RawSample(
                    workload_id,
                    region_id,
                    run_config,
                    tuple(float(value) for value in labels[0]),
                    trace_path,
                )
            )

    return tuple(sorted(samples, key=lambda item: (item.region_id, item.config.config_id))), skipped


def _run_config_from_log(sample_root: Path, config: PeregrineConfig) -> RunConfig:
    log_path = sample_root / "run.log"
    try:
        command = next(
            line.partition(":")[2].strip()
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.startswith("command line:")
        )
    except (OSError, StopIteration) as error:
        raise ValueError(f"missing gem5 command line for raw sample: {sample_root}") from error
    parameter_flags = {
        parameter.gem5_flag
        for parameter in config.microarchitecture.gem5_argument_parameters
    }
    gem5_args = tuple(
        item
        for item in shlex.split(command)
        if item.startswith("--") and item.partition("=")[0] in parameter_flags
    )
    return config.run_config_from_args(sample_root.name, gem5_args)


def _build_full_roi_workload_dataset(
    *,
    config: PeregrineConfig,
    raw_root: Path,
    workload_id: str,
    destination: Path,
    full_roi_window_size: int,
) -> dict[str, int | str]:
    raw = load_config_stats_workload(config=config, raw_root=raw_root, workload_id=workload_id)
    trace_path = raw.reference_trace_path
    samples = _full_roi_config_samples(config=config, raw_root=raw_root, workload_id=workload_id)
    if not samples:
        raise ValueError(f"full-ROI workload contains no config samples: {workload_id}")
    window_count = samples[0].labels.shape[0]
    if window_count < 2:
        raise ValueError(f"full-ROI workload needs one warm-up and one labeled window: {workload_id}")
    expected_rows = tuple(
        (window_index, sample.config_id)
        for window_index in range(1, window_count)
        for sample in samples
    )
    if destination.exists():
        try:
            _validate_full_roi_dataset_shard(
                destination,
                workload_id=workload_id,
                config=config,
                expected_rows=expected_rows,
            )
        except (OSError, TypeError, ValueError):
            destination.unlink()
        else:
            return {"workload_id": workload_id, "rows": len(expected_rows), "configs": len(samples), "windows": window_count - 1}
    rows = _write_full_roi_workload_dataset(
        config=config,
        workload_id=workload_id,
        trace_path=trace_path,
        samples=samples,
        destination=destination,
        full_roi_window_size=full_roi_window_size,
    )
    _validate_full_roi_dataset_shard(
        destination,
        workload_id=workload_id,
        config=config,
        expected_rows=expected_rows,
    )
    return {"workload_id": workload_id, "rows": rows, "configs": len(samples), "windows": window_count - 1}


def _build_full_roi_workload_artifacts(
    *, config: PeregrineConfig, raw_root: Path, workload_id: str, output_root: Path,
    full_roi_window_size: int,
) -> dict[str, int | str]:
    """Create reusable base/component artifacts from one causal trace scan."""
    aggregate = output_root / "aggregated" / f"{_shard_stem(workload_id)}.parquet"
    report = _build_full_roi_workload_dataset(
        config=config, raw_root=raw_root, workload_id=workload_id, destination=aggregate,
        full_roi_window_size=full_roi_window_size,
    )
    source = pq.read_table(aggregate)
    base_columns = _full_roi_base_columns(config)
    _write_table(output_root / "base" / aggregate.name, source.select(base_columns))
    for mechanism in config.microarchitecture.mechanisms:
        columns = [*FULL_ROI_IDENTITY_COLUMNS, *analytical_feature_columns((mechanism,))]
        _write_table(output_root / "components" / mechanism.name / aggregate.name, source.select(columns))
    aggregated = _aggregate_full_roi_workload_artifacts(
        config=config, workload_id=workload_id, output_root=output_root,
    )
    if not source.equals(pq.read_table(aggregated), check_metadata=False):
        raise ValueError(f"full-ROI component aggregation differs from direct analysis: {workload_id}")
    return report


def _regenerate_full_roi_workload_components(
    *, config: PeregrineConfig, raw_root: Path, output_root: Path, workload_id: str,
    mechanisms: tuple[Any, ...], full_roi_window_size: int,
) -> None:
    samples = _full_roi_config_samples(config=config, raw_root=raw_root, workload_id=workload_id)
    if not samples:
        raise ValueError(f"full-ROI workload contains no config samples: {workload_id}")
    features = analyze_full_roi_windows(
        trace_path=load_config_stats_workload(config=config, raw_root=raw_root, workload_id=workload_id).reference_trace_path,
        configs=tuple(_analytical_config(sample.config, config) for sample in samples),
        full_roi_window_size=full_roi_window_size, analysis_window_size=config.analysis.window_size,
        window_count=samples[0].labels.shape[0], mechanisms=mechanisms,
    )
    base = pq.read_table(output_root / "base" / f"{_shard_stem(workload_id)}.parquet").to_pandas()
    expected_rows = features.shape[0] * features.shape[1]
    if len(base) != expected_rows:
        raise ValueError(f"full-ROI base/component row count differs: {workload_id}")
    import pandas as pd

    offset = 0
    identities = list(FULL_ROI_IDENTITY_COLUMNS)
    for mechanism in mechanisms:
        columns = list(analytical_feature_columns((mechanism,)))
        width = len(columns)
        values = features[:, :, offset:offset + width].reshape(expected_rows, width)
        offset += width
        frame = pd.concat([base.loc[:, identities], pd.DataFrame(values, columns=columns)], axis=1)
        _write_table(
            output_root / "components" / mechanism.name / f"{_shard_stem(workload_id)}.parquet",
            pa.Table.from_pandas(frame, preserve_index=False),
        )
    if offset != features.shape[2]:
        raise ValueError("full-ROI selected component feature dimensions differ from configured contract")


def _full_roi_base_columns(config: PeregrineConfig) -> list[str]:
    dynamic = set(analytical_feature_columns(config.microarchitecture.mechanisms))
    context = set(workload_context_feature_columns(config.microarchitecture.mechanisms))
    return [
        *FULL_ROI_IDENTITY_COLUMNS,
        *(column for column in config.feature_columns if column not in dynamic and column not in context),
        *config.label_columns,
    ]


def _aggregate_full_roi_workload_artifacts(
    *, config: PeregrineConfig, workload_id: str, output_root: Path,
) -> Path:
    stem = f"{_shard_stem(workload_id)}.parquet"
    base = pq.read_table(output_root / "base" / stem).to_pandas()
    identities = list(FULL_ROI_IDENTITY_COLUMNS)
    if base.duplicated(identities).any():
        raise ValueError(f"full-ROI base artifact has duplicate identities: {workload_id}")
    parts = [base]
    reference_id = config.microarchitecture.config_id({})
    context_values: dict[str, float] = {}
    for mechanism in config.microarchitecture.mechanisms:
        component = pq.read_table(output_root / "components" / mechanism.name / stem).to_pandas()
        if not component.loc[:, identities].equals(base.loc[:, identities]):
            raise ValueError(f"full-ROI component identities differ from base: {workload_id}/{mechanism.name}")
        dynamic_columns = list(analytical_feature_columns((mechanism,)))
        parts.append(component.loc[:, dynamic_columns])
        reference = component.loc[component["config_id"] == reference_id, dynamic_columns]
        if reference.empty:
            raise ValueError(f"full-ROI component lacks reference configuration: {workload_id}/{mechanism.name}")
        mean_column = f"dynamic_{mechanism.name}_mean"
        values = reference[mean_column].to_numpy(dtype=np.float64)
        context_values.update({
            f"workload_context_mean__{mechanism.name}": float(np.mean(values)),
            f"workload_context_p90__{mechanism.name}": float(np.percentile(values, 90)),
            f"workload_context_std__{mechanism.name}": float(np.std(values)),
            f"workload_context_active_ratio__{mechanism.name}": float(np.mean(values > 0.0)),
        })
    import pandas as pd

    frame = pd.concat(parts, axis=1)
    for column in workload_context_feature_columns(config.microarchitecture.mechanisms):
        frame[column] = context_values[column]
    expected = _full_roi_dataset_columns(config)
    if tuple(frame.columns) != tuple(expected):
        frame = frame.loc[:, expected]
    if not np.isfinite(frame.loc[:, config.feature_columns + config.label_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"full-ROI aggregation contains non-finite values: {workload_id}")
    destination = output_root / "aggregated" / stem
    _write_table(destination, pa.Table.from_pandas(frame, preserve_index=False).cast(_full_roi_dataset_schema(config)))
    _validate_full_roi_dataset_shard(
        destination, workload_id=workload_id, config=config,
        expected_rows=tuple(zip(base.window_index.astype(int), base.config_id.astype(str), strict=True)),
    )
    return destination


def _write_table(destination: Path, table: pa.Table) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    pq.write_table(table, partial, compression="zstd")
    partial.replace(destination)


def _full_roi_config_samples(
    *, config: PeregrineConfig, raw_root: Path, workload_id: str,
) -> tuple[FullRoiConfigSample, ...]:
    raw = load_config_stats_workload(config=config, raw_root=raw_root, workload_id=workload_id)
    samples: list[FullRoiConfigSample] = []
    for sample in raw.samples:
        if sample.config is None:
            raise ValueError(f"config-stats sample lacks parsed configuration: {sample.stats_path}")
        labels = read_label_values(sample.stats_path, config.labels)
        if labels.ndim != 2 or labels.shape[1] != len(config.labels.labels) or not np.isfinite(labels).all():
            raise ValueError(f"invalid full-ROI labels: {sample.stats_path}")
        samples.append(
            FullRoiConfigSample(
                config_id=sample.config_id,
                config=sample.config,
                stats_path=sample.stats_path,
                labels=labels,
            )
        )
    return tuple(samples)


def _write_full_roi_workload_dataset(
    *,
    config: PeregrineConfig,
    workload_id: str,
    trace_path: Path,
    samples: tuple[FullRoiConfigSample, ...],
    destination: Path,
    full_roi_window_size: int,
) -> int:
    schema = _full_roi_dataset_schema(config)
    table_context = _dataset_table_context(config)
    run_configs = tuple(sample.config for sample in samples)
    features = analyze_full_roi_windows(
        trace_path=trace_path,
        configs=tuple(_analytical_config(item, config) for item in run_configs),
        full_roi_window_size=full_roi_window_size,
        analysis_window_size=config.analysis.window_size,
        window_count=samples[0].labels.shape[0],
        mechanisms=config.microarchitecture.mechanisms,
    )
    workload_context = _workload_context_values(
        features,
        reference_config_index=_reference_config_index(run_configs, config),
        table_context=table_context,
    )
    labels = np.stack([sample.labels[1:] for sample in samples], axis=1)
    if features.shape[:2] != labels.shape[:2]:
        raise ValueError(
            f"full-ROI feature/label window dimensions differ for {workload_id}: "
            f"features={features.shape[:2]} labels={labels.shape[:2]}"
        )
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    writer = pq.ParquetWriter(partial, schema, compression="zstd")
    rows = 0
    try:
        for feature_index, window_index in enumerate(range(1, samples[0].labels.shape[0])):
            table = _full_roi_dataset_table(
                workload_id=workload_id,
                window_index=window_index,
                feature_values=features[feature_index],
                workload_context=workload_context,
                labels=labels[feature_index],
                configs=run_configs,
                config=config,
                schema=schema,
                table_context=table_context,
            )
            writer.write_table(table)
            rows += table.num_rows
        if rows == 0:
            raise ValueError(f"Anamol emitted no full-ROI rows for {workload_id}")
        writer.close()
        partial.replace(destination)
        return rows
    except BaseException:
        writer.close()
        partial.unlink(missing_ok=True)
        raise


def _build_workload_dataset(
    *,
    config: PeregrineConfig,
    workload_id: str,
    samples: tuple[RawSample, ...],
    destination: Path,
    workers: int,
) -> int:
    schema = _dataset_schema(config)
    table_context = _dataset_table_context(config)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    rows = 0
    pending: list[pa.Table] = []
    pending_rows = 0
    writer = pq.ParquetWriter(partial, schema, compression="zstd")
    try:
        groups = _samples_by_region(samples)
        with ThreadPoolExecutor(max_workers=min(workers, len(groups))) as executor:
            tables_by_group = executor.map(
                lambda group: _feature_tables(
                    config=config,
                    workload_id=workload_id,
                    region_id=group[0],
                    samples=group[1],
                    schema=schema,
                    table_context=table_context,
                ),
                groups,
            )
            for tables in tables_by_group:
                for table in tables:
                    pending.append(table)
                    pending_rows += len(table)
                    rows += len(table)
                    if pending_rows >= PARQUET_BATCH_ROWS:
                        writer.write_table(pa.concat_tables(pending))
                        pending.clear()
                        pending_rows = 0
        if pending:
            writer.write_table(pa.concat_tables(pending))
        if rows == 0:
            raise ValueError(f"Anamol emitted no dataset rows for {workload_id}")
        writer.close()
        partial.replace(destination)
        return rows
    except BaseException:
        writer.close()
        partial.unlink(missing_ok=True)
        raise


def _feature_tables(
    *,
    config: PeregrineConfig,
    workload_id: str,
    region_id: str,
    samples: tuple[RawSample, ...],
    schema: pa.Schema,
    table_context: DatasetTableContext,
) -> tuple[pa.Table, ...]:
    trace_path = samples[0].trace_path
    row_configs = tuple(sample.config for sample in samples)
    analytical_configs = tuple(_analytical_config(item, config) for item in row_configs)
    labels = np.asarray([sample.labels for sample in samples], dtype=np.float64)
    return tuple(
        _dataset_table(
            workload_id=workload_id,
            batch=batch,
            region_id=region_id,
            configs=row_configs,
            labels=labels,
            config=config,
            schema=schema,
            table_context=table_context,
        )
        for batch in iter_anamol_feature_batches(
            trace_path=trace_path,
            configs=analytical_configs,
            window_size=config.analysis.window_size,
            mechanisms=config.microarchitecture.mechanisms,
        )
    )


def _samples_by_region(samples: tuple[RawSample, ...]) -> tuple[tuple[str, tuple[RawSample, ...]], ...]:
    grouped: dict[str, list[RawSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.region_id, []).append(sample)
    return tuple(
        (region_id, tuple(sorted(items, key=lambda item: item.config.config_id)))
        for region_id, items in sorted(grouped.items())
    )


def _analytical_config(
    run_config: RunConfig,
    config: PeregrineConfig,
) -> dict[str, int | float]:
    return dict(config.microarchitecture.analytical_values(run_config.parameter_values))


def _reference_config_index(run_configs: tuple[RunConfig, ...], config: PeregrineConfig) -> int:
    reference_id = config.microarchitecture.config_id({})
    matches = [index for index, run_config in enumerate(run_configs) if run_config.config_id == reference_id]
    if len(matches) != 1:
        raise ValueError("full-ROI workload must contain exactly one reference config for workload context")
    return matches[0]


def _workload_context_values(
    features: np.ndarray,
    *,
    reference_config_index: int,
    table_context: DatasetTableContext,
) -> np.ndarray:
    if features.ndim != 3:
        raise ValueError("workload context requires a window/config/feature tensor")
    if not 0 <= reference_config_index < features.shape[1]:
        raise ValueError("reference config index is outside the feature tensor")
    reference = features[:, reference_config_index, :]
    if reference.shape[0] <= 0 or not np.isfinite(reference).all():
        raise ValueError("reference workload context features are invalid")
    by_name = {column: index for index, column in enumerate(table_context.analytical_columns)}
    values: list[float] = []
    for context_column in table_context.workload_context_columns[::4]:
        mechanism_name = context_column.rsplit("__", maxsplit=1)[1]
        source = reference[:, by_name[f"dynamic_{mechanism_name}_mean"]]
        values.extend((
            float(np.mean(source)),
            float(np.percentile(source, 90)),
            float(np.std(source)),
            float(np.mean(source > 0.0)),
        ))
    return np.asarray(values, dtype=np.float64)


def _dataset_schema(config: PeregrineConfig) -> pa.Schema:
    return pa.schema(
        [
            pa.field("workload_id", pa.string()),
            pa.field("region_id", pa.string()),
            pa.field("config_id", pa.string()),
            *(pa.field(column, pa.float64()) for column in config.feature_columns),
            *(pa.field(column, pa.float64()) for column in config.label_columns),
        ]
    )


def _full_roi_dataset_schema(config: PeregrineConfig) -> pa.Schema:
    return pa.schema(
        [
            pa.field("workload_id", pa.string()),
            pa.field("window_index", pa.int64()),
            pa.field("config_id", pa.string()),
            *(pa.field(column, pa.float64()) for column in config.feature_columns),
            *(pa.field(column, pa.float64()) for column in config.label_columns),
        ]
    )


def _dataset_columns(config: PeregrineConfig) -> list[str]:
    return [
        *IDENTITY_COLUMNS,
        *config.feature_columns,
        *config.label_columns,
    ]


def _full_roi_dataset_columns(config: PeregrineConfig) -> list[str]:
    return [
        *FULL_ROI_IDENTITY_COLUMNS,
        *config.feature_columns,
        *config.label_columns,
    ]


def _dataset_table_context(config: PeregrineConfig) -> DatasetTableContext:
    return DatasetTableContext(
        analytical_columns=analytical_feature_columns(config.microarchitecture.mechanisms),
        workload_context_columns=workload_context_feature_columns(config.microarchitecture.mechanisms),
        categorical_values={
            parameter_name: _categorical_values(parameter_name, config=config)
            for parameter_name in config.categorical_design_parameter_names
        },
        reciprocal_parameters=tuple(
            column.removeprefix("inv_")
            for column in reciprocal_feature_columns()
        ),
    )


def _full_roi_dataset_table(
    *,
    workload_id: str,
    window_index: int,
    feature_values: np.ndarray,
    workload_context: np.ndarray,
    labels: np.ndarray,
    configs: tuple[RunConfig, ...],
    config: PeregrineConfig,
    schema: pa.Schema,
    table_context: DatasetTableContext,
) -> pa.Table:
    table = _full_roi_feature_table(
        workload_id=workload_id,
        window_index=window_index,
        feature_values=feature_values,
        workload_context=workload_context,
        configs=configs,
        config=config,
        table_context=table_context,
    )
    config_count = len(configs)
    if labels.shape != (config_count, len(config.label_columns)):
        raise ValueError("full-ROI label dimensions differ from the configured contract")
    if not np.isfinite(labels).all():
        raise ValueError("full-ROI dataset table contains non-finite generated values")
    for index in range(labels.shape[1]):
        table = table.append_column(config.label_columns[index], pa.array(labels[:, index], type=pa.float64()))
    return table.cast(schema)


def _full_roi_feature_table(
    *,
    workload_id: str,
    window_index: int,
    feature_values: np.ndarray,
    workload_context: np.ndarray,
    configs: tuple[RunConfig, ...],
    config: PeregrineConfig,
    table_context: DatasetTableContext,
) -> pa.Table:
    config_count = len(configs)
    if feature_values.shape != (config_count, len(table_context.analytical_columns)):
        raise ValueError("Anamol full-ROI feature dimensions differ from the configured contract")
    if workload_context.shape != (len(table_context.workload_context_columns),):
        raise ValueError("workload context dimensions differ from the configured contract")
    arrays: list[pa.Array] = [
        pa.array([workload_id] * config_count, type=pa.string()),
        pa.array([window_index] * config_count, type=pa.int64()),
        pa.array([run_config.config_id for run_config in configs], type=pa.string()),
    ]
    arrays.extend(
        pa.array(feature_values[:, index], type=pa.float64())
        for index in range(feature_values.shape[1])
    )
    arrays.extend(
        pa.array(
            np.fromiter(
                (float(run_config.parameter_values[parameter]) for run_config in configs),
                dtype=np.float64,
                count=config_count,
            ),
            type=pa.float64(),
        )
        for parameter in config.numeric_design_parameter_names
    )
    for parameter_name in config.categorical_design_parameter_names:
        values = tuple(str(run_config.parameter_values[parameter_name]) for run_config in configs)
        for category in table_context.categorical_values[parameter_name]:
            arrays.append(
                pa.array(
                    np.fromiter(
                        (1.0 if value == category else 0.0 for value in values),
                        dtype=np.float64,
                        count=config_count,
                    ),
                    type=pa.float64(),
                )
            )
    analytical_configs = tuple(_analytical_config(item, config) for item in configs)
    arrays.extend(
        pa.array(
            np.fromiter(
                (1.0 / float(item[parameter]) for item in analytical_configs),
                dtype=np.float64,
                count=config_count,
            ),
            type=pa.float64(),
        )
        for parameter in table_context.reciprocal_parameters
    )
    arrays.extend(
        pa.array([float(value)] * config_count, type=pa.float64())
        for value in workload_context
    )
    if not np.isfinite(feature_values).all() or not np.isfinite(workload_context).all():
        raise ValueError("full-ROI feature table contains non-finite generated values")
    return pa.Table.from_arrays(
        arrays,
        names=["workload_id", "window_index", "config_id", *config.feature_columns],
    )


def build_candidate_feature_table(
    *,
    config: PeregrineConfig,
    workload_id: str,
    trace_path: str | Path,
    window_count: int,
    run_configs: tuple[RunConfig, ...],
    full_roi_window_size: int,
) -> pa.Table:
    """Analyze one existing rich trace for a batch of candidate configurations."""
    features = analyze_full_roi_windows(
        trace_path=trace_path,
        configs=(
            _analytical_config(
                config.run_config_from_args(
                    config.microarchitecture.config_id({}),
                    config.microarchitecture.gem5_args(),
                ),
                config,
            ),
            *tuple(_analytical_config(item, config) for item in run_configs),
        ),
        full_roi_window_size=full_roi_window_size,
        analysis_window_size=config.analysis.window_size,
        window_count=window_count,
        mechanisms=config.microarchitecture.mechanisms,
    )
    context = _dataset_table_context(config)
    workload_context = _workload_context_values(features, reference_config_index=0, table_context=context)
    candidate_features = features[:, 1:, :]
    return pa.concat_tables([
        _full_roi_feature_table(
            workload_id=workload_id,
            window_index=window_index,
            feature_values=candidate_features[feature_index],
            workload_context=workload_context,
            configs=run_configs,
            config=config,
            table_context=context,
        )
        for feature_index, window_index in enumerate(range(1, window_count))
    ])


def _dataset_table(
    *,
    workload_id: str,
    batch: FeatureBatch,
    region_id: str,
    configs: tuple[RunConfig, ...],
    labels: np.ndarray,
    config: PeregrineConfig,
    schema: pa.Schema,
    table_context: DatasetTableContext,
) -> pa.Table:
    config_count = len(configs)
    if batch.values.shape != (config_count, len(table_context.analytical_columns)):
        raise ValueError("Anamol feature batch dimensions differ from the configured contract")
    workload_context = _batch_workload_context_values(batch.values, table_context)
    arrays: list[pa.Array] = [
        pa.array([workload_id] * config_count, type=pa.string()),
        pa.array([region_id] * config_count, type=pa.string()),
        pa.array([run_config.config_id for run_config in configs], type=pa.string()),
    ]
    arrays.extend(
        pa.array(batch.values[:, index], type=pa.float64())
        for index in range(batch.values.shape[1])
    )
    arrays.extend(
        pa.array(
            np.fromiter(
                (float(run_config.parameter_values[parameter]) for run_config in configs),
                dtype=np.float64,
                count=config_count,
            ),
            type=pa.float64(),
        )
        for parameter in config.numeric_design_parameter_names
    )
    for parameter_name in config.categorical_design_parameter_names:
        values = tuple(
            str(run_config.parameter_values[parameter_name])
            for run_config in configs
        )
        for category in table_context.categorical_values[parameter_name]:
            arrays.append(
                pa.array(
                    np.fromiter(
                        (1.0 if value == category else 0.0 for value in values),
                        dtype=np.float64,
                        count=config_count,
                    ),
                    type=pa.float64(),
                )
            )
    analytical_configs = tuple(_analytical_config(item, config) for item in configs)
    arrays.extend(
        pa.array(
            np.fromiter(
                (1.0 / float(item[parameter]) for item in analytical_configs),
                dtype=np.float64,
                count=config_count,
            ),
            type=pa.float64(),
        )
        for parameter in table_context.reciprocal_parameters
    )
    arrays.extend(
        pa.array([float(value)] * config_count, type=pa.float64())
        for value in workload_context
    )
    if not np.isfinite(batch.values).all() or not np.isfinite(labels).all():
        raise ValueError("dataset table contains non-finite generated values")
    arrays.extend(
        pa.array(labels[:, index], type=pa.float64())
        for index in range(labels.shape[1])
    )
    return pa.Table.from_arrays(arrays, schema=schema)


def _batch_workload_context_values(values: np.ndarray, table_context: DatasetTableContext) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] <= 0 or not np.isfinite(values).all():
        raise ValueError("batch workload context features are invalid")
    expanded = values[:, np.newaxis, :]
    return _workload_context_values(expanded, reference_config_index=0, table_context=table_context)




def _validate_dataset_shard(
    path: str | Path,
    *,
    workload_id: str,
    config: PeregrineConfig,
    expected_rows: tuple[tuple[str, str], ...],
) -> None:
    shard = Path(path)
    expected_columns = _dataset_columns(config)
    parquet = pq.ParquetFile(shard)
    if parquet.schema_arrow.names != expected_columns:
        raise ValueError(f"dataset shard contract differs from current source: {shard}")
    if not expected_rows:
        raise ValueError("dataset shard validation requires non-empty expected rows")
    row_count = len(expected_rows)
    if parquet.metadata.num_rows != row_count:
        raise ValueError(f"dataset shard row arithmetic differs from raw stats: {shard}")
    frame = pq.read_table(shard, columns=list(IDENTITY_COLUMNS)).to_pandas()
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"dataset shard contains duplicate identities: {shard}")
    if set(frame.workload_id.astype(str)) != {workload_id}:
        raise ValueError(f"dataset shard workload identity differs: {shard}")
    if frame["region_id"].astype(str).eq("").any():
        raise ValueError(f"dataset shard contains empty region identities: {shard}")
    observed = tuple(
        sorted(
            zip(
                frame.region_id.astype(str).tolist(),
                frame.config_id.astype(str).tolist(),
                strict=True,
            )
        )
    )
    if observed != tuple(sorted(expected_rows)):
        raise ValueError(f"dataset shard sample identities differ: {shard}")


def _validate_full_roi_dataset_shard(
    path: str | Path,
    *,
    workload_id: str,
    config: PeregrineConfig,
    expected_rows: tuple[tuple[int, str], ...],
) -> None:
    shard = Path(path)
    expected_columns = _full_roi_dataset_columns(config)
    parquet = pq.ParquetFile(shard)
    if parquet.schema_arrow.names != expected_columns:
        raise ValueError(f"full-ROI dataset shard contract differs from current source: {shard}")
    if not expected_rows:
        raise ValueError("full-ROI dataset shard validation requires non-empty expected rows")
    if parquet.metadata.num_rows != len(expected_rows):
        raise ValueError(f"full-ROI dataset shard row arithmetic differs from raw stats: {shard}")
    frame = pq.read_table(shard, columns=list(FULL_ROI_IDENTITY_COLUMNS)).to_pandas()
    if frame.duplicated(list(FULL_ROI_IDENTITY_COLUMNS)).any():
        raise ValueError(f"full-ROI dataset shard contains duplicate identities: {shard}")
    if set(frame.workload_id.astype(str)) != {workload_id}:
        raise ValueError(f"full-ROI dataset shard workload identity differs: {shard}")
    if (frame["window_index"].astype(int) < 0).any():
        raise ValueError(f"full-ROI dataset shard contains negative window identities: {shard}")
    observed = tuple(
        sorted(
            zip(
                frame.window_index.astype(int).tolist(),
                frame.config_id.astype(str).tolist(),
                strict=True,
            )
        )
    )
    if observed != tuple(sorted(expected_rows)):
        raise ValueError(f"full-ROI dataset shard sample identities differ: {shard}")


def _shard_stem(workload_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", workload_id):
        raise ValueError(f"dataset shard identity is not filesystem-safe: {workload_id}")
    return workload_id


def _categorical_values(parameter_name: str, *, config: PeregrineConfig) -> tuple[str, ...]:
    parameter = config.microarchitecture.parameters_by_name[parameter_name]
    if parameter.param_type != "categorical":
        raise ValueError(f"design parameter is not categorical: {parameter_name}")
    return tuple(str(value) for value in parameter.values)
