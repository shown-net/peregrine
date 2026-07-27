from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import re
import shlex
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .design_space import PeregrineConfig, reciprocal_feature_columns
from .feature_pipeline import (
    FeatureBatch,
    analytical_feature_columns,
    iter_anamol_feature_batches,
)
from .gem5_stats import read_label_values
from .run_config import RunConfig


IDENTITY_COLUMNS = ("workload_id", "region_id", "config_id")
TRACE_FILE = "peregrine.trace.pb.zst"
PARQUET_BATCH_ROWS = 8192
FINAL_CONFIG_ID = re.compile(r"config_[0-9a-f]{16}$")


@dataclass(frozen=True)
class RawSample:
    workload_id: str
    region_id: str
    config: RunConfig
    labels: tuple[float, ...]
    trace_path: Path


def build_dataset_shards(
    *,
    config: PeregrineConfig,
    raw_root: str | Path,
    output_dir: str | Path,
    workload_ids: tuple[str, ...] | None = None,
    workers: int = 24,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("dataset workers must be positive")
    raw = Path(raw_root).resolve()
    workloads_root = raw / "workloads"
    if not workloads_root.is_dir():
        raise ValueError(f"raw root contains no workloads: {raw}")
    by_workload = {
        path.name: path
        for path in workloads_root.iterdir()
        if path.is_dir()
    }
    canonical_workloads = tuple(workload_ids) if workload_ids is not None else tuple(sorted(by_workload))
    if not canonical_workloads or len(set(canonical_workloads)) != len(canonical_workloads):
        raise ValueError("dataset workload selection is empty or duplicated")
    if set(canonical_workloads) - set(by_workload):
        raise ValueError("raw workload layout differs from the canonical roster")
    workload_roots = tuple(by_workload[workload] for workload in canonical_workloads)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for partial in out.glob("*.partial"):
        partial.unlink()
    expected_shards = {
        f"{_shard_stem(workload_id)}.parquet"
        for workload_id in canonical_workloads
    }
    workload_reports: list[dict[str, int | str]] = []
    for stale in out.glob("*.parquet"):
        if stale.name not in expected_shards:
            stale.unlink()

    for workload_root in workload_roots:
        workload_id = workload_root.name
        regions_root = workload_root / "regions"
        if not regions_root.is_dir():
            raise FileNotFoundError(f"missing planned region directory: {regions_root}")
        region_ids = tuple(sorted(path.name for path in regions_root.iterdir() if path.is_dir()))
        if not region_ids:
            raise ValueError(f"raw dataset contains no region samples: {workload_root}")
        shard = out / f"{_shard_stem(workload_id)}.parquet"
        samples, skipped = _discover_raw_samples(
            workload_root,
            workload_id=workload_id,
            config=config,
        )
        expected_rows = tuple((item.region_id, item.config.config_id) for item in samples)
        if not expected_rows:
            raise ValueError(f"raw dataset contains no valid samples: {workload_root}")
        if shard.exists():
            try:
                _validate_dataset_shard(
                    shard,
                    workload_id=workload_id,
                    config=config,
                    expected_rows=expected_rows,
                )
            except (OSError, TypeError, ValueError):
                shard.unlink()
            else:
                workload_reports.append({"workload_id": workload_id, "rows": len(samples), "skipped": skipped})
                continue
        rows = _build_workload_dataset(
            config=config,
            workload_root=workload_root,
            workload_id=workload_id,
            samples=samples,
            destination=shard,
            workers=workers,
        )
        _validate_dataset_shard(
            shard,
            workload_id=workload_id,
            config=config,
            expected_rows=expected_rows,
        )
        workload_reports.append({"workload_id": workload_id, "rows": rows, "skipped": skipped})
    return {
        "dataset": str(out),
        "shards": sorted(expected_shards),
        "workloads": workload_reports,
    }


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
        config.microarchitecture.parameters_by_name[name].gem5_arg
        for name in config.design_parameter_names
    }
    gem5_args = tuple(
        item
        for item in shlex.split(command)
        if item.startswith("--") and item.partition("=")[0] in parameter_flags
    )
    return config.run_config_from_args(sample_root.name, gem5_args)


def _build_workload_dataset(
    *,
    config: PeregrineConfig,
    workload_root: Path,
    workload_id: str,
    samples: tuple[RawSample, ...],
    destination: Path,
    workers: int,
) -> int:
    schema = _dataset_schema(config)
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
                    workload_root=workload_root,
                    workload_id=workload_id,
                    region_id=group[0],
                    samples=group[1],
                    schema=schema,
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
    workload_root: Path,
    workload_id: str,
    region_id: str,
    samples: tuple[RawSample, ...],
    schema: pa.Schema,
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


def _dataset_table(
    *,
    workload_id: str,
    batch: FeatureBatch,
    region_id: str,
    configs: tuple[RunConfig, ...],
    labels: np.ndarray,
    config: PeregrineConfig,
    schema: pa.Schema,
) -> pa.Table:
    config_count = len(configs)
    analytical_columns = analytical_feature_columns(config.microarchitecture.mechanisms)
    if batch.values.shape != (config_count, len(analytical_columns)):
        raise ValueError("Anamol feature batch dimensions differ from the configured contract")
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
        for category in _categorical_values(parameter_name, config=config):
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
    reciprocal_parameters = tuple(
        column.removeprefix("inv_")
        for column in reciprocal_feature_columns()
    )
    arrays.extend(
        pa.array(
            np.fromiter(
                (1.0 / float(_analytical_config(item, config)[parameter]) for item in configs),
                dtype=np.float64,
                count=config_count,
            ),
            type=pa.float64(),
        )
        for parameter in reciprocal_parameters
    )
    arrays.extend(
        pa.array(labels[:, index], type=pa.float64())
        for index in range(labels.shape[1])
    )
    return pa.Table.from_arrays(arrays, schema=schema)




def _validate_dataset_shard(
    path: str | Path,
    *,
    workload_id: str,
    config: PeregrineConfig,
    expected_rows: tuple[tuple[str, str], ...],
) -> None:
    shard = Path(path)
    expected_columns = [
        *IDENTITY_COLUMNS,
        *config.feature_columns,
        *config.label_columns,
    ]
    parquet = pq.ParquetFile(shard)
    if parquet.schema_arrow.names != expected_columns:
        raise ValueError(f"dataset shard contract differs from current source: {shard}")
    if not expected_rows:
        raise ValueError("dataset shard validation requires non-empty expected rows")
    row_count = len(expected_rows)
    if parquet.metadata.num_rows != row_count:
        raise ValueError(f"dataset shard row arithmetic differs from raw stats: {shard}")
    frame = pq.read_table(shard, columns=expected_columns).to_pandas()
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
    numeric = frame[
        [*config.feature_columns, *config.label_columns]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"dataset shard contains non-finite values: {shard}")


def _shard_stem(workload_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", workload_id):
        raise ValueError(f"dataset shard identity is not filesystem-safe: {workload_id}")
    return workload_id


def _categorical_values(parameter_name: str, *, config: PeregrineConfig) -> tuple[str, ...]:
    parameter = config.microarchitecture.parameters_by_name[parameter_name]
    if parameter.param_type != "categorical":
        raise ValueError(f"design parameter is not categorical: {parameter_name}")
    return tuple(str(value) for value in parameter.values)
