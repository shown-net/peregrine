"""Direct L2 search over CPU-owned legal configurations and L1 predictions."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from anamol.python.dataset import build_candidate_feature_table
from anamol.python.design_space import PeregrineConfig
from anamol.python.run_config import RunConfig
from ml_model.inference import PredictorBundle


def generate_legal_configs(
    *, config: PeregrineConfig, count: int, seed: int, exclude_ids: set[str] | None = None,
) -> tuple[RunConfig, ...]:
    if count < 1:
        raise ValueError("L2 exploration count must be positive")
    names = config.design_parameter_names
    domains = {name: config.microarchitecture.parameters_by_name[name].values for name in names}
    excluded = set(exclude_ids or ())
    result: list[RunConfig] = []
    rng = random.Random(seed)
    attempts = 0
    while len(result) < count and attempts < count * 10000:
        attempts += 1
        values = {name: rng.choice(domains[name]) for name in names}
        try:
            resolved = config.microarchitecture.parameter_values(values)
        except ValueError:
            continue
        config_id = config.microarchitecture.config_id(resolved)
        if config_id in excluded:
            continue
        excluded.add(config_id)
        result.append(RunConfig(config_id, resolved, config.microarchitecture.gem5_args(resolved)))
    if len(result) != count:
        raise ValueError("unable to generate requested distinct legal L2 configurations")
    return tuple(result)


def pareto_mask(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("Pareto objectives must be a finite nonempty matrix")
    keep = np.ones(len(matrix), dtype=bool)
    for index, row in enumerate(matrix):
        keep[index] = not np.any(np.all(matrix <= row, axis=1) & np.any(matrix < row, axis=1))
    return keep


def prioritized_pareto_indices(values: np.ndarray, config_ids: tuple[str, ...], limit: int) -> tuple[int, ...]:
    if limit < 1 or len(values) != len(config_ids):
        raise ValueError("invalid L2 queue request")
    front = np.flatnonzero(pareto_mask(values))
    if not len(front):
        return ()
    lo = values[front].min(axis=0)
    span = np.maximum(values[front].max(axis=0) - lo, 1e-12)
    normalized = (values[front] - lo) / span
    first = min(range(len(front)), key=lambda offset: config_ids[front[offset]])
    chosen = [first]
    while len(chosen) < min(limit, len(front)):
        remaining = [offset for offset in range(len(front)) if offset not in chosen]
        next_offset = min(
            remaining,
            key=lambda offset: (
                -min(float(np.linalg.norm(normalized[offset] - normalized[item])) for item in chosen),
                config_ids[front[offset]],
            ),
        )
        chosen.append(next_offset)
    return tuple(int(front[offset]) for offset in chosen)


def _existing_config_ids(dataset_dir: Path, config: PeregrineConfig) -> set[str]:
    first = next(iter(sorted(dataset_dir.glob("*.parquet"))), None)
    if first is None:
        raise FileNotFoundError(f"L1 dataset contains no shards: {dataset_dir}")
    columns = [*config.numeric_design_parameter_names, *config.categorical_feature_columns]
    frame = pq.read_table(first, columns=columns).to_pandas().drop_duplicates()
    result = set()
    for row in frame.to_dict("records"):
        values: dict[str, int | float | str] = {}
        for name in config.numeric_design_parameter_names:
            parameter = config.microarchitecture.parameters_by_name[name]
            values[name] = int(row[name]) if parameter.param_type == "int" else float(row[name])
        for name in config.categorical_design_parameter_names:
            parameter = config.microarchitecture.parameters_by_name[name]
            selected = [value for value in parameter.values if float(row[f"{name}_{value}"]) == 1.0]
            if len(selected) != 1:
                raise ValueError(f"dataset categorical design value is invalid: {name}")
            values[name] = selected[0]
        result.add(config.microarchitecture.config_id(values))
    return result


def _candidate_predictions(
    *, config: PeregrineConfig, predictor: PredictorBundle, candidates: tuple[RunConfig, ...],
    raw_root: Path, dataset_dir: Path, analysis_batch_size: int, inference_batch_size: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    expected_rows: dict[tuple[str, str], int] = {}
    for shard in sorted(dataset_dir.glob("*.parquet")):
        workload_id = shard.stem
        identities = pq.read_table(shard, columns=["window_index", "config_id"]).to_pandas()
        baseline = identities[identities.config_id.astype(str) == "baseline"]
        window_count = int(baseline.window_index.nunique())
        if len(baseline) != window_count or window_count < 1:
            raise ValueError(f"L1 shard has invalid baseline window identities: {shard}")
        expected_rows.update({
            (workload_id, item.config_id): window_count
            for item in candidates
        })
        trace = raw_root / "workloads" / workload_id / "configs" / "baseline" / "peregrine.trace.pb.zst"
        if not trace.is_file() or trace.stat().st_size == 0:
            raise FileNotFoundError(f"missing shared L1 trace: {trace}")
        log_path = trace.parent / "run.log"
        match = re.search(
            r"PEREGRINE_L1_FULL_ROI_WINDOW_DUMP window_index=0 length=(\d+)",
            log_path.read_text(encoding="utf-8", errors="replace"),
        ) if log_path.is_file() else None
        if match is None:
            raise ValueError(f"missing CPU-owned full-ROI window identity: {log_path}")
        full_roi_window_size = int(match.group(1))
        for start in range(0, len(candidates), analysis_batch_size):
            selected = candidates[start:start + analysis_batch_size]
            table = build_candidate_feature_table(
                config=config, workload_id=workload_id, trace_path=trace,
                window_count=window_count, run_configs=selected,
                full_roi_window_size=full_roi_window_size,
            )
            for batch in table.to_batches(max_chunksize=inference_batch_size):
                output = pa.Table.from_batches([batch]).select(list(predictor.identity_columns)).to_pandas()
                for metric, values in predictor.predict_batch(batch, batch_size=inference_batch_size).items():
                    output[f"prediction_{metric}"] = values
                frames.append(output)
    result = pd.concat(frames, ignore_index=True)
    identities = ["workload_id", "window_index", "config_id"]
    if result.duplicated(identities).any() or not np.isfinite(result.select_dtypes(include=[np.number])).all().all():
        raise ValueError("L2 produced invalid prediction identities or values")
    actual_rows = result.groupby(["workload_id", "config_id"], sort=True).size().to_dict()
    if actual_rows != expected_rows:
        raise ValueError("L2 prediction coverage differs from the candidate/workload windows")
    return result


def _proxy_robustness(predictions: pd.DataFrame, objectives: tuple[str, ...], full_queue: set[str], queue_size: int) -> dict[str, Any]:
    result = {}
    columns = [f"prediction_{metric}" for metric in objectives]
    per_workload = predictions.groupby(["workload_id", "config_id"], sort=True)[columns].mean()
    for heldout in sorted(predictions.workload_id.unique()):
        training = per_workload.loc[per_workload.index.get_level_values("workload_id") != heldout].groupby("config_id", sort=True).mean()
        heldout_values = per_workload.loc[heldout].reindex(training.index)
        ids = tuple(str(value) for value in training.index)
        selected_indices = prioritized_pareto_indices(training.to_numpy(), ids, queue_size)
        selected = {ids[index] for index in selected_indices}
        heldout_ranks = heldout_values.rank(method="average", pct=True)
        result[str(heldout)] = {
            "scope": "proxy_selection_robustness",
            "queue_overlap": len(selected & full_queue) / max(1, len(selected | full_queue)),
            "heldout_mean_percentile_by_objective": {
                metric: float(heldout_ranks.loc[list(selected), f"prediction_{metric}"].mean())
                for metric in objectives
            },
            "aggregate_to_heldout_rank_correlation": {
                metric: _rank_correlation(
                    training[f"prediction_{metric}"],
                    heldout_values[f"prediction_{metric}"],
                )
                for metric in objectives
            },
        }
    return result


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    value = float(left.corr(right, method="spearman"))
    return value if np.isfinite(value) else 0.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def consume_candidate_queue(queue_path: str | Path, *, config: PeregrineConfig) -> dict[str, Any]:
    queue = json.loads(Path(queue_path).read_text(encoding="utf-8"))
    objectives = tuple(queue["objectives"])
    configured_objectives = tuple(
        label.metric_id for label in config.labels.labels if label.role == "minimize"
    )
    if objectives != configured_objectives:
        raise ValueError("L2 queue objectives differ from the CPU metric configuration")
    batches: dict[int, list[dict[str, Any]]] = {}
    config_ids: set[str] = set()
    priorities: set[int] = set()
    for item in queue["candidates"]:
        values = config.microarchitecture.parameter_values(item["parameter_values"])
        if config.microarchitecture.config_id(values) != item["config_id"]:
            raise ValueError("L2 queue configuration identity is not canonical")
        if set(item["predicted_objectives"]) != set(objectives) or item["evidence"] != "l1_prediction":
            raise ValueError("L2 queue objective evidence is invalid")
        predicted = np.asarray(list(item["predicted_objectives"].values()), dtype=np.float64)
        priority = int(item["priority"])
        batch = int(item["batch"])
        if not np.isfinite(predicted).all() or priority < 1 or batch < 0:
            raise ValueError("L2 queue priority, batch, or prediction is invalid")
        if item["config_id"] in config_ids or priority in priorities:
            raise ValueError("L2 queue contains duplicate configuration or priority identities")
        config_ids.add(item["config_id"])
        priorities.add(priority)
        batches.setdefault(batch, []).append({
            "priority": priority, "config_id": item["config_id"],
            "parameter_values": dict(values),
            "gem5_args": config.microarchitecture.gem5_args(values),
            "predicted_objectives": item["predicted_objectives"], "evidence": item["evidence"],
        })
    if priorities != set(range(1, len(priorities) + 1)):
        raise ValueError("L2 queue priorities must be contiguous")
    return {
        "source_bundle": queue["source_bundle"], "objectives": objectives,
        "batches": [{"batch": batch, "candidates": sorted(items, key=lambda item: item["priority"])} for batch, items in sorted(batches.items())],
    }


def search_l2(
    *, config: PeregrineConfig, bundle_path: str | Path, raw_root: str | Path,
    dataset_dir: str | Path, output_dir: str | Path, exploration_count: int,
    queue_size: int, queue_batch_size: int, analysis_batch_size: int,
    inference_batch_size: int, seed: int,
) -> dict[str, Any]:
    if queue_size < 1 or queue_size > exploration_count:
        raise ValueError("L2 queue size must be within the exploration count")
    if queue_batch_size < 1 or analysis_batch_size < 1 or inference_batch_size < 1:
        raise ValueError("L2 batch sizes must be positive")
    objectives = tuple(label.metric_id for label in config.labels.labels if label.role == "minimize")
    if not objectives:
        raise ValueError("CPU metric configuration contains no minimization objectives")
    predictor = PredictorBundle(bundle_path, num_threads=config.training.num_threads)
    if not set(objectives) <= set(predictor.output_metrics):
        raise ValueError("L1 bundle does not emit every configured L2 objective")
    dataset = Path(dataset_dir)
    candidates = generate_legal_configs(
        config=config, count=exploration_count, seed=seed,
        exclude_ids=_existing_config_ids(dataset, config),
    )
    predictions = _candidate_predictions(
        config=config, predictor=predictor, candidates=candidates, raw_root=Path(raw_root),
        dataset_dir=dataset, analysis_batch_size=analysis_batch_size,
        inference_batch_size=inference_batch_size,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "search_predictions.parquet"
    partial = predictions_path.with_suffix(".parquet.partial")
    predictions.to_parquet(partial, index=False)
    partial.replace(predictions_path)
    columns = [f"prediction_{metric}" for metric in objectives]
    per_workload = predictions.groupby(["workload_id", "config_id"], sort=True)[columns].mean()
    aggregate = per_workload.groupby("config_id", sort=True).mean()
    ids = tuple(str(value) for value in aggregate.index)
    pareto = pareto_mask(aggregate.to_numpy())
    archive = aggregate.reset_index()
    archive["pareto"] = pareto
    archive_path = output / "candidate_archive.parquet"
    archive_partial = archive_path.with_suffix(".parquet.partial")
    archive.to_parquet(archive_partial, index=False)
    archive_partial.replace(archive_path)
    selected_indices = prioritized_pareto_indices(aggregate.to_numpy(), ids, queue_size)
    by_id = {item.config_id: item for item in candidates}
    queue = {
        "source_bundle": str(Path(bundle_path).resolve()), "objectives": objectives,
        "candidates": [{
            "priority": priority, "batch": (priority - 1) // queue_batch_size,
            "config_id": ids[index], "parameter_values": dict(by_id[ids[index]].parameter_values),
            "predicted_objectives": {metric: float(aggregate.iloc[index][f"prediction_{metric}"]) for metric in objectives},
            "evidence": "l1_prediction",
        } for priority, index in enumerate(selected_indices, start=1)],
    }
    queue_path = output / "candidate_queue.json"
    _write_json(queue_path, queue)
    handoff = consume_candidate_queue(queue_path, config=config)
    handoff_path = output / "validation_handoff.json"
    _write_json(handoff_path, handoff)
    full_queue = {ids[index] for index in selected_indices}
    report = {
        "scope": "proxy_selection_not_measured_configuration_optimality",
        "explored_configs": len(candidates), "prediction_rows": len(predictions),
        "pareto_configs": int(pareto.sum()),
        "queue_configs": len(queue["candidates"]), "objectives": objectives,
        "proxy_selection_robustness": _proxy_robustness(predictions, objectives, full_queue, queue_size),
        "predictions": str(predictions_path), "archive": str(archive_path),
        "queue": str(queue_path), "handoff": str(handoff_path),
    }
    report_path = output / "search_report.json"
    _write_json(report_path, report)
    return report
