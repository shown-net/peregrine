"""Canonical fixed-model training and leakage-free grouped evaluation."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .dataset_io import read_dataset_shards
from .multitask import FittedMultiHead
from .multitask import MultiHeadTraining
from .multitask import fit_multihead
from .multitask import fit_multihead_epochs
from .multitask import predict_multihead
from .multitask import regression_error_report


VALIDATION_FRACTION = 0.15


@dataclass(frozen=True)
class FeatureSet:
    feature_set_id: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class PredictionTask:
    task_id: str
    identity_columns: tuple[str, ...]
    group_column: str
    feature_set: FeatureSet
    label_columns: tuple[str, ...]
    output_metrics: tuple[str, ...]
    training: MultiHeadTraining
    num_threads: int
    seed: int
    evaluation_folds: int = 5
    data_limitations: tuple[str, ...] = ()


def train_prediction_task(
    *, task: PredictionTask, dataset_dir: str | Path, output_dir: str | Path,
    workload_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Fit one fixed scalar predictor per target on all selected data."""
    frame = _read_task_frame(task, dataset_dir, workload_ids)
    truth = frame.loc[:, task.label_columns].to_numpy(dtype=np.float32)
    features = frame.loc[:, task.feature_set.columns].to_numpy(dtype=np.float32)
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir()
    models = partial / "models"
    models.mkdir()
    selected: dict[str, dict[str, object]] = {}
    paths: dict[str, str] = {}
    for index, metric in enumerate(task.output_metrics):
        fitted = fit_multihead_epochs(
            training=task.training, features=features, labels=truth[:, index:index + 1],
            epochs=task.training.max_epochs, seed=task.seed + index,
            label_columns=(task.label_columns[index],),
        )
        candidate_id = f"singlehead:{task.feature_set.feature_set_id}:{metric}"
        path = models / f"{candidate_id.replace(':', '_')}.pt"
        torch.save(_checkpoint(fitted, task.feature_set.columns, (task.label_columns[index],), task, metric), path)
        selected[metric] = {
            "candidate_id": candidate_id, "predictor_kind": "singlehead",
            "feature_set": task.feature_set.feature_set_id,
        }
        paths[candidate_id] = str(path.relative_to(partial))
    report = {
        "task_id": task.task_id,
        "protocol": "full_dataset_deployment_training",
        "samples": len(frame),
        "workloads": int(frame["workload_id"].nunique()) if "workload_id" in frame else 0,
        "configurations": int(frame["config_id"].nunique()) if "config_id" in frame else 0,
        "selected": selected,
    }
    training_report = partial / "training_report.json"
    training_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bundle = partial / "predictor_bundle.json"
    bundle.write_text(json.dumps({
        "task_id": task.task_id, "identity_columns": task.identity_columns,
        "output_metrics": task.output_metrics, "selected": selected, "model_paths": paths,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    from .inference import PredictorBundle

    PredictorBundle(bundle, num_threads=task.num_threads)
    previous = output.with_name(f".{output.name}.previous")
    if previous.exists():
        raise RuntimeError(f"previous deployment model directory exists: {previous}")
    if output.exists():
        output.replace(previous)
    try:
        partial.replace(output)
    except BaseException:
        if previous.exists():
            previous.replace(output)
        raise
    shutil.rmtree(previous, ignore_errors=True)
    return {
        "bundle": str(output / bundle.name),
        "training_report": str(output / training_report.name),
    }


def evaluate_prediction_task(
    *, task: PredictionTask, dataset_dir: str | Path, output_dir: str | Path,
    workload_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Evaluate fixed scalar models with whole-group train/validation/test isolation."""
    frame = _read_task_frame(task, dataset_dir, workload_ids)
    values = frame[task.group_column].astype(str).to_numpy()
    groups = tuple(sorted(set(values)))
    evaluation_groups = groups
    fold_count = task.evaluation_folds if task.group_column == "config_id" else len(groups)
    if len(evaluation_groups) < fold_count:
        raise ValueError("grouped evaluation has fewer groups than folds")
    truth = frame.loc[:, task.label_columns].to_numpy(dtype=np.float32)
    features = frame.loc[:, task.feature_set.columns].to_numpy(dtype=np.float32)
    prediction = np.empty_like(truth)
    evaluated = np.zeros(len(frame), dtype=bool)
    folds: list[dict[str, object]] = []
    fold_groups = _group_folds(evaluation_groups, fold_count, task.seed)
    for offset, heldout in enumerate(fold_groups):
        validation_groups = fold_groups[(offset + 1) % len(fold_groups)]
        test = np.isin(values, heldout)
        valid = np.isin(values, validation_groups)
        train = ~(test | valid)
        evaluated |= test
        fitted = fit_multihead(
            training=task.training, train_x=features[train], train_y=truth[train],
            validation_x=features[valid], validation_y=truth[valid], seed=task.seed + offset,
            label_columns=task.label_columns,
        )
        prediction[test] = predict_multihead(fitted, features[test])
        folds.append({
            "heldout_groups": heldout,
            "validation_groups": validation_groups,
            "train_rows": int(train.sum()), "validation_rows": int(valid.sum()),
            "test_rows": int(test.sum()), "epochs": fitted.epochs,
        })
    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = destination.with_name(f".{destination.name}.partial")
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir()
    oof_path = output / "oof_predictions.parquet"
    evaluated_frame = frame.loc[evaluated]
    evaluated_truth = truth[evaluated]
    evaluated_prediction = prediction[evaluated]
    _write_predictions(
        evaluated_frame,
        task,
        evaluated_truth,
        evaluated_prediction,
        oof_path,
    )
    if task.group_column == "config_id":
        metric_report = {
            "roi_weighted": regression_error_report(task.label_columns, evaluated_truth, evaluated_prediction),
            "per_metric_absolute": regression_error_report(task.label_columns, evaluated_truth, evaluated_prediction),
        }
        primary_metric = _primary_metric("roi_weighted")
    else:
        metric_report = {
            "roi_weighted": regression_error_report(task.label_columns, evaluated_truth, evaluated_prediction),
            "macro_workload": _macro_workload_report(task, values[evaluated], evaluated_truth, evaluated_prediction),
            "per_workload": _per_workload_report(task, values[evaluated], evaluated_truth, evaluated_prediction),
        }
        primary_metric = _primary_metric("macro_workload")
    report = {
        "task_id": task.task_id,
        "protocol": "grouped_config_kfold" if task.group_column == "config_id" else "leave_one_workload_out",
        "generalization_scope": "held_out_configuration" if task.group_column == "config_id" else "joint_program_microarchitecture_ood",
        "primary_metric": primary_metric,
        "samples": int(evaluated.sum()), "groups": len(evaluation_groups),
        "data_limitations": task.data_limitations,
        "metrics": metric_report,
        "outer_folds": folds,
        "oof_predictions": str(destination / oof_path.name),
    }
    evaluation = output / "evaluation.json"
    evaluation.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    previous = destination.with_name(f".{destination.name}.previous")
    if previous.exists():
        raise RuntimeError(f"previous evaluation directory exists: {previous}")
    if destination.exists():
        destination.replace(previous)
    try:
        output.replace(destination)
    except BaseException:
        if previous.exists():
            previous.replace(destination)
        raise
    shutil.rmtree(previous, ignore_errors=True)
    return {
        "evaluation": str(destination / evaluation.name),
        "oof_predictions": str(destination / oof_path.name),
    }


def evaluate_random_roi_split_prediction_task(
    *, task: PredictionTask, dataset_dir: str | Path, output_dir: str | Path,
    workload_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Evaluate the fixed model with the historical random 70/15/15 ROI split."""
    frame = _read_task_frame(task, dataset_dir, workload_ids)
    train, valid, test = _random_roi_masks(len(frame), task.seed)
    truth = frame.loc[:, task.label_columns].to_numpy(dtype=np.float32)
    features = frame.loc[:, task.feature_set.columns].to_numpy(dtype=np.float32)
    prediction = np.empty((int(test.sum()), len(task.output_metrics)), dtype=np.float32)
    epochs: dict[str, int] = {}
    for index, metric in enumerate(task.output_metrics):
        fitted = fit_multihead(
            training=task.training, train_x=features[train], train_y=truth[train, index:index + 1],
            validation_x=features[valid], validation_y=truth[valid, index:index + 1],
            seed=task.seed + index, label_columns=(task.label_columns[index],),
        )
        prediction[:, index] = predict_multihead(fitted, features[test])[:, 0]
        epochs[metric] = fitted.epochs
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "test_predictions.parquet"
    _write_predictions(frame.loc[test], task, truth[test], prediction, predictions_path)
    report = {
        "task_id": task.task_id,
        "protocol": "random_roi_split",
        "generalization_scope": "in_distribution_random_roi",
        "primary_metric": _primary_metric("roi_weighted"),
        "samples": len(frame),
        "split": {
            "seed": task.seed, "train_fraction": 0.70, "validation_fraction": 0.15,
            "test_fraction": 0.15, "train_rows": int(train.sum()),
            "validation_rows": int(valid.sum()), "test_rows": int(test.sum()), "epochs": epochs,
        },
        "metrics": {
            "roi_weighted": regression_error_report(task.label_columns, truth[test], prediction),
        },
        "test_predictions": str(predictions_path),
    }
    evaluation = output / "evaluation.json"
    evaluation.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"evaluation": str(evaluation), "test_predictions": str(predictions_path)}


def _read_task_frame(task, dataset_dir, workload_ids):
    if len(task.label_columns) != len(task.output_metrics):
        raise ValueError("task labels and output metrics differ")
    torch.set_num_threads(task.num_threads)
    columns = tuple(dict.fromkeys((*task.identity_columns, *task.label_columns, *task.feature_set.columns)))
    frame = read_dataset_shards(dataset_dir, columns=columns)
    if workload_ids is not None:
        frame = frame[frame["workload_id"].isin(workload_ids)].copy()
    _validate_task_frame(task, frame)
    return frame


def _group_folds(groups, count, seed):
    shuffled = np.asarray(groups, dtype=object)[np.random.default_rng(seed).permutation(len(groups))]
    return tuple(tuple(str(item) for item in fold) for fold in np.array_split(shuffled, count))


def _candidate_baseline_deltas(*, frame, metric_index, truth, candidate_prediction, candidate_mask, baseline_prediction):
    keys = [name for name in ("workload_id", "window_index") if name in frame]
    baseline = frame.loc[frame.config_id.astype(str) == "baseline", keys].copy()
    baseline["truth_baseline"] = truth[frame.config_id.astype(str).to_numpy() == "baseline", metric_index]
    baseline["prediction_baseline"] = baseline_prediction
    candidate = frame.loc[candidate_mask, keys].copy()
    candidate["truth_candidate"] = truth[candidate_mask, metric_index]
    candidate["prediction_candidate"] = candidate_prediction
    paired = candidate.merge(baseline, on=keys, validate="many_to_one")
    return (
        paired.truth_candidate.to_numpy() - paired.truth_baseline.to_numpy(),
        paired.prediction_candidate.to_numpy() - paired.prediction_baseline.to_numpy(),
    )


def _delta_report(task, truth_parts, prediction_parts):
    report = {}
    if not truth_parts:
        return report
    truth = np.concatenate(truth_parts)
    prediction = np.concatenate(prediction_parts)
    for index, metric in enumerate(task.output_metrics):
        selected = truth[:, 0] == index
        report[metric] = regression_error_report(
            (task.label_columns[index],), truth[selected, 1:2], prediction[selected, 1:2],
        )[metric]
    return report


def _random_roi_masks(rows, seed):
    if rows < 3:
        raise ValueError("random ROI split requires at least three rows")
    order = np.random.default_rng(seed).permutation(rows)
    validation_rows = max(1, int(round(rows * VALIDATION_FRACTION)))
    test_rows = max(1, int(round(rows * VALIDATION_FRACTION)))
    if validation_rows + test_rows >= rows:
        raise ValueError("random ROI split leaves no training rows")
    train = np.zeros(rows, dtype=bool)
    valid = np.zeros(rows, dtype=bool)
    test = np.zeros(rows, dtype=bool)
    valid[order[:validation_rows]] = True
    test[order[validation_rows:validation_rows + test_rows]] = True
    train[order[validation_rows + test_rows:]] = True
    return train, valid, test


def _per_workload_report(task, values, truth, prediction):
    return {
        workload: regression_error_report(task.label_columns, truth[values == workload], prediction[values == workload])
        for workload in sorted(set(values))
    }


def _primary_metric(aggregation):
    return {"target": "CPI", "metric": "mape_pct", "aggregation": aggregation}


def _macro_workload_report(task, values, truth, prediction):
    per_workload = _per_workload_report(task, values, truth, prediction)
    report: dict[str, dict[str, float | int | None]] = {}
    for metric in task.output_metrics:
        reports = [per_workload[workload][metric] for workload in per_workload]
        report[metric] = {
            key: _optional_mean(entry[key] for entry in reports)
            for key in ("mae", "rmse", "mape_pct", "p90_absolute_error", "wape_pct", "smape_pct")
        }
        report[metric]["mape_nonzero_rows"] = int(sum(entry["mape_nonzero_rows"] for entry in reports))
    return report


def _optional_mean(values):
    finite = [float(value) for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def _checkpoint(fitted: FittedMultiHead, features, labels, task, metric):
    return {
        "state_dict": fitted.model.state_dict(), "feature_columns": features,
        "label_columns": labels, "output_metrics": (metric,),
        "hidden_dims": task.training.hidden_dims, "num_threads": task.num_threads,
        "feature_mean": torch.from_numpy(fitted.feature_mean), "feature_scale": torch.from_numpy(fitted.feature_scale),
        "label_mean": torch.from_numpy(fitted.label_mean), "label_scale": torch.from_numpy(fitted.label_scale),
        "target_transforms": fitted.target_transforms,
    }


def _write_predictions(
    frame, task, truth, prediction, path, *, delta_truth=None, delta_prediction=None,
):
    table = pa.Table.from_pandas(frame.loc[:, task.identity_columns], preserve_index=False)
    for index, metric in enumerate(task.output_metrics):
        table = table.append_column(f"truth_{metric}", pa.array(truth[:, index]))
        table = table.append_column(f"prediction_{metric}", pa.array(prediction[:, index]))
        if delta_truth is not None and delta_prediction is not None:
            table = table.append_column(f"truth_delta_{metric}", pa.array(delta_truth[:, index]))
            table = table.append_column(f"prediction_delta_{metric}", pa.array(delta_prediction[:, index]))
    pq.write_table(table, path, compression="zstd")


def _validate_task_frame(task, frame):
    required = {*task.identity_columns, task.group_column, *task.label_columns, *task.feature_set.columns}
    if frame.empty or not required <= set(frame) or frame.duplicated(list(task.identity_columns)).any():
        raise ValueError("prediction task dataset has invalid identities or columns")
    numeric = [*task.label_columns, *task.feature_set.columns]
    if not np.isfinite(frame.loc[:, list(dict.fromkeys(numeric))].to_numpy(dtype=np.float64)).all():
        raise ValueError("prediction task dataset contains non-finite values")
    if (frame.loc[:, task.label_columns].nunique(dropna=False) <= 1).any():
        raise ValueError("prediction task dataset has zero-variance labels")
