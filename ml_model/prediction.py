"""Canonical fixed-model training and workload OOD evaluation."""

from __future__ import annotations

import json
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
    output.mkdir(parents=True, exist_ok=True)
    models = output / "models"
    models.mkdir(exist_ok=True)
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
        paths[candidate_id] = str(path.relative_to(output))
    report = {
        "task_id": task.task_id,
        "protocol": "full_dataset_deployment_training",
        "samples": len(frame),
        "workloads": int(frame[task.group_column].nunique()),
        "selected": selected,
    }
    training_report = output / "training_report.json"
    training_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bundle = output / "predictor_bundle.json"
    bundle.write_text(json.dumps({
        "task_id": task.task_id, "identity_columns": task.identity_columns,
        "output_metrics": task.output_metrics, "selected": selected, "model_paths": paths,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"bundle": str(bundle), "training_report": str(training_report)}


def evaluate_prediction_task(
    *, task: PredictionTask, dataset_dir: str | Path, output_dir: str | Path,
    workload_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Evaluate fixed scalar models with leave-one-workload-out isolation."""
    frame = _read_task_frame(task, dataset_dir, workload_ids)
    values = frame[task.group_column].astype(str).to_numpy()
    groups = tuple(sorted(set(values)))
    if len(groups) < 3:
        raise ValueError("workload OOD requires at least three groups")
    truth = frame.loc[:, task.label_columns].to_numpy(dtype=np.float32)
    features = frame.loc[:, task.feature_set.columns].to_numpy(dtype=np.float32)
    prediction = np.empty_like(truth)
    folds: list[dict[str, object]] = []
    for offset, heldout in enumerate(groups):
        test = values == heldout
        valid = _validation_mask(values, test, task.seed + offset)
        train = ~(test | valid)
        metric_epochs: dict[str, int] = {}
        for index, metric in enumerate(task.output_metrics):
            fitted = fit_multihead(
                training=task.training, train_x=features[train], train_y=truth[train, index:index + 1],
                validation_x=features[valid], validation_y=truth[valid, index:index + 1],
                seed=task.seed + offset * len(task.output_metrics) + index,
                label_columns=(task.label_columns[index],),
            )
            prediction[test, index] = predict_multihead(fitted, features[test])[:, 0]
            metric_epochs[metric] = fitted.epochs
        folds.append({
            "heldout_workload": heldout,
            "training_workloads": tuple(group for group in groups if group != heldout),
            "train_rows": int(train.sum()), "validation_rows": int(valid.sum()),
            "test_rows": int(test.sum()), "epochs": metric_epochs,
        })
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    oof_path = output / "oof_predictions.parquet"
    _write_predictions(frame, task, truth, prediction, oof_path)
    report = {
        "task_id": task.task_id,
        "protocol": "leave_one_workload_out",
        "generalization_scope": "joint_program_microarchitecture_ood",
        "primary_metric": _primary_metric("macro_workload"),
        "samples": len(frame), "workloads": len(groups),
        "data_limitations": task.data_limitations,
        "metrics": {
            "roi_weighted": regression_error_report(task.label_columns, truth, prediction),
            "macro_workload": _macro_workload_report(task, values, truth, prediction),
            "per_workload": _per_workload_report(task, values, truth, prediction),
        },
        "outer_folds": folds,
        "oof_predictions": str(oof_path),
    }
    evaluation = output / "evaluation.json"
    evaluation.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"evaluation": str(evaluation), "oof_predictions": str(oof_path)}


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
        frame = frame[frame[task.group_column].isin(workload_ids)].copy()
    _validate_task_frame(task, frame)
    return frame


def _validation_mask(values, test, seed):
    valid = np.zeros(len(values), dtype=bool)
    for offset, workload in enumerate(sorted(set(values[~test]))):
        rows = np.flatnonzero((values == workload) & ~test)
        count = max(1, int(round(len(rows) * VALIDATION_FRACTION)))
        rng = np.random.default_rng(seed + offset)
        valid[rng.permutation(rows)[:count]] = True
    return valid


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
    report: dict[str, dict[str, float | int]] = {}
    for metric in task.output_metrics:
        reports = [per_workload[workload][metric] for workload in per_workload]
        report[metric] = {
            key: float(np.mean([entry[key] for entry in reports]))
            for key in ("mae", "rmse", "mape_pct", "p90_absolute_error", "wape_pct", "smape_pct")
        }
        report[metric]["mape_nonzero_rows"] = int(sum(entry["mape_nonzero_rows"] for entry in reports))
    return report


def _checkpoint(fitted: FittedMultiHead, features, labels, task, metric):
    return {
        "state_dict": fitted.model.state_dict(), "feature_columns": features,
        "label_columns": labels, "output_metrics": (metric,),
        "hidden_dims": task.training.hidden_dims, "num_threads": task.num_threads,
        "feature_mean": torch.from_numpy(fitted.feature_mean), "feature_scale": torch.from_numpy(fitted.feature_scale),
        "label_mean": torch.from_numpy(fitted.label_mean), "label_scale": torch.from_numpy(fitted.label_scale),
        "target_transforms": fitted.target_transforms,
    }


def _write_predictions(frame, task, truth, prediction, path):
    table = pa.Table.from_pandas(frame.loc[:, task.identity_columns], preserve_index=False)
    for index, metric in enumerate(task.output_metrics):
        table = table.append_column(f"truth_{metric}", pa.array(truth[:, index]))
        table = table.append_column(f"prediction_{metric}", pa.array(prediction[:, index]))
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
