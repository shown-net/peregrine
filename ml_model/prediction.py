"""Grouped evaluation and deployment training for the canonical surrogate."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanMetric

from .dataset_io import read_dataset_shards
from .error_metrics import compute_regression_metrics
from .module import SurrogateModule, fit_normalization
from .model import CONSTANT_ZERO_HEAD, POSITIVE_HEAD, ZERO_INFLATED_HEAD
from .tasks import SurrogateTask, TargetSpec


def evaluate_surrogate(
    *, task: SurrogateTask, dataset_dir: str | Path, output_dir: str | Path,
    workload_ids: tuple[str, ...] | None = None,
) -> dict[str, str]:
    frame = _read_frame(task, dataset_dir, workload_ids)
    groups = frame[task.group_column].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    folds = min(task.evaluation_folds, len(unique_groups))
    if folds < 3:
        raise ValueError("surrogate grouped evaluation requires at least three workloads")
    features = _feature_matrix(task, frame)
    raw_labels = frame.loc[:, task.label_columns].to_numpy(dtype=np.float32)
    labels = raw_labels
    prediction = np.empty_like(labels)
    reports: list[dict[str, object]] = []
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=task.seed)
    for fold_index, (train_valid, test) in enumerate(splitter.split(features, labels, groups)):
        train, valid = _train_validation_split(task, train_valid, groups)
        module, epoch = _fit_partition(
            task, features[train], labels[train], features[valid], labels[valid],
            Path(output_dir) / ".folds" / str(fold_index),
        )
        prediction[test] = _predict_array(module, features[test], task.batch_size, task.feature_columns)
        reports.append({
            "heldout_groups": sorted(set(groups[test])),
            "train_rows": int(len(train)), "validation_rows": int(len(valid)), "test_rows": int(len(test)),
            "epochs": epoch,
            "model_parameters": sum(parameter.numel() for parameter in module.parameters()),
        })
    return _publish_evaluation(task, frame, raw_labels, prediction, reports, output_dir)


def evaluate_random_pair_surrogate(
    *, task: SurrogateTask, dataset_dir: str | Path, output_dir: str | Path,
    workload_ids: tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Evaluate in-distribution random ROI/configuration pairs with a fixed split."""
    frame = _read_frame(task, dataset_dir, workload_ids)
    features = _feature_matrix(task, frame)
    raw_labels = frame.loc[:, task.label_columns].to_numpy(dtype=np.float32)
    labels = raw_labels
    train, valid, test = _random_pair_split(len(frame), task.seed)
    module, epochs = _fit_partition(
        task, features[train], labels[train], features[valid], labels[valid],
        Path(output_dir) / ".selection",
    )
    prediction = _predict_array(module, features[test], task.batch_size, task.feature_columns)
    return _publish_random_pair_evaluation(
        task, frame.iloc[test].copy(), raw_labels[test], prediction, output_dir,
        train_rows=len(train), validation_rows=len(valid), epochs=epochs,
    )


def train_surrogate(
    *, task: SurrogateTask, dataset_dir: str | Path, output_dir: str | Path,
    workload_ids: tuple[str, ...] | None = None,
) -> dict[str, str]:
    frame = _read_frame(task, dataset_dir, workload_ids)
    features = _feature_matrix(task, frame)
    raw_labels = frame.loc[:, task.label_columns].to_numpy(dtype=np.float32)
    labels = raw_labels
    groups = frame[task.group_column].astype(str).to_numpy()
    train, valid = _train_validation_split(task, np.arange(len(frame)), groups)
    selection_root = Path(output_dir) / ".selection"
    _selected, epochs = _fit_partition(
        task, features[train], labels[train], features[valid], labels[valid], selection_root,
    )
    shutil.rmtree(selection_root, ignore_errors=True)
    L.seed_everything(task.seed, workers=True)
    task, features, _ = _fit_task(task, features, labels, features)
    normalization = fit_normalization(features, labels, task.targets)
    module = _module(task, normalization)
    trainer = _trainer(task, max_epochs=epochs, callbacks=[])
    trainer.fit(module, train_dataloaders=_loader(features, labels, task.batch_size, shuffle=True))
    destination = Path(output_dir) / task.checkpoint_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".ckpt.partial")
    trainer.save_checkpoint(partial, weights_only=False)
    partial.replace(destination)
    return {"checkpoint": str(destination)}


def compare_oof_evaluations(
    *, task: SurrogateTask, current_oof_path: str | Path, historical_oof_path: str | Path,
    output_path: str | Path,
) -> dict[str, str]:
    """Recompute both versions with the canonical metrics before comparison."""
    current = _read_oof(task, current_oof_path)
    historical = _read_oof(task, historical_oof_path)
    identities = list(task.identity_columns)
    if not current.loc[:, identities].equals(historical.loc[:, identities]):
        raise ValueError("historical and current OOF predictions have different identities")
    if not np.array_equal(_oof_matrix(current, task, "truth"), _oof_matrix(historical, task, "truth")):
        raise ValueError("historical and current OOF predictions have different labels")
    current_report = _metric_report(
        task, current.loc[:, identities], _oof_matrix(current, task, "truth"), _oof_matrix(current, task, "prediction"),
    )
    historical_report = _metric_report(
        task, historical.loc[:, identities], _oof_matrix(historical, task, "truth"), _oof_matrix(historical, task, "prediction"),
    )
    payload = {
        "comparison_scope": "same_dataset_grouped_workload_oof_nonpaired_folds",
        "interpretation": (
            "Dataset identities are identical, but outer folds differ; comparison is directional. "
            "The historical model included per-target adapter selection and isotonic calibration."
        ),
        "samples": int(len(current)),
        "workload_groups": int(current[task.group_column].nunique()),
        "metrics": _compare_metric_reports(historical_report, current_report),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(destination)
    return {"comparison": str(destination)}


def _fit_partition(
    task: SurrogateTask, train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray,
    valid_y: np.ndarray, root: Path,
) -> tuple[SurrogateModule, int]:
    L.seed_everything(task.seed, workers=True)
    fitted_task, train_x, valid_x = _fit_task(task, train_x, train_y, valid_x)
    normalization = fit_normalization(train_x, train_y, fitted_task.targets)
    module = _module(fitted_task, normalization)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=root, filename="best", monitor="val/loss", mode="min", save_top_k=1,
        auto_insert_metric_name=False,
    )
    stopping = EarlyStopping(monitor="val/loss", mode="min", patience=task.early_stopping_patience)
    trainer = _trainer(fitted_task, max_epochs=fitted_task.max_epochs, callbacks=[checkpoint, stopping])
    trainer.fit(
        module,
        train_dataloaders=_loader(train_x, train_y, fitted_task.batch_size, shuffle=True),
        val_dataloaders=_loader(valid_x, valid_y, fitted_task.batch_size, shuffle=False),
    )
    if not checkpoint.best_model_path:
        raise RuntimeError("surrogate training produced no checkpoint")
    fitted = SurrogateModule.load_from_checkpoint(checkpoint.best_model_path, map_location="cpu")
    return fitted, int(trainer.current_epoch + 1)


def _module(task: SurrogateTask, normalization: dict[str, np.ndarray]) -> SurrogateModule:
    return SurrogateModule(
        feature_columns=task.feature_columns, targets=task.targets, hidden_dims=task.hidden_dims,
        learning_rate=task.learning_rate, weight_decay=task.weight_decay,
        dropout=task.dropout,
        **normalization,
    )


def _fit_task(task: SurrogateTask, train_x: np.ndarray, train_y: np.ndarray,
              valid_x: np.ndarray) -> tuple[SurrogateTask, np.ndarray, np.ndarray]:
    targets = tuple(TargetSpec(
        spec.metric,
        spec.label_column,
        spec.head_kind or (
            CONSTANT_ZERO_HEAD if np.all(train_y[:, index] == 0.0)
            else ZERO_INFLATED_HEAD if np.any(train_y[:, index] == 0.0)
            else POSITIVE_HEAD
        ),
        spec.primary_metric,
    ) for index, spec in enumerate(task.targets))
    selected = np.flatnonzero(np.std(train_x, axis=0) > 0.0)
    if not len(selected):
        raise ValueError("training fold has no varying raw statistics")
    columns = tuple(task.feature_columns[index] for index in selected)
    from dataclasses import replace
    return replace(task, feature_columns=columns, targets=targets), train_x[:, selected], valid_x[:, selected]


def _trainer(task: SurrogateTask, *, max_epochs: int, callbacks: list[object]) -> L.Trainer:
    torch.set_num_threads(task.num_threads)
    return L.Trainer(
        accelerator="cpu", devices=1, max_epochs=max_epochs, logger=False,
        enable_progress_bar=False, enable_model_summary=False, callbacks=callbacks,
        enable_checkpointing=bool(callbacks),
        deterministic=True, num_sanity_val_steps=0,
    )


def _loader(features: np.ndarray, labels: np.ndarray, batch_size: int, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(features)), torch.from_numpy(np.ascontiguousarray(labels))),
        batch_size=batch_size, shuffle=shuffle,
    )


def _predict_array(module: SurrogateModule, features: np.ndarray, batch_size: int,
                   source_columns: tuple[str, ...] | None = None) -> np.ndarray:
    if source_columns is not None:
        indices = tuple(source_columns.index(column) for column in module.feature_columns)
        features = features[:, indices]
    module.eval()
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in DataLoader(torch.from_numpy(np.ascontiguousarray(features)), batch_size=batch_size):
            predicted = module(batch).cpu().numpy()
            if not np.isfinite(predicted).all():
                raise ValueError("surrogate prediction produced non-finite values")
            values.append(predicted)
    return np.concatenate(values, axis=0).astype(np.float32, copy=False)


def _train_validation_split(task: SurrogateTask, indices: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected_groups = groups[indices]
    if len(set(selected_groups)) < 2:
        raise ValueError("surrogate training requires at least two workload groups")
    splitter = GroupShuffleSplit(n_splits=1, test_size=task.validation_fraction, random_state=task.seed)
    train_local, valid_local = next(splitter.split(indices, groups=selected_groups))
    return indices[train_local], indices[valid_local]


def _random_pair_split(rows: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rows < 3:
        raise ValueError("random-pair evaluation requires at least three rows")
    holdout = int(rows * 0.15)
    if holdout < 1 or rows - 2 * holdout < 1:
        raise ValueError("random-pair evaluation cannot allocate 70/15/15 rows")
    permutation = np.random.default_rng(seed).permutation(rows)
    test = permutation[:holdout]
    valid = permutation[holdout:2 * holdout]
    train = permutation[2 * holdout:]
    return train, valid, test


def _read_frame(task: SurrogateTask, dataset_dir: str | Path, workload_ids: tuple[str, ...] | None) -> pd.DataFrame:
    columns = tuple(dict.fromkeys((*task.identity_columns, *task.feature_columns, *task.label_columns)))
    frame = read_dataset_shards(dataset_dir, columns=columns)
    if workload_ids is not None:
        frame = frame[frame["workload_id"].isin(workload_ids)].copy()
    required = set(columns)
    if frame.empty or not required <= set(frame) or frame.duplicated(list(task.identity_columns)).any():
        raise ValueError("surrogate dataset has invalid identities or columns")
    numeric = np.concatenate((_feature_matrix(task, frame), frame.loc[:, task.label_columns].to_numpy(dtype=np.float32)), axis=1)
    if not np.isfinite(numeric).all():
        raise ValueError("surrogate dataset contains non-finite values")
    return frame


def _feature_matrix(task: SurrogateTask, frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, task.feature_columns].to_numpy(dtype=np.float32)


def _read_oof(task: SurrogateTask, path: str | Path) -> pd.DataFrame:
    frame = pq.read_table(path).to_pandas()
    required = [
        *task.identity_columns,
        *(f"{kind}_{metric}" for kind in ("truth", "prediction") for metric in task.output_metrics),
    ]
    if frame.empty or not set(required) <= set(frame) or frame.duplicated(list(task.identity_columns)).any():
        raise ValueError("OOF predictions have invalid identities or columns")
    values = frame.loc[:, required[len(task.identity_columns):]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("OOF predictions contain non-finite values")
    return frame.loc[:, required].copy()


def _oof_matrix(frame: pd.DataFrame, task: SurrogateTask, kind: str) -> np.ndarray:
    return frame.loc[:, [f"{kind}_{metric}" for metric in task.output_metrics]].to_numpy(dtype=np.float32)


def _publish_evaluation(
    task: SurrogateTask, frame: pd.DataFrame, labels: np.ndarray, prediction: np.ndarray,
    folds: list[dict[str, object]], output_dir: str | Path,
) -> dict[str, str]:
    destination = Path(output_dir)
    partial = destination.with_name(f".{destination.name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    table = pa.Table.from_pandas(frame.loc[:, task.identity_columns], preserve_index=False)
    for index, metric in enumerate(task.output_metrics):
        table = table.append_column(f"truth_{metric}", pa.array(labels[:, index]))
        table = table.append_column(f"prediction_{metric}", pa.array(prediction[:, index]))
    oof = partial / "oof_predictions.parquet"
    pq.write_table(table, oof, compression="zstd")
    payload = {
        "protocol": "grouped_workload_kfold",
        "metrics": _metric_report(task, frame, labels, prediction),
        "target_support": _target_support(task, labels, prediction),
        "model_parameters": int(max(int(fold["model_parameters"]) for fold in folds)),
        "outer_folds": folds, "samples": int(len(frame)), "groups": int(frame[task.group_column].nunique()),
    }
    report = partial / "evaluation.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if destination.exists():
        shutil.rmtree(destination)
    partial.replace(destination)
    return {"evaluation": str(destination / report.name), "oof_predictions": str(destination / oof.name)}


def _publish_random_pair_evaluation(
    task: SurrogateTask, frame: pd.DataFrame, labels: np.ndarray, prediction: np.ndarray,
    output_dir: str | Path, *, train_rows: int, validation_rows: int, epochs: int,
) -> dict[str, str]:
    destination = Path(output_dir)
    partial = destination.with_name(f".{destination.name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    table = pa.Table.from_pandas(frame.loc[:, task.identity_columns], preserve_index=False)
    for index, metric in enumerate(task.output_metrics):
        table = table.append_column(f"truth_{metric}", pa.array(labels[:, index]))
        table = table.append_column(f"prediction_{metric}", pa.array(prediction[:, index]))
    predictions = partial / "test_predictions.parquet"
    pq.write_table(table, predictions, compression="zstd")
    payload = {
        "protocol": "random_pair_split",
        "generalization_scope": "in_distribution_random_pair",
        "metrics": _metric_report(task, frame, labels, prediction),
        "target_support": _target_support(task, labels, prediction),
        "samples": train_rows + validation_rows + len(frame),
        "split": {
            "seed": task.seed,
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "test_rows": int(len(frame)),
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "epochs": epochs,
        },
    }
    report = partial / "evaluation.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if destination.exists():
        shutil.rmtree(destination)
    partial.replace(destination)
    return {"evaluation": str(destination / report.name), "test_predictions": str(destination / predictions.name)}


def _metric_report(
    task: SurrogateTask, frame: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray,
) -> dict[str, object]:
    per_workload: dict[str, dict[str, dict[str, float]]] = {}
    for workload, indices in frame.groupby(task.group_column, sort=True).indices.items():
        per_workload[str(workload)] = _metrics(task, truth[indices], prediction[indices])
    return {
        "macro_workload": _macro_metrics(per_workload),
        "roi_weighted": _metrics(task, truth, prediction),
        "per_workload": per_workload,
    }


def _metrics(task: SurrogateTask, truth: np.ndarray, prediction: np.ndarray) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for index, spec in enumerate(task.targets):
        report[spec.metric] = compute_regression_metrics(
            spec.metric,
            torch.from_numpy(prediction[:, index].copy()),
            torch.from_numpy(truth[:, index].copy()),
        )
    return report


def _target_support(
    task: SurrogateTask, truth: np.ndarray, prediction: np.ndarray,
) -> dict[str, dict[str, float | str]]:
    report: dict[str, dict[str, float | str]] = {}
    for index, spec in enumerate(task.targets):
        actual = truth[:, index].astype(np.float64, copy=False)
        estimated = prediction[:, index].astype(np.float64, copy=False)
        positive = actual > 0.0
        prevalence = float(np.mean(positive))
        values: dict[str, float | str] = {"nonzero_rate": prevalence}
        if not positive.any():
            values["support_kind"] = CONSTANT_ZERO_HEAD
            values["false_positive_rate"] = float(np.mean(estimated > 0.0))
        elif positive.all():
            values["support_kind"] = POSITIVE_HEAD
        else:
            values["support_kind"] = ZERO_INFLATED_HEAD
            values["nonzero_average_precision"] = float(
                average_precision_score(positive.astype(np.int8), estimated)
            )
            values["positive_wmape_pct"] = float(
                100.0 * np.sum(np.abs(estimated[positive] - actual[positive]))
                / np.sum(np.abs(actual[positive]))
            )
        report[spec.metric] = values
    return report


def _macro_metrics(
    per_workload: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, MeanMetric]] = {}
    for workload in per_workload.values():
        for target, values in workload.items():
            target_metrics = metrics.setdefault(target, {})
            for name, value in values.items():
                target_metrics.setdefault(name, MeanMetric()).update(torch.tensor(value))
    return {
        target: {name: float(metric.compute().item()) for name, metric in values.items()}
        for target, values in metrics.items()
    }


def _compare_metric_reports(historical: dict[str, object], current: dict[str, object]) -> dict[str, object]:
    comparison: dict[str, object] = {}
    for aggregation in ("macro_workload", "roi_weighted", "per_workload"):
        previous = historical[aggregation]
        present = current[aggregation]
        assert isinstance(previous, dict) and isinstance(present, dict)
        aggregation_report: dict[str, object] = {}
        comparison[aggregation] = aggregation_report
        if aggregation == "per_workload":
            for workload, previous_workload in previous.items():
                current_workload = present[workload]
                assert isinstance(previous_workload, dict) and isinstance(current_workload, dict)
                aggregation_report[workload] = _compare_targets(previous_workload, current_workload)
            continue
        aggregation_report.update(_compare_targets(previous, present))
    return comparison


def _compare_targets(previous: dict[str, object], present: dict[str, object]) -> dict[str, object]:
    target_report: dict[str, object] = {}
    for target, previous_values in previous.items():
        current_values = present[target]
        assert isinstance(previous_values, dict) and isinstance(current_values, dict)
        metric_report: dict[str, dict[str, float | None]] = {}
        target_report[target] = metric_report
        for name, old_value in previous_values.items():
            new_value = current_values[name]
            assert isinstance(old_value, float) and isinstance(new_value, float)
            delta = new_value - old_value
            metric_report[name] = {
                "historical": old_value,
                "current": new_value,
                "delta_current_minus_historical": delta,
                "relative_change_pct": delta / abs(old_value) * 100.0 if old_value else None,
            }
    return target_report
