from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from anamol.python.design_space import PeregrineConfig
from .dataset_io import read_dataset_shards
from .inference import IDENTITY_COLUMNS
from .model import MultiHeadPeregrineModel


def train_surrogate(
    *, config: PeregrineConfig, dataset_dir: str | Path, output_dir: str | Path,
    workload_ids: tuple[str, ...] | None = None,
    feature_columns: tuple[str, ...] | None = None,
    label_columns: tuple[str, ...] | None = None,
    output_metrics: tuple[str, ...] | None = None,
) -> dict[str, object]:
    features = feature_columns or tuple(config.feature_columns)
    labels = label_columns or tuple(config.label_columns)
    metrics = output_metrics or tuple(label.removeprefix("label_") for label in labels)
    if len(metrics) != len(labels):
        raise ValueError("output metrics must match label columns")
    frame = read_dataset_shards(
        dataset_dir, columns=[*IDENTITY_COLUMNS, *features, *labels]
    )
    if workload_ids is not None:
        frame = frame[frame.workload_id.isin(workload_ids)].copy()
    _validate(frame, features, labels)
    if config.training.num_threads is not None:
        torch.set_num_threads(config.training.num_threads)
    x = frame.loc[:, features].to_numpy(dtype=np.float32)
    y = frame.loc[:, labels].to_numpy(dtype=np.float32)
    split = _split(len(frame), config.training.seed, config.training.paper_test_fraction)
    model, x_mean, x_scale, y_mean, y_scale, epochs = _fit(
        config, x[split.train], y[split.train], x[split.test], y[split.test], config.training.seed, labels
    )
    test_prediction = _predict(model, x[split.test], x_mean, x_scale, y_mean, y_scale)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    torch.save({
        "state_dict": model.state_dict(), "feature_columns": features,
        "label_columns": labels, "output_metrics": metrics,
        "hidden_dims": tuple(config.training.hidden_dims),
        "num_threads": int(torch.get_num_threads()),
        "feature_mean": torch.from_numpy(x_mean),
        "feature_scale": torch.from_numpy(x_scale),
        "label_mean": torch.from_numpy(y_mean),
        "label_scale": torch.from_numpy(y_scale),
    }, checkpoint)
    return {
        "checkpoint": str(checkpoint),
        "random_split": {
            "seed": config.training.seed, "train_rows": int(split.train.sum()),
            "test_rows": int(split.test.sum()), "epochs": epochs,
            **_error_metrics(labels, y[split.test], test_prediction),
        },
    }


def evaluate_workload_ood(
    *, config: PeregrineConfig, dataset_dir: str | Path,
    workload_ids: tuple[str, ...] | None = None,
    feature_columns: tuple[str, ...] | None = None,
    label_columns: tuple[str, ...] | None = None,
) -> dict[str, object]:
    features = feature_columns or tuple(config.feature_columns)
    labels = label_columns or tuple(config.label_columns)
    frame = read_dataset_shards(
        dataset_dir, columns=[*IDENTITY_COLUMNS, *features, *labels]
    )
    if workload_ids is not None:
        frame = frame[frame.workload_id.isin(workload_ids)].copy()
    _validate(frame, features, labels)
    if config.training.num_threads is not None:
        torch.set_num_threads(config.training.num_threads)
    x = frame.loc[:, features].to_numpy(dtype=np.float32)
    y = frame.loc[:, labels].to_numpy(dtype=np.float32)
    return {
        "workload_ood": _workload_ood(config, frame, x, y, labels),
    }


class _Split:
    def __init__(self, train: np.ndarray, test: np.ndarray) -> None:
        self.train, self.test = train, test


def _split(rows: int, seed: int, test_fraction: float) -> _Split:
    order = np.random.default_rng(seed).permutation(rows)
    test_rows = max(1, int(round(rows * test_fraction)))
    test = np.zeros(rows, dtype=bool)
    test[order[:test_rows]] = True
    return _Split(~test, test)


def _fit(config: PeregrineConfig, train_x: np.ndarray, train_y: np.ndarray,
         validation_x: np.ndarray, validation_y: np.ndarray, seed: int,
         label_columns: tuple[str, ...]):
    x_mean, x_scale = _standardize(train_x)
    y_mean, y_scale = _standardize(train_y)
    train = TensorDataset(torch.from_numpy(_scale(train_x, x_mean, x_scale)), torch.from_numpy(_scale(train_y, y_mean, y_scale)))
    loader = DataLoader(train, batch_size=config.training.batch_size, shuffle=True)
    torch.manual_seed(seed)
    model = MultiHeadPeregrineModel(train_x.shape[1], config.training.hidden_dims, label_columns)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    validation_x_tensor = torch.from_numpy(_scale(validation_x, x_mean, x_scale))
    validation_y_tensor = torch.from_numpy(_scale(validation_y, y_mean, y_scale))
    best_state, best_loss, stale, best_epoch = None, float("inf"), 0, 0
    for epoch in range(1, config.training.max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            F.l1_loss(model(batch_x), batch_y).backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            loss = float(F.l1_loss(model(validation_x_tensor), validation_y_tensor))
        if loss < best_loss:
            best_state, best_loss, stale, best_epoch = copy.deepcopy(model.state_dict()), loss, 0, epoch
        else:
            stale += 1
            if stale >= config.training.early_stopping_patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, x_mean, x_scale, y_mean, y_scale, best_epoch


def _workload_ood(config, frame, x, y, label_columns: tuple[str, ...]) -> tuple[dict[str, object], ...]:
    reports = []
    for offset, workload in enumerate(sorted(frame.workload_id.astype(str).unique())):
        heldout = frame.workload_id.astype(str).to_numpy() == workload
        if heldout.all() or (~heldout).sum() < 2:
            continue
        training = np.flatnonzero(~heldout)
        validation = _split(len(training), config.training.seed + offset + 1, config.training.paper_test_fraction)
        model, x_mean, x_scale, y_mean, y_scale, epochs = _fit(
            config, x[training[validation.train]], y[training[validation.train]],
            x[training[validation.test]], y[training[validation.test]],
            config.training.seed + offset + 1,
            label_columns,
        )
        reports.append({
            "heldout_workload": workload,
            "rows": int(heldout.sum()),
            "epochs": epochs,
            **_error_metrics(
                label_columns,
                y[heldout],
                _predict(model, x[heldout], x_mean, x_scale, y_mean, y_scale),
            ),
        })
    return tuple(reports)


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(scale, np.float32(1e-6))


def _scale(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((values - mean) / scale, dtype=np.float32)


def _predict(model, x, x_mean, x_scale, y_mean, y_scale) -> np.ndarray:
    with torch.inference_mode():
        return (model(torch.from_numpy(_scale(x, x_mean, x_scale))).numpy() * y_scale + y_mean).astype(np.float32)


def _error_metrics(
    label_columns: tuple[str, ...],
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, dict[str, float]]:
    absolute = np.abs(truth - prediction).astype(np.float64, copy=False)
    denominator = np.abs(truth).sum(axis=0, dtype=np.float64)
    smape_denominator = np.abs(truth) + np.abs(prediction)
    smape = np.divide(
        2.0 * absolute,
        smape_denominator,
        out=np.zeros_like(absolute, dtype=np.float64),
        where=smape_denominator != 0.0,
    ).mean(axis=0)
    values = {
        "mae": absolute.mean(axis=0),
        "rmse": np.sqrt(np.square(truth - prediction, dtype=np.float64).mean(axis=0)),
        "wape": absolute.sum(axis=0, dtype=np.float64) / denominator,
        "smape": smape,
    }
    return {
        name: {
            label.removeprefix("label_"): float(value)
            for label, value in zip(label_columns, metric_values, strict=True)
        }
        for name, metric_values in values.items()
    }


def _validate(frame, features, labels) -> None:
    if frame.empty or frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("dataset must be non-empty with unique sample identities")
    if not np.isfinite(frame.loc[:, [*features, *labels]].to_numpy(dtype=np.float64)).all():
        raise ValueError("dataset contains non-finite numeric values")
    label_values = frame.loc[:, labels].to_numpy(dtype=np.float64)
    constant = tuple(
        name
        for name, spread in zip(labels, np.ptp(label_values, axis=0), strict=True)
        if spread == 0.0
    )
    if constant:
        raise ValueError(f"dataset contains zero-variance labels: {constant}")
