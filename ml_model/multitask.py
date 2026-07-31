"""Domain-neutral training primitives for Peregrine multi-output regressors."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .model import MultiHeadPeregrineModel


IDENTITY_TARGET_TRANSFORM = "identity"
LOG1P_NONNEGATIVE_TARGET_TRANSFORM = "log1p_nonnegative"
SUPPORTED_TARGET_TRANSFORMS = frozenset(
    (IDENTITY_TARGET_TRANSFORM, LOG1P_NONNEGATIVE_TARGET_TRANSFORM)
)
FEATURE_Z_LIMIT = np.float32(8.0)


@dataclass(frozen=True)
class MultiHeadTraining:
    hidden_dims: tuple[int, int]
    max_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int


@dataclass(frozen=True)
class FittedMultiHead:
    model: MultiHeadPeregrineModel
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    label_mean: np.ndarray
    label_scale: np.ndarray
    target_transforms: tuple[str, ...]
    epochs: int


def fit_multihead(
    *,
    training: MultiHeadTraining,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    seed: int,
    label_columns: Iterable[str],
    target_transforms: Iterable[str] | None = None,
) -> FittedMultiHead:
    labels = tuple(label_columns)
    transforms = normalize_target_transforms(labels, target_transforms)
    feature_mean, feature_scale = standardize(train_x)
    transformed_train_y = transform_targets(train_y, transforms)
    transformed_validation_y = transform_targets(validation_y, transforms)
    label_mean, label_scale = standardize(transformed_train_y)
    train = TensorDataset(
        torch.from_numpy(scale(train_x, feature_mean, feature_scale)),
        torch.from_numpy(scale(transformed_train_y, label_mean, label_scale)),
    )
    loader = DataLoader(train, batch_size=training.batch_size, shuffle=True)
    torch.manual_seed(seed)
    model = MultiHeadPeregrineModel(train_x.shape[1], training.hidden_dims, labels)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    validation_x_tensor = torch.from_numpy(scale(validation_x, feature_mean, feature_scale))
    validation_y_tensor = torch.from_numpy(scale(transformed_validation_y, label_mean, label_scale))
    best_state, best_loss, stale, best_epoch = None, float("inf"), 0, 0
    for epoch in range(1, training.max_epochs + 1):
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
            if stale >= training.early_stopping_patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return FittedMultiHead(
        model, feature_mean, feature_scale, label_mean, label_scale, transforms, best_epoch
    )


def predict_multihead(fitted: FittedMultiHead, features: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        learned = fitted.model(torch.from_numpy(scale(features, fitted.feature_mean, fitted.feature_scale))).numpy()
    transformed = (learned * fitted.label_scale + fitted.label_mean).astype(np.float32)
    return inverse_transform_targets(transformed, fitted.target_transforms)


def fit_multihead_epochs(
    *, training: MultiHeadTraining, features: np.ndarray, labels: np.ndarray,
    epochs: int, seed: int, label_columns: Iterable[str],
    target_transforms: Iterable[str] | None = None,
) -> FittedMultiHead:
    """Fit a deployable model on all rows for an epoch count selected out of fold."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    label_names = tuple(label_columns)
    transforms = normalize_target_transforms(label_names, target_transforms)
    feature_mean, feature_scale = standardize(features)
    transformed_labels = transform_targets(labels, transforms)
    label_mean, label_scale = standardize(transformed_labels)
    rows = TensorDataset(
        torch.from_numpy(scale(features, feature_mean, feature_scale)),
        torch.from_numpy(scale(transformed_labels, label_mean, label_scale)),
    )
    loader = DataLoader(rows, batch_size=training.batch_size, shuffle=True)
    torch.manual_seed(seed)
    model = MultiHeadPeregrineModel(features.shape[1], training.hidden_dims, label_names)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            F.l1_loss(model(batch_x), batch_y).backward()
            optimizer.step()
    model.eval()
    return FittedMultiHead(model, feature_mean, feature_scale, label_mean, label_scale, transforms, epochs)


def normalize_target_transforms(
    label_columns: Iterable[str],
    target_transforms: Iterable[str] | None,
) -> tuple[str, ...]:
    labels = tuple(label_columns)
    transforms = (
        (IDENTITY_TARGET_TRANSFORM,) * len(labels)
        if target_transforms is None
        else tuple(str(item) for item in target_transforms)
    )
    if len(transforms) != len(labels):
        raise ValueError("target transforms must match label columns")
    unknown = sorted(set(transforms) - SUPPORTED_TARGET_TRANSFORMS)
    if unknown:
        raise ValueError(f"unknown target transforms: {unknown}")
    return transforms


def transform_targets(values: np.ndarray, target_transforms: Iterable[str]) -> np.ndarray:
    transforms = tuple(target_transforms)
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(transforms) or not np.isfinite(matrix).all():
        raise ValueError("invalid target matrix")
    transformed = matrix.copy()
    for index, transform in enumerate(transforms):
        if transform == LOG1P_NONNEGATIVE_TARGET_TRANSFORM:
            if np.any(transformed[:, index] < 0.0):
                raise ValueError("log1p_nonnegative target transform received negative labels")
            transformed[:, index] = np.log1p(transformed[:, index])
        elif transform != IDENTITY_TARGET_TRANSFORM:
            raise ValueError(f"unknown target transform: {transform}")
    return np.ascontiguousarray(transformed, dtype=np.float32)


def inverse_transform_targets(values: np.ndarray, target_transforms: Iterable[str]) -> np.ndarray:
    transforms = tuple(target_transforms)
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(transforms) or not np.isfinite(matrix).all():
        raise ValueError("invalid transformed target matrix")
    restored = matrix.copy()
    for index, transform in enumerate(transforms):
        if transform == LOG1P_NONNEGATIVE_TARGET_TRANSFORM:
            ceiling = np.float32(np.log(np.finfo(np.float32).max) - 1.0)
            restored[:, index] = np.maximum(np.expm1(np.minimum(restored[:, index], ceiling)), 0.0)
        elif transform != IDENTITY_TARGET_TRANSFORM:
            raise ValueError(f"unknown target transform: {transform}")
    return np.ascontiguousarray(restored, dtype=np.float32)


def standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale_value = values.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(scale_value, np.float32(1.0))


def scale(values: np.ndarray, mean: np.ndarray, scale_value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.clip((values - mean) / scale_value, -FEATURE_Z_LIMIT, FEATURE_Z_LIMIT),
        dtype=np.float32,
    )


def regression_error_report(
    label_columns: Iterable[str], truth: np.ndarray, prediction: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    """Return the canonical per-output regression report used by all multi-head tasks.

    Percentage fields use the 0--100 scale.  MAPE excludes zero-valued targets and
    records its denominator; WAPE is null when the aggregate target magnitude is zero.
    """
    labels = tuple(label_columns)
    expected = (len(labels),)
    if truth.ndim != 2 or prediction.shape != truth.shape or truth.shape[1:] != expected:
        raise ValueError("truth and prediction must be matching two-dimensional label matrices")
    absolute = np.abs(truth - prediction).astype(np.float64, copy=False)
    report: dict[str, dict[str, float | int | None]] = {}
    for index, label in enumerate(labels):
        target = truth[:, index].astype(np.float64, copy=False)
        estimate = prediction[:, index].astype(np.float64, copy=False)
        error = absolute[:, index]
        nonzero = target != 0.0
        magnitude = float(np.abs(target).sum())
        smape_denominator = np.abs(target) + np.abs(estimate)
        report[label.removeprefix("label_")] = {
            "mae": float(error.mean()),
            "rmse": float(np.sqrt(np.square(target - estimate, dtype=np.float64).mean())),
            "mape_pct": float((error[nonzero] / np.abs(target[nonzero])).mean() * 100.0) if nonzero.any() else None,
            "mape_nonzero_rows": int(nonzero.sum()),
            "p90_absolute_error": float(np.percentile(error, 90)),
            "wape_pct": float(error.sum() / magnitude * 100.0) if magnitude else None,
            "smape_pct": float(np.mean(np.divide(200.0 * error, smape_denominator, out=np.zeros_like(error), where=smape_denominator != 0.0))),
        }
    return report
