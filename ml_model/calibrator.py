"""Workload-generalizing gem5-to-PMU calibration with torch."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.class_weight import compute_sample_weight
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


SPARSE_ZERO_FRACTION = 0.25
SPARSE_MIN_POSITIVES = 20
TRAINING_EPOCHS = 60
TRAINING_BATCH_SIZE = 256
TRAINING_PATIENCE = 8
TRAINING_LEARNING_RATE = 1e-3
TRAINING_WEIGHT_DECAY = 1e-4
RANDOM_SEED = 19


@dataclass(frozen=True)
class CalibratorDataset:
    source_workload_ids: tuple[str, ...]
    source_interval_indices: np.ndarray
    source_proxy_values: np.ndarray
    target_workload_ids: tuple[str, ...]
    target_interval_indices: np.ndarray
    target_means: np.ndarray
    target_measurement_covariances: np.ndarray
    source_proxy_names: tuple[str, ...]
    target_names: tuple[str, ...]
    fingerprint: str
    target_anchor_period: int


@dataclass(frozen=True)
class CalibratorInputs:
    workload_ids: tuple[str, ...]
    interval_indices: np.ndarray
    source_proxy_values: np.ndarray
    source_proxy_names: tuple[str, ...]


@dataclass(frozen=True)
class FittedCalibrator:
    estimator: Any
    source_proxy_names: tuple[str, ...]
    target_names: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True)
class CalibratorPrediction:
    workload_id: str
    interval_index: int
    mean: np.ndarray


@dataclass(frozen=True)
class HeldoutCalibratorPredictions:
    workload_ids: tuple[str, ...]
    interval_indices: np.ndarray
    target_names: tuple[str, ...]
    truth: np.ndarray
    prediction: np.ndarray
    baseline: np.ndarray
    measurement_covariances: np.ndarray


def build_calibrator_dataset(records: Sequence[dict[str, Any]], *, target_records: Sequence[dict[str, Any]], source_names: Sequence[str], target_names: Sequence[str], target_anchor_period: int) -> CalibratorDataset:
    sources, targets = tuple(source_names), tuple(target_names)
    if not records or not target_records or not sources or not targets or target_anchor_period <= 0:
        raise ValueError("calibrator dataset needs source rows, mean targets, and an anchor period")
    source_rows = _source_rows(records, len(sources))
    target_rows = _target_rows(target_records, len(targets))
    if {key[:2] for key in source_rows} != {key[:2] for key in target_rows}:
        raise ValueError("source and PMU mean windows differ")
    targets_by_key = {(workload, interval): (mean, covariance) for workload, interval, mean, covariance in target_rows}
    aligned_targets = [(workload, interval, *targets_by_key[(workload, interval)]) for workload, interval, _ in source_rows]
    source_payload = [{"workload_id": w, "interval_index": i, "source_proxy_values": v.tolist()} for w, i, v in source_rows]
    target_payload = [{"workload_id": w, "interval_index": i, "mean": m.tolist(), "measurement_covariance": c.tolist()} for w, i, m, c in aligned_targets]
    return CalibratorDataset(
        tuple(w for w, _, _ in source_rows), np.asarray([i for _, i, _ in source_rows], dtype=np.int64), np.stack([v for _, _, v in source_rows]),
        tuple(w for w, _, _, _ in aligned_targets), np.asarray([i for _, i, _, _ in aligned_targets], dtype=np.int64), np.stack([m for _, _, m, _ in aligned_targets]), np.stack([c for _, _, _, c in aligned_targets]),
        sources, targets, _fingerprint(source_payload, target_payload, sources, targets, target_anchor_period), target_anchor_period,
    )


def write_calibrator_source(data: CalibratorDataset, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {b"calibrator_source_proxies": ",".join(data.source_proxy_names).encode(), b"calibrator_target_events": ",".join(data.target_names).encode(), b"calibrator_target_anchor_period": str(data.target_anchor_period).encode(), b"calibrator_fingerprint": data.fingerprint.encode()}
    rows = [{"workload_id": w, "interval_index": int(i), "source_proxy_values": v.tolist()} for w, i, v in zip(data.source_workload_ids, data.source_interval_indices, data.source_proxy_values, strict=True)]
    pq.write_table(pa.Table.from_pylist(rows).replace_schema_metadata(metadata), destination, compression="zstd")


def load_calibrator_dataset(source_path: str | Path, target_path: str | Path) -> CalibratorDataset:
    source, target = pq.read_table(source_path), pq.read_table(target_path)
    metadata, target_metadata = source.schema.metadata or {}, target.schema.metadata or {}
    raw_source, raw_target, raw_period, fingerprint = (metadata.get(b"calibrator_source_proxies"), metadata.get(b"calibrator_target_events"), metadata.get(b"calibrator_target_anchor_period"), metadata.get(b"calibrator_fingerprint"))
    if not raw_source or not raw_target or not raw_period or not fingerprint or target_metadata.get(b"calibrator_target_provenance") != b"cross_run_mean_vector":
        raise ValueError("source and target are not a cross-run calibrator dataset")
    if target_metadata.get(b"calibrator_target_events") != raw_target or target_metadata.get(b"calibrator_target_anchor_period") != raw_period:
        raise ValueError("source and PMU target contracts differ")
    data = build_calibrator_dataset(source.to_pylist(), target_records=target.to_pylist(), source_names=raw_source.decode().split(","), target_names=raw_target.decode().split(","), target_anchor_period=int(raw_period))
    if data.fingerprint != fingerprint.decode():
        raise ValueError("calibrator source content differs from its fingerprint")
    return data


def build_calibrator_inputs(records: Sequence[dict[str, Any]], *, source_names: Sequence[str]) -> CalibratorInputs:
    rows = _source_rows(records, len(source_names))
    return CalibratorInputs(tuple(w for w, _, _ in rows), np.asarray([i for _, i, _ in rows], dtype=np.int64), np.stack([v for _, _, v in rows]), tuple(source_names))


def train_calibrator(data: CalibratorDataset, *, output_path: str | Path | None = None) -> FittedCalibrator:
    _require_workloads(data)
    estimator = _estimator().fit(
        data.source_proxy_values,
        data.target_means,
        sample_weight=_workload_weights(data.source_workload_ids),
        source_names=data.source_proxy_names,
        target_names=data.target_names,
    )
    fitted = FittedCalibrator(estimator, data.source_proxy_names, data.target_names, data.fingerprint)
    if output_path is not None:
        _save(fitted, Path(output_path))
    return fitted


def predict_calibrator(fitted: FittedCalibrator | str | Path, source: CalibratorDataset | CalibratorInputs) -> tuple[CalibratorPrediction, ...]:
    model, inputs = _coerce_fitted(fitted), _coerce_inputs(source)
    if inputs.source_proxy_names != model.source_proxy_names:
        raise ValueError("calibrator source proxy names differ")
    mean = np.asarray(model.estimator.predict(_source_matrix(inputs.source_proxy_values, len(model.source_proxy_names))), dtype=np.float64)
    return tuple(CalibratorPrediction(w, int(i), point.astype(np.float64)) for w, i, point in zip(inputs.workload_ids, inputs.interval_indices, mean, strict=True))


def heldout_calibrator_predictions(data: CalibratorDataset) -> HeldoutCalibratorPredictions:
    _require_workloads(data)
    groups = np.asarray(data.source_workload_ids, dtype=object)
    prediction = cross_val_predict(
        _estimator(), data.source_proxy_values, data.target_means,
        groups=groups, cv=LeaveOneGroupOut(), n_jobs=1,
        params={
            "sample_weight": _workload_weights(data.source_workload_ids),
            "source_names": data.source_proxy_names,
            "target_names": data.target_names,
        },
    )
    baseline = _lowo_median_baseline(data)
    return HeldoutCalibratorPredictions(
        workload_ids=data.target_workload_ids,
        interval_indices=data.target_interval_indices.copy(),
        target_names=data.target_names,
        truth=data.target_means.copy(),
        prediction=np.asarray(prediction, dtype=np.float64),
        baseline=baseline,
        measurement_covariances=data.target_measurement_covariances.copy(),
    )


class _SparseCalibratorNet(nn.Module):
    def __init__(self, input_size: int, target_modes: tuple[str, ...]) -> None:
        super().__init__()
        hidden = max(16, min(64, input_size * 4))
        self.target_modes = target_modes
        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_heads = nn.ModuleList(nn.Linear(hidden, 1) for _ in target_modes)
        self.presence_heads = nn.ModuleList(
            nn.Linear(hidden, 1) if mode == "sparse" else nn.Identity()
            for mode in target_modes
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.trunk(features)
        values = torch.cat([head(encoded) for head in self.value_heads], dim=1)
        presences = []
        for mode, head in zip(self.target_modes, self.presence_heads, strict=True):
            if mode == "sparse":
                presences.append(head(encoded))
            else:
                presences.append(torch.zeros((len(features), 1), dtype=features.dtype, device=features.device))
        return values, torch.cat(presences, dim=1)


@dataclass(frozen=True)
class _PreparedTargets:
    transformed_values: np.ndarray
    presence: np.ndarray
    value_mean: np.ndarray
    value_scale: np.ndarray


class TorchSparseCalibrator(RegressorMixin, BaseEstimator):
    def __init__(self, *, seed: int = RANDOM_SEED, shrinkage: float = 0.0) -> None:
        self.seed = seed
        self.shrinkage = shrinkage

    def fit(
        self,
        x: Any,
        y: Any,
        sample_weight: Any | None = None,
        source_names: Sequence[str] | None = None,
        target_names: Sequence[str] | None = None,
    ) -> "TorchSparseCalibrator":
        features = _source_matrix(x, len(source_names) if source_names is not None else np.asarray(x).shape[1])
        targets = _target_matrix(y)
        if len(features) != len(targets):
            raise ValueError("calibrator feature and target rows differ")
        if not 0.0 <= float(self.shrinkage) <= 1.0:
            raise ValueError("calibrator shrinkage must be between zero and one")
        active = np.std(features, axis=0) > 0.0
        if not active.any():
            raise ValueError("calibrator source contains no varying features")
        self.source_names = tuple(source_names or tuple(f"source_{index}" for index in range(features.shape[1])))
        self.target_names = tuple(target_names or tuple(f"target_{index}" for index in range(targets.shape[1])))
        self.active_mask = active
        self.active_source_names = tuple(name for name, keep in zip(self.source_names, active, strict=True) if keep)
        self.target_min = targets.min(axis=0).astype(np.float64)
        self.target_max = targets.max(axis=0).astype(np.float64)
        self.target_median = np.median(targets, axis=0).astype(np.float64)
        self.target_modes = _target_modes(targets)
        prepared_x = self._prepare_features_for_fit(features)
        prepared_y = _prepare_targets(targets, self.target_modes)
        weights = (
            np.ones(len(features), dtype=np.float32)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float32)
        )
        if weights.shape != (len(features),) or not np.isfinite(weights).all() or (weights <= 0.0).any():
            raise ValueError("calibrator sample weights must be positive")
        self.target_value_mean = prepared_y.value_mean
        self.target_value_scale = prepared_y.value_scale
        self.model = _train_model(
            prepared_x,
            prepared_y,
            weights,
            target_modes=self.target_modes,
            seed=self.seed,
        )
        return self

    def predict(self, x: Any) -> np.ndarray:
        features = _source_matrix(x, len(self.source_names))
        prepared = self._prepare_features_for_predict(features)
        self.model.eval()
        with torch.inference_mode():
            values, presences = self.model(torch.from_numpy(prepared))
        transformed = values.numpy() * self.target_value_scale + self.target_value_mean
        prediction = _inverse_targets(transformed, presences.numpy(), self.target_modes)
        prediction = self.target_median + float(self.shrinkage) * (prediction - self.target_median)
        return np.clip(prediction, self.target_min, self.target_max).astype(np.float64)

    def _prepare_features_for_fit(self, values: np.ndarray) -> np.ndarray:
        selected = np.log1p(values[:, self.active_mask])
        self.feature_mean = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.feature_scale = np.maximum(selected.std(axis=0, dtype=np.float64).astype(np.float32), np.float32(1.0))
        return _scale(selected, self.feature_mean, self.feature_scale)

    def _prepare_features_for_predict(self, values: np.ndarray) -> np.ndarray:
        selected = np.log1p(values[:, self.active_mask])
        return _scale(selected, self.feature_mean, self.feature_scale)


def _estimator() -> TorchSparseCalibrator:
    return TorchSparseCalibrator()


def _target_modes(targets: np.ndarray) -> tuple[str, ...]:
    modes = []
    for column in targets.T:
        zero_fraction = float(np.mean(np.isclose(column, 0.0, atol=1e-12)))
        positives = int(np.count_nonzero(column > 0.0))
        modes.append(
            "sparse"
            if zero_fraction >= SPARSE_ZERO_FRACTION and positives >= SPARSE_MIN_POSITIVES
            else "dense"
        )
    return tuple(modes)


def _prepare_targets(targets: np.ndarray, target_modes: tuple[str, ...]) -> _PreparedTargets:
    values = np.log1p(targets).astype(np.float32)
    presence = (targets > 0.0).astype(np.float32)
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(values.std(axis=0, dtype=np.float64).astype(np.float32), np.float32(1.0))
    return _PreparedTargets(_scale(values, mean, scale), presence, mean, scale)


def _train_model(
    features: np.ndarray,
    targets: _PreparedTargets,
    sample_weight: np.ndarray,
    *,
    target_modes: tuple[str, ...],
    seed: int,
) -> _SparseCalibratorNet:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = _SparseCalibratorNet(features.shape[1], target_modes)
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(targets.transformed_values),
        torch.from_numpy(targets.presence),
        torch.from_numpy(sample_weight.reshape(-1, 1)),
    )
    loader = DataLoader(dataset, batch_size=min(TRAINING_BATCH_SIZE, len(features)), shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_LEARNING_RATE, weight_decay=TRAINING_WEIGHT_DECAY)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    all_x = torch.from_numpy(features)
    all_y = torch.from_numpy(targets.transformed_values)
    all_presence = torch.from_numpy(targets.presence)
    all_weight = torch.from_numpy(sample_weight.reshape(-1, 1))
    for _epoch in range(TRAINING_EPOCHS):
        model.train()
        for batch_x, batch_y, batch_presence, batch_weight in loader:
            optimizer.zero_grad()
            loss = _training_loss(model, batch_x, batch_y, batch_presence, batch_weight)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            current = float(_training_loss(model, all_x, all_y, all_presence, all_weight))
        if current + 1e-7 < best_loss:
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_loss = current
            stale = 0
        else:
            stale += 1
            if stale >= TRAINING_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _training_loss(
    model: _SparseCalibratorNet,
    features: torch.Tensor,
    transformed_targets: torch.Tensor,
    presence: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    values, presences = model(features)
    losses = []
    for index, mode in enumerate(model.target_modes):
        weight = sample_weight.reshape(-1)
        if mode == "sparse":
            cls_loss = F.binary_cross_entropy_with_logits(
                presences[:, index],
                presence[:, index],
                weight=weight,
                reduction="mean",
            )
            positive = presence[:, index] > 0.0
            if positive.any():
                reg_loss = F.smooth_l1_loss(
                    values[positive, index],
                    transformed_targets[positive, index],
                    reduction="none",
                )
                reg_loss = (reg_loss * weight[positive]).mean()
            else:
                reg_loss = torch.zeros((), dtype=features.dtype, device=features.device)
            losses.append(cls_loss + reg_loss)
        else:
            reg_loss = F.smooth_l1_loss(
                values[:, index],
                transformed_targets[:, index],
                reduction="none",
            )
            losses.append((reg_loss * weight).mean())
    return torch.stack(losses).mean()


def _inverse_targets(values: np.ndarray, presences: np.ndarray, target_modes: tuple[str, ...]) -> np.ndarray:
    restored = np.maximum(np.expm1(values), 0.0)
    for index, mode in enumerate(target_modes):
        if mode == "sparse":
            restored[:, index] *= 1.0 / (1.0 + np.exp(-presences[:, index]))
    return restored.astype(np.float32)


def _scale(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.clip((values - mean) / scale, -8.0, 8.0), dtype=np.float32)


def _workload_weights(workloads: Sequence[str]) -> np.ndarray:
    return compute_sample_weight("balanced", np.asarray(workloads, dtype=object))


def _lowo_median_baseline(data: CalibratorDataset) -> np.ndarray:
    groups = np.asarray(data.target_workload_ids, dtype=object)
    out = np.empty_like(data.target_means)
    for train_rows, heldout_rows in LeaveOneGroupOut().split(data.target_means, groups=groups):
        out[heldout_rows] = np.median(data.target_means[train_rows], axis=0)
    return out


def _require_workloads(data: CalibratorDataset) -> None:
    if len(set(data.source_workload_ids)) < 3:
        raise ValueError("calibrator training requires at least three workloads")


def _source_rows(records: Sequence[dict[str, Any]], width: int) -> list[tuple[str, int, np.ndarray]]:
    rows = []
    identities = set()
    for record in records:
        key = str(record["workload_id"]), int(record["interval_index"])
        if key in identities:
            raise ValueError(f"duplicate calibrator source window: {key}")
        identities.add(key)
        rows.append((*key, _source_matrix(record["source_proxy_values"], width).reshape(-1)))
    if not rows:
        raise ValueError("calibrator inputs are empty")
    return rows


def _target_rows(records: Sequence[dict[str, Any]], width: int) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    rows = []
    identities = set()
    for record in records:
        key = str(record["workload_id"]), int(record["interval_index"])
        if key in identities:
            raise ValueError(f"duplicate PMU mean identity: {key}")
        identities.add(key)
        mean = _target_matrix(record["mean"]).reshape(-1)
        covariance = np.asarray(record["measurement_covariance"], dtype=np.float64)
        symmetric = (covariance + covariance.T) / 2.0 if covariance.ndim == 2 else covariance
        if mean.shape != (width,) or covariance.shape != (width, width) or not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, rtol=1e-5, atol=1e-7) or np.linalg.eigvalsh(symmetric).min() < -1e-5:
            raise ValueError("PMU mean/covariance has an invalid shape")
        rows.append((*key, mean, symmetric))
    return rows


def _source_matrix(values: Any, width: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    matrix = matrix.reshape(1, -1) if matrix.ndim == 1 else matrix
    if matrix.ndim != 2 or matrix.shape[1] != width or not np.isfinite(matrix).all() or (matrix < 0.0).any():
        raise ValueError("calibrator source proxies must be finite and nonnegative")
    return matrix


def _target_matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    matrix = matrix.reshape(-1, 1) if matrix.ndim == 1 else matrix
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or (matrix < 0.0).any():
        raise ValueError("calibrator target rates must be finite and nonnegative")
    return matrix


def _fingerprint(source: Sequence[dict[str, Any]], target: Sequence[dict[str, Any]], source_names: Sequence[str], target_names: Sequence[str], period: int) -> str:
    return sha256(json.dumps({"source": source, "target": target, "source_names": list(source_names), "target_names": list(target_names), "target_anchor_period": period}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _save(fitted: FittedCalibrator, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, path)


def load_calibrator(path: str | Path) -> FittedCalibrator:
    fitted = joblib.load(path)
    if not isinstance(fitted, FittedCalibrator) or not isinstance(getattr(fitted, "estimator", None), TorchSparseCalibrator):
        raise ValueError("calibrator artifact does not use the torch sparse-head contract")
    return fitted


def _coerce_inputs(value: CalibratorDataset | CalibratorInputs) -> CalibratorInputs:
    return CalibratorInputs(value.source_workload_ids, value.source_interval_indices, value.source_proxy_values, value.source_proxy_names) if isinstance(value, CalibratorDataset) else value


def _coerce_fitted(value: FittedCalibrator | str | Path) -> FittedCalibrator:
    return value if isinstance(value, FittedCalibrator) else load_calibrator(value)
