"""Residual simulator-to-N2 calibration with Gaussian performance intervals."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.stats import chi2
from torch import nn
from torch.nn import functional as F

import ot

from .multitask import scale, standardize


DEFAULT_EPOCHS = 200
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_CONFIDENCE_LEVEL = 0.95
OT_WEIGHT_CANDIDATES = (0.0, 0.03, 0.1, 0.3)
OT_UPDATE_INTERVAL = 4


@dataclass(frozen=True)
class CalibratorDataset:
    reference_config_id: str
    workload_ids: tuple[str, ...]
    interval_indices: np.ndarray
    config_ids: tuple[str, ...]
    configs: np.ndarray
    proxy_metrics: np.ndarray
    target_workload_ids: tuple[str, ...]
    target_interval_indices: np.ndarray
    target_means: np.ndarray
    target_sample_covariances: np.ndarray
    config_names: tuple[str, ...]
    proxy_names: tuple[str, ...]
    output_names: tuple[str, ...]


@dataclass(frozen=True)
class CalibratorInputs:
    workload_ids: tuple[str, ...]
    interval_indices: np.ndarray
    config_ids: tuple[str, ...]
    configs: np.ndarray
    proxy_metrics: np.ndarray
    config_names: tuple[str, ...]
    proxy_names: tuple[str, ...]


@dataclass(frozen=True)
class FittedCalibrator:
    model: nn.Module
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    output_mean: np.ndarray
    output_scale: np.ndarray
    config_names: tuple[str, ...]
    proxy_names: tuple[str, ...]
    output_names: tuple[str, ...]
    confidence_level: float
    ot_weight: float
    epochs: int


@dataclass(frozen=True)
class CalibratorPrediction:
    workload_id: str
    interval_index: int
    config_id: str
    lower: np.ndarray
    mean: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class _TransportGroup:
    workload_id: str
    config_id: str
    source: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True)
class _PreparedTargets:
    means: torch.Tensor
    covariances: torch.Tensor
    reference_source: torch.Tensor
    reference_target: torch.Tensor
    reference_weights: torch.Tensor
    groups: tuple[_TransportGroup, ...]


@dataclass(frozen=True)
class _PreparedCalibratorData:
    x: torch.Tensor
    proxy_base: torch.Tensor
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    output_mean: np.ndarray
    output_scale: np.ndarray
    targets: _PreparedTargets


class _GaussianCalibratorModel(nn.Module):
    """Shared residual mean trunk plus a PSD covariance head."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.output_size = output_size
        self.trunk = nn.Sequential(
            nn.Linear(input_size, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
        self.mean_residual = nn.Linear(16, output_size)
        self.covariance_factor = nn.Linear(16, output_size * (output_size + 1) // 2)
        nn.init.zeros_(self.mean_residual.weight)
        nn.init.zeros_(self.mean_residual.bias)
        nn.init.zeros_(self.covariance_factor.weight)
        nn.init.zeros_(self.covariance_factor.bias)
        lower = torch.tril_indices(output_size, output_size)
        diagonal_positions = torch.nonzero(lower[0] == lower[1], as_tuple=False).flatten()
        initial_diagonal = torch.linspace(0.5, 0.9, output_size)
        with torch.no_grad():
            self.covariance_factor.bias[diagonal_positions] = torch.log(torch.expm1(initial_diagonal - 1e-5))

    def forward(self, features: torch.Tensor, proxy_base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.trunk(features)
        mean = proxy_base + self.mean_residual(encoded)
        packed = self.covariance_factor(encoded)
        factor = torch.zeros((len(features), self.output_size, self.output_size), dtype=features.dtype, device=features.device)
        lower = torch.tril_indices(self.output_size, self.output_size, device=features.device)
        factor[:, lower[0], lower[1]] = 3.0 * torch.tanh(packed / 3.0)
        diagonal = torch.arange(self.output_size, device=features.device)
        factor[:, diagonal, diagonal] = F.softplus(factor[:, diagonal, diagonal]).clamp(max=3.0) + 1e-3
        return mean, factor @ factor.transpose(-1, -2)


def build_calibrator_dataset(
    records: Sequence[dict[str, Any]], *, target_records: Sequence[dict[str, Any]],
    config_names: Sequence[str], proxy_names: Sequence[str], output_names: Sequence[str],
    reference_config_id: str, output_path: str | Path | None = None, require_reference: bool = True,
) -> CalibratorDataset:
    """Validate simulator views and canonical N2 performance moments."""
    configs_names, proxies_names, outputs = tuple(config_names), tuple(proxy_names), tuple(output_names)
    if not records or not target_records or not configs_names or not proxies_names or not outputs:
        raise ValueError("calibrator dataset needs source rows, target rows, and metric names")
    if len(set(configs_names)) != len(configs_names) or len(set(proxies_names)) != len(proxies_names):
        raise ValueError("calibrator input names must be unique")
    if set(outputs) - set(proxies_names):
        raise ValueError("calibrator outputs must be simulator proxy metrics")

    identities: set[tuple[str, int, str]] = set()
    workloads: list[str] = []; intervals: list[int] = []; config_ids: list[str] = []
    configs: list[np.ndarray] = []; proxies: list[np.ndarray] = []
    for record in records:
        workload, interval, config_id = str(record["workload_id"]), int(record["interval_index"]), str(record["config_id"])
        identity = (workload, interval, config_id)
        if identity in identities:
            raise ValueError(f"duplicate calibrator identity: {identity}")
        identities.add(identity)
        config = np.asarray(record["config"], dtype=np.float32)
        proxy = np.asarray(record["proxy_metrics"], dtype=np.float32)
        if config.shape != (len(configs_names),) or proxy.shape != (len(proxies_names),):
            raise ValueError("calibrator source record has invalid vector widths")
        if not np.isfinite(config).all() or not np.isfinite(proxy).all() or (proxy < 0.0).any():
            raise ValueError("calibrator source rows must be finite with nonnegative metrics")
        workloads.append(workload); intervals.append(interval); config_ids.append(config_id); configs.append(config); proxies.append(proxy)

    target_ids: set[tuple[str, int]] = set()
    target_workloads: list[str] = []; target_intervals: list[int] = []
    means: list[np.ndarray] = []; covariances: list[np.ndarray] = []
    for record in target_records:
        workload, interval = str(record["workload_id"]), int(record["interval_index"])
        identity = (workload, interval)
        if identity in target_ids:
            raise ValueError(f"duplicate PMU target identity: {identity}")
        target_ids.add(identity)
        mean = np.asarray(record["mean"], dtype=np.float32)
        covariance = np.asarray(record["sample_covariance"], dtype=np.float32)
        if mean.shape != (len(outputs),) or covariance.shape != (len(outputs), len(outputs)):
            raise ValueError("calibrator target row has invalid mean/covariance widths")
        if not np.isfinite(mean).all() or (mean < 0.0).any() or not np.isfinite(covariance).all():
            raise ValueError("calibrator target rows must be finite with nonnegative means")
        _validate_covariance(covariance)
        target_workloads.append(workload); target_intervals.append(interval); means.append(mean); covariances.append(covariance)

    if {(workload, interval) for workload, interval in zip(workloads, intervals, strict=True)} - target_ids:
        raise ValueError("calibrator source windows lack fixed-N2 PMU targets")
    data = CalibratorDataset(
        reference_config_id, tuple(workloads), np.asarray(intervals, dtype=np.int64), tuple(config_ids),
        np.stack(configs), np.stack(proxies), tuple(target_workloads), np.asarray(target_intervals, dtype=np.int64),
        np.stack(means), np.stack(covariances), configs_names, proxies_names, outputs,
    )
    if require_reference:
        _validate_reference_coverage(data)
    if output_path is not None:
        _write_source_dataset(data, Path(output_path))
    return data


def build_calibrator_inputs(records: Sequence[dict[str, Any]], *, config_names: Sequence[str], proxy_names: Sequence[str]) -> CalibratorInputs:
    configs_names, proxies_names = tuple(config_names), tuple(proxy_names)
    if not records or not configs_names or not proxies_names:
        raise ValueError("calibrator inputs need records, config names, and proxy names")
    identities: set[tuple[str, int, str]] = set()
    workloads: list[str] = []; intervals: list[int] = []; config_ids: list[str] = []
    configs: list[np.ndarray] = []; proxies: list[np.ndarray] = []
    for record in records:
        workload, interval, config_id = str(record["workload_id"]), int(record["interval_index"]), str(record["config_id"])
        identity = (workload, interval, config_id)
        if identity in identities:
            raise ValueError(f"duplicate calibrator identity: {identity}")
        identities.add(identity)
        config = np.asarray(record["config"], dtype=np.float32)
        proxy = np.asarray(record["proxy_metrics"], dtype=np.float32)
        if config.shape != (len(configs_names),) or proxy.shape != (len(proxies_names),):
            raise ValueError("calibrator input has invalid vector widths")
        if not np.isfinite(config).all() or not np.isfinite(proxy).all() or (proxy < 0.0).any():
            raise ValueError("calibrator inputs must be finite with nonnegative metrics")
        workloads.append(workload); intervals.append(interval); config_ids.append(config_id); configs.append(config); proxies.append(proxy)
    return CalibratorInputs(tuple(workloads), np.asarray(intervals, dtype=np.int64), tuple(config_ids), np.stack(configs), np.stack(proxies), configs_names, proxies_names)


def train_calibrator(
    dataset: CalibratorDataset | str | Path, *, output_path: str | Path | None = None,
    epochs: int = DEFAULT_EPOCHS, learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY, ot_weight: float | None = None,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL, seed: int = 19,
) -> FittedCalibrator:
    data = _coerce_dataset(dataset)
    if epochs < 1 or learning_rate <= 0.0 or weight_decay < 0.0 or not 0.0 < confidence_level < 1.0:
        raise ValueError("invalid calibrator training parameters")
    chosen_weight = _select_ot_weight(data, epochs=min(epochs, 10), learning_rate=learning_rate, weight_decay=weight_decay, confidence_level=confidence_level, seed=seed) if ot_weight is None else ot_weight
    if chosen_weight not in OT_WEIGHT_CANDIDATES:
        raise ValueError("calibrator OT weight is not an approved candidate")
    fitted = _fit(data, epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay, confidence_level=confidence_level, ot_weight=chosen_weight, seed=seed)
    if output_path is not None:
        _save(fitted, Path(output_path))
    return fitted


def predict_calibrator(fitted: FittedCalibrator | str | Path, dataset: CalibratorDataset | CalibratorInputs | str | Path) -> tuple[CalibratorPrediction, ...]:
    model, data = _coerce_fitted(fitted), _coerce_inputs(dataset)
    if (model.config_names, model.proxy_names) != (data.config_names, data.proxy_names):
        raise ValueError("calibrator model and dataset inputs differ")
    means, covariances = _predict_moments(model, data)
    radius = float(chi2.ppf(model.confidence_level, len(model.output_names)))
    half_widths = np.sqrt(np.maximum(np.diagonal(covariances, axis1=1, axis2=2), 0.0) * radius)
    lower = np.maximum(means - half_widths, 0.0).astype(np.float32)
    upper = (means + half_widths).astype(np.float32)
    return tuple(
        CalibratorPrediction(workload, int(interval), config_id, low, mean, high)
        for workload, interval, config_id, low, mean, high in zip(
            data.workload_ids, data.interval_indices, data.config_ids, lower, means, upper, strict=True,
        )
    )


def evaluate_calibrator(dataset: CalibratorDataset | str | Path, *, ot_weight: float, folds: int = 4, seed: int = 19) -> dict[str, Any]:
    """Workload/config out-of-domain evaluation with a fixed OT weight."""
    if ot_weight not in OT_WEIGHT_CANDIDATES:
        raise ValueError("calibrator OT weight is not an approved candidate")
    data = _coerce_dataset(dataset)
    workloads = tuple(sorted(set(data.target_workload_ids)))
    configs = tuple(config for config in sorted(set(data.config_ids)) if config != data.reference_config_id)
    if folds < 2 or len(workloads) < folds or len(configs) < folds:
        raise ValueError("calibrator evaluation needs enough workload and non-reference config groups")
    rng = np.random.default_rng(seed)
    shuffled_workloads = list(workloads); rng.shuffle(shuffled_workloads)
    shuffled_configs = list(configs); rng.shuffle(shuffled_configs)
    reports: dict[str, list[dict[str, Any]]] = {"workload_ood": [], "config_ood": [], "joint_ood": []}
    for index in range(folds):
        held_workloads = tuple(shuffled_workloads[index::folds])
        held_configs = tuple(shuffled_configs[index::folds])
        train = _subset(data, excluded_workloads=held_workloads, excluded_configs=held_configs, require_reference=True)
        fitted = _fit(train, epochs=DEFAULT_EPOCHS, learning_rate=DEFAULT_LEARNING_RATE, weight_decay=DEFAULT_WEIGHT_DECAY, confidence_level=DEFAULT_CONFIDENCE_LEVEL, ot_weight=ot_weight, seed=seed + index)
        reports["workload_ood"].append(_moment_report(fitted, _subset(data, selected_workloads=held_workloads, excluded_configs=held_configs, require_reference=False)))
        reports["config_ood"].append(_moment_report(fitted, _subset(data, excluded_workloads=held_workloads, selected_configs=held_configs, require_reference=False)))
        reports["joint_ood"].append(_moment_report(fitted, _subset(data, selected_workloads=held_workloads, selected_configs=held_configs, require_reference=False)))
    return {"protocol": "joint_workload_config_holdout", "folds": folds, **{name: _mean_reports(values, data.output_names) for name, values in reports.items()}}


def _fit(data: CalibratorDataset, *, epochs: int, learning_rate: float, weight_decay: float, confidence_level: float, ot_weight: float, seed: int) -> FittedCalibrator:
    prepared = _prepare_calibrator_data(data)
    torch.manual_seed(seed)
    model = _GaussianCalibratorModel(prepared.x.shape[1], len(data.output_names))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    ot_batches = _ot_group_batches(
        prepared.targets.groups,
        updates=(epochs + OT_UPDATE_INTERVAL - 1) // OT_UPDATE_INTERVAL,
        seed=seed,
    ) if ot_weight else ()
    with _small_matrix_threads():
        for epoch in range(epochs):
            optimizer.zero_grad()
            mean, covariance = model(prepared.x, prepared.proxy_base)
            targets = prepared.targets
            paired = _paired_w2_squared(
                mean[targets.reference_source], _ot_covariances(covariance[targets.reference_source]),
                targets.means[targets.reference_target], _ot_covariances(targets.covariances[targets.reference_target]),
            )
            anchor_loss = (paired * targets.reference_weights).sum()
            if ot_weight and epoch % OT_UPDATE_INTERVAL == 0:
                ot_loss = torch.stack([
                    ot.gmm.gmm_ot_loss(
                        mean[group.source], targets.means[group.target], _ot_covariances(covariance[group.source]), _ot_covariances(targets.covariances[group.target]),
                        torch.full((len(group.source),), 1.0 / len(group.source), dtype=mean.dtype),
                        torch.full((len(group.target),), 1.0 / len(group.target), dtype=mean.dtype),
                    )
                    for group in ot_batches[epoch // OT_UPDATE_INTERVAL]
                ]).mean()
            else:
                ot_loss = anchor_loss.new_zeros(())
            (anchor_loss + ot_weight * ot_loss).backward()
            optimizer.step()
    model.eval()
    return FittedCalibrator(model, prepared.feature_mean, prepared.feature_scale, prepared.output_mean, prepared.output_scale, data.config_names, data.proxy_names, data.output_names, confidence_level, ot_weight, epochs)


def _prepare_calibrator_data(data: CalibratorDataset) -> _PreparedCalibratorData:
    features = np.concatenate((data.configs, np.log1p(data.proxy_metrics)), axis=1)
    feature_mean, feature_scale = standardize(features)
    output_mean, output_scale = standardize(data.target_means)
    positions = tuple(data.proxy_names.index(name) for name in data.output_names)
    proxy_base = (data.proxy_metrics[:, positions] - output_mean) / output_scale
    scale_matrix = np.diag(1.0 / output_scale).astype(np.float32)
    target_covariances = np.asarray([scale_matrix @ covariance @ scale_matrix for covariance in data.target_sample_covariances], dtype=np.float32)
    lookup = _target_lookup(data)
    reference_source = torch.as_tensor([index for index, config in enumerate(data.config_ids) if config == data.reference_config_id], dtype=torch.long)
    reference_target = torch.as_tensor([lookup[(data.workload_ids[index], int(data.interval_indices[index]))] for index in reference_source.tolist()], dtype=torch.long)
    reference_workloads = tuple(data.workload_ids[index] for index in reference_source.tolist())
    return _PreparedCalibratorData(
        x=torch.from_numpy(scale(features, feature_mean, feature_scale)), proxy_base=torch.from_numpy(proxy_base.astype(np.float32)),
        feature_mean=feature_mean, feature_scale=feature_scale, output_mean=output_mean, output_scale=output_scale,
        targets=_PreparedTargets(
            means=torch.from_numpy(((data.target_means - output_mean) / output_scale).astype(np.float32)),
            covariances=_stabilize_covariances(torch.from_numpy(target_covariances)), reference_source=reference_source,
            reference_target=reference_target, reference_weights=torch.from_numpy(_macro_weights(reference_workloads)),
            groups=_transport_groups(data),
        ),
    )


def _paired_w2_squared(source_mean: torch.Tensor, source_covariance: torch.Tensor, target_mean: torch.Tensor, target_covariance: torch.Tensor) -> torch.Tensor:
    costs = ot.gmm.dist_bures_squared(source_mean, target_mean, source_covariance, target_covariance)
    return torch.diagonal(costs)


def _stabilize_covariances(covariances: torch.Tensor, *, floor: float = 1e-3) -> torch.Tensor:
    symmetric = (covariances + covariances.transpose(-1, -2)) / 2.0
    values, vectors = torch.linalg.eigh(symmetric)
    return (vectors * values.clamp_min(floor).unsqueeze(-2)) @ vectors.transpose(-1, -2)


def _ot_covariances(covariances: torch.Tensor) -> torch.Tensor:
    dimension = covariances.shape[-1]
    axes = torch.arange(1, dimension + 1, dtype=covariances.dtype, device=covariances.device)
    return (covariances + covariances.transpose(-1, -2)) / 2.0 + torch.diag(axes * 1e-3)


@contextmanager
def _small_matrix_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _transport_groups(data: CalibratorDataset) -> tuple[_TransportGroup, ...]:
    targets_by_workload: dict[str, list[int]] = {}
    sources_by_group: dict[tuple[str, str], list[int]] = {}
    for index, workload in enumerate(data.target_workload_ids):
        targets_by_workload.setdefault(workload, []).append(index)
    for index, (workload, config) in enumerate(zip(data.workload_ids, data.config_ids, strict=True)):
        if config != data.reference_config_id:
            sources_by_group.setdefault((workload, config), []).append(index)
    groups: list[_TransportGroup] = []
    for (workload, config), source_indices in sorted(sources_by_group.items()):
        target_indices = targets_by_workload[workload]
        if np.sort(data.interval_indices[source_indices]).tolist() != np.sort(data.target_interval_indices[target_indices]).tolist():
            raise ValueError("each non-reference config must cover every PMU target interval")
        groups.append(_TransportGroup(workload, config, torch.as_tensor(source_indices, dtype=torch.long), torch.as_tensor(target_indices, dtype=torch.long)))
    return tuple(groups)


def _ot_group_batches(groups: Sequence[_TransportGroup], *, updates: int, seed: int, batch_size: int = 8) -> tuple[tuple[_TransportGroup, ...], ...]:
    if not groups or updates < 1 or batch_size < 1:
        raise ValueError("OT domain batches require groups, updates, and batch size")
    rng = np.random.default_rng(seed)
    batches: list[tuple[_TransportGroup, ...]] = []
    while len(batches) < updates:
        order = rng.permutation(len(groups))
        for start in range(0, len(groups), batch_size):
            batches.append(tuple(groups[index] for index in order[start:start + batch_size]))
            if len(batches) == updates:
                break
    return tuple(batches)


def _predict_moments(fitted: FittedCalibrator, data: CalibratorInputs) -> tuple[np.ndarray, np.ndarray]:
    features = np.concatenate((data.configs, np.log1p(data.proxy_metrics)), axis=1)
    positions = tuple(data.proxy_names.index(name) for name in fitted.output_names)
    proxy_base = (data.proxy_metrics[:, positions] - fitted.output_mean) / fitted.output_scale
    with torch.inference_mode():
        mean, covariance = fitted.model(torch.from_numpy(scale(features, fitted.feature_mean, fitted.feature_scale)), torch.from_numpy(proxy_base.astype(np.float32)))
    scale_matrix = np.diag(fitted.output_scale).astype(np.float32)
    raw_mean = np.maximum(mean.numpy() * fitted.output_scale + fitted.output_mean, 0.0).astype(np.float32)
    raw_covariance = np.asarray([scale_matrix @ item @ scale_matrix for item in covariance.numpy()], dtype=np.float32)
    return raw_mean, raw_covariance


def _moment_report(fitted: FittedCalibrator, data: CalibratorDataset) -> dict[str, Any]:
    inputs = _coerce_inputs(data)
    predicted_mean, predicted_covariance = _predict_moments(fitted, inputs)
    lookup = _target_lookup(data)
    target_indices = np.asarray([lookup[(workload, int(interval))] for workload, interval in zip(data.workload_ids, data.interval_indices, strict=True)], dtype=np.int64)
    target_mean = data.target_means[target_indices]
    target_covariance = data.target_sample_covariances[target_indices]
    scale_matrix = np.diag(1.0 / fitted.output_scale).astype(np.float32)
    predicted_mean_normalized = (predicted_mean - fitted.output_mean) / fitted.output_scale
    target_mean_normalized = (target_mean - fitted.output_mean) / fitted.output_scale
    predicted_covariance_normalized = np.asarray([scale_matrix @ item @ scale_matrix for item in predicted_covariance], dtype=np.float32)
    target_covariance_normalized = np.asarray([scale_matrix @ item @ scale_matrix for item in target_covariance], dtype=np.float32)
    w2 = _paired_w2_squared(torch.from_numpy(predicted_mean_normalized), _ot_covariances(_stabilize_covariances(torch.from_numpy(predicted_covariance_normalized))), torch.from_numpy(target_mean_normalized), _ot_covariances(_stabilize_covariances(torch.from_numpy(target_covariance_normalized)))).detach().numpy()
    radius = float(chi2.ppf(fitted.confidence_level, len(fitted.output_names)))
    predicted_width = np.sqrt(np.maximum(np.diagonal(predicted_covariance, axis1=1, axis2=2), 0.0) * radius)
    target_width = np.sqrt(np.maximum(np.diagonal(target_covariance, axis1=1, axis2=2), 0.0) * radius)
    return {
        "w2": _macro_mean(w2, data.workload_ids),
        "mean_error": _metric_error(target_mean, predicted_mean, fitted.output_names, data.workload_ids),
        "interval_error": _metric_error(target_width, predicted_width, fitted.output_names, data.workload_ids),
    }


def _mean_reports(reports: Sequence[dict[str, Any]], outputs: Sequence[str]) -> dict[str, Any]:
    return {
        "gaussian_w2": float(np.mean([report["w2"] for report in reports])),
        "per_metric_mean_error": {
            output: {name: float(np.mean([report["mean_error"][output][name] for report in reports])) for name in ("mae", "rmse")}
            for output in outputs
        },
        "per_metric_interval_error": {
            output: {name: float(np.mean([report["interval_error"][output][name] for report in reports])) for name in ("mae", "rmse")}
            for output in outputs
        },
    }


def _metric_error(actual: np.ndarray, predicted: np.ndarray, outputs: Sequence[str], workloads: Sequence[str]) -> dict[str, dict[str, float]]:
    return {
        output: {
            "mae": _macro_mean(np.abs(actual[:, index] - predicted[:, index]), workloads),
            "rmse": float(np.sqrt(_macro_mean(np.square(actual[:, index] - predicted[:, index]), workloads))),
        }
        for index, output in enumerate(outputs)
    }


def _select_ot_weight(data: CalibratorDataset, *, epochs: int, learning_rate: float, weight_decay: float, confidence_level: float, seed: int) -> float:
    workloads = tuple(sorted(set(data.target_workload_ids)))
    configs = tuple(config for config in sorted(set(data.config_ids)) if config != data.reference_config_id)
    if len(workloads) < 3 or len(configs) < 2:
        return 0.0
    workload_validation = tuple(workloads[index::3] for index in range(3))
    config_validation = tuple(configs[index::3] for index in range(3))
    scores: list[tuple[float, float]] = []
    for weight in OT_WEIGHT_CANDIDATES:
        values: list[float] = []
        for held_workloads, held_configs in zip(workload_validation, config_validation, strict=True):
            train = _subset(data, excluded_workloads=held_workloads, excluded_configs=held_configs, require_reference=True)
            fitted = _fit(train, epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay, confidence_level=confidence_level, ot_weight=weight, seed=seed)
            test = _subset(data, selected_workloads=held_workloads, selected_configs=held_configs, require_reference=False)
            values.append(_moment_report(fitted, test)["w2"])
        scores.append((float(np.mean(values)), weight))
    return min(scores)[1]


def _macro_weights(workloads: Sequence[str]) -> np.ndarray:
    counts = {workload: workloads.count(workload) for workload in set(workloads)}
    return np.asarray([1.0 / (len(counts) * counts[workload]) for workload in workloads], dtype=np.float32)


def _macro_mean(values: np.ndarray, workloads: Sequence[str]) -> float:
    return float(np.mean([np.asarray(values)[[index for index, workload in enumerate(workloads) if workload == selected]].mean() for selected in sorted(set(workloads))]))


def _validate_covariance(covariance: np.ndarray) -> None:
    symmetric = (covariance + covariance.T) / 2.0
    tolerance = 1e-6 * max(1.0, float(np.abs(symmetric).max()))
    if float(np.linalg.eigvalsh(symmetric).min()) < -tolerance:
        raise ValueError("calibrator sample covariance is not positive semidefinite")


def _validate_reference_coverage(data: CalibratorDataset) -> None:
    expected = set(zip(data.target_workload_ids, data.target_interval_indices, strict=True))
    observed = {(workload, int(interval)) for workload, interval, config in zip(data.workload_ids, data.interval_indices, data.config_ids, strict=True) if config == data.reference_config_id}
    if observed != expected:
        raise ValueError("every calibrator target requires one reference proxy point")


def _target_lookup(data: CalibratorDataset) -> dict[tuple[str, int], int]:
    return {(workload, int(interval)): index for index, (workload, interval) in enumerate(zip(data.target_workload_ids, data.target_interval_indices, strict=True))}


def _write_source_dataset(data: CalibratorDataset, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        b"calibrator_config": ",".join(data.config_names).encode(), b"calibrator_proxy": ",".join(data.proxy_names).encode(),
        b"calibrator_outputs": ",".join(data.output_names).encode(), b"calibrator_reference_config_id": data.reference_config_id.encode(),
    }
    source = pa.Table.from_pylist([
        {"workload_id": workload, "interval_index": int(interval), "config_id": config, "config": values.tolist(), "proxy_metrics": proxy.tolist()}
        for workload, interval, config, values, proxy in zip(data.workload_ids, data.interval_indices, data.config_ids, data.configs, data.proxy_metrics, strict=True)
    ]).replace_schema_metadata(metadata)
    pq.write_table(source, directory / "source.parquet", compression="zstd")


def _read_dataset(directory: Path) -> CalibratorDataset:
    source_path, target_path = directory / "source.parquet", directory / "target.parquet"
    source = pq.read_table(source_path); target = pq.read_table(target_path)
    metadata = source.schema.metadata or {}
    required = (b"calibrator_config", b"calibrator_proxy", b"calibrator_outputs", b"calibrator_reference_config_id")
    if any(not metadata.get(key) for key in required):
        raise ValueError("source table is not a calibrator dataset")
    target_metadata = target.schema.metadata or {}
    if target_metadata.get(b"calibrator_target_metrics") != metadata[b"calibrator_outputs"]:
        raise ValueError("source and target metric orders differ")
    return build_calibrator_dataset(source.to_pylist(), target_records=target.to_pylist(), config_names=metadata[b"calibrator_config"].decode().split(","), proxy_names=metadata[b"calibrator_proxy"].decode().split(","), output_names=metadata[b"calibrator_outputs"].decode().split(","), reference_config_id=metadata[b"calibrator_reference_config_id"].decode())


def _save(fitted: FittedCalibrator, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state": fitted.model.state_dict(), "input_size": fitted.model.trunk[0].in_features,
        "config_names": fitted.config_names, "proxy_names": fitted.proxy_names, "outputs": fitted.output_names,
        "feature_mean": fitted.feature_mean, "feature_scale": fitted.feature_scale,
        "output_mean": fitted.output_mean, "output_scale": fitted.output_scale,
        "confidence_level": fitted.confidence_level, "ot_weight": fitted.ot_weight, "epochs": fitted.epochs,
    }, path)


def _load(path: Path) -> FittedCalibrator:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    outputs = tuple(saved["outputs"])
    model = _GaussianCalibratorModel(int(saved["input_size"]), len(outputs))
    model.load_state_dict(saved["state"]); model.eval()
    return FittedCalibrator(model, saved["feature_mean"], saved["feature_scale"], saved["output_mean"], saved["output_scale"], tuple(saved["config_names"]), tuple(saved["proxy_names"]), outputs, float(saved["confidence_level"]), float(saved["ot_weight"]), int(saved["epochs"]))


def load_calibrator(path: str | Path) -> FittedCalibrator:
    return _load(Path(path))


def _coerce_dataset(value: CalibratorDataset | str | Path) -> CalibratorDataset:
    return value if isinstance(value, CalibratorDataset) else _read_dataset(Path(value))


def _coerce_inputs(value: CalibratorDataset | CalibratorInputs | str | Path) -> CalibratorInputs:
    if isinstance(value, CalibratorInputs):
        return value
    if isinstance(value, CalibratorDataset):
        return CalibratorInputs(value.workload_ids, value.interval_indices, value.config_ids, value.configs, value.proxy_metrics, value.config_names, value.proxy_names)
    return _coerce_inputs(_read_dataset(Path(value)))


def _coerce_fitted(value: FittedCalibrator | str | Path) -> FittedCalibrator:
    return value if isinstance(value, FittedCalibrator) else _load(Path(value))


def _subset(
    data: CalibratorDataset, *, selected_workloads: Sequence[str] | None = None, excluded_workloads: Sequence[str] = (),
    selected_configs: Sequence[str] | None = None, excluded_configs: Sequence[str] = (), require_reference: bool,
) -> CalibratorDataset:
    workloads = set(selected_workloads) if selected_workloads is not None else set(data.target_workload_ids)
    workloads -= set(excluded_workloads)
    configs = set(selected_configs) if selected_configs is not None else set(data.config_ids)
    configs -= set(excluded_configs)
    source = [
        {"workload_id": workload, "interval_index": int(interval), "config_id": config, "config": values, "proxy_metrics": proxy}
        for workload, interval, config, values, proxy in zip(data.workload_ids, data.interval_indices, data.config_ids, data.configs, data.proxy_metrics, strict=True)
        if workload in workloads and config in configs
    ]
    target = [
        {"workload_id": workload, "interval_index": int(interval), "mean": mean, "sample_covariance": covariance}
        for workload, interval, mean, covariance in zip(data.target_workload_ids, data.target_interval_indices, data.target_means, data.target_sample_covariances, strict=True)
        if workload in workloads
    ]
    return build_calibrator_dataset(source, target_records=target, config_names=data.config_names, proxy_names=data.proxy_names, output_names=data.output_names, reference_config_id=data.reference_config_id, require_reference=require_reference)
