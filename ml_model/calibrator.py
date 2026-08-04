"""N2-anchored, pointwise simulator-to-machine metric calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

import ot

from .model import MultiHeadPeregrineModel
from .multitask import regression_error_report, scale, standardize


@dataclass(frozen=True)
class CalibratorDataset:
    reference_config_id: str
    workload_ids: tuple[str, ...]
    interval_indices: np.ndarray
    config_ids: tuple[str, ...]
    configs: np.ndarray
    proxy_metrics: np.ndarray
    labels: np.ndarray
    config_names: tuple[str, ...]
    proxy_names: tuple[str, ...]
    output_names: tuple[str, ...]


@dataclass(frozen=True)
class FittedCalibrator:
    model: MultiHeadPeregrineModel
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    output_mean: np.ndarray
    output_scale: np.ndarray
    config_names: tuple[str, ...]
    proxy_names: tuple[str, ...]
    output_names: tuple[str, ...]
    epochs: int


@dataclass(frozen=True)
class CalibratorPrediction:
    workload_id: str
    interval_index: int
    config_id: str
    prediction: np.ndarray


DEFAULT_EPOCHS = 200
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_SINKHORN_REGULARIZATION = 0.1
DEFAULT_SINKHORN_WEIGHT = 1.0
DEFAULT_VIEW_WEIGHT = 1.0


def build_calibrator_dataset(
    records: Sequence[dict[str, Any]], *, config_names: Sequence[str], proxy_names: Sequence[str],
    output_names: Sequence[str], reference_config_id: str, output_path: str | Path | None = None, require_reference: bool = True,
) -> CalibratorDataset:
    """Validate canonical 1M proxy points paired with fixed-N2 PMU labels."""
    configs_names, proxies_names, outputs = tuple(config_names), tuple(proxy_names), tuple(output_names)
    if not records or not configs_names or not proxies_names or not outputs:
        raise ValueError("calibrator dataset needs records, config, proxy, and output names")
    if len(set(configs_names)) != len(configs_names) or len(set(proxies_names)) != len(proxies_names):
        raise ValueError("calibrator input names must be unique")
    if set(outputs) - set(proxies_names):
        raise ValueError("calibrator outputs must be predicted proxy metrics")
    identities: set[tuple[str, int, str]] = set()
    workloads, intervals, config_ids, configs, proxies, labels = [], [], [], [], [], []
    for record in records:
        workload_id, interval_index, config_id = str(record["workload_id"]), int(record["interval_index"]), str(record["config_id"])
        identity = (workload_id, interval_index, config_id)
        if identity in identities:
            raise ValueError(f"duplicate calibrator identity: {identity}")
        identities.add(identity)
        config = np.asarray(record["config"], dtype=np.float32)
        proxy = np.asarray(record["proxy_metrics"], dtype=np.float32)
        label = np.asarray(record["label"], dtype=np.float32)
        if config.shape != (len(configs_names),) or proxy.shape != (len(proxies_names),) or label.shape != (len(outputs),):
            raise ValueError("calibrator record has invalid vector widths")
        if not (np.isfinite(config).all() and np.isfinite(proxy).all() and np.isfinite(label).all()):
            raise ValueError("calibrator records must be finite")
        if (proxy < 0.0).any() or (label < 0.0).any():
            raise ValueError("calibrator metrics must be nonnegative")
        workloads.append(workload_id); intervals.append(interval_index); config_ids.append(config_id)
        configs.append(config); proxies.append(proxy); labels.append(label)
    if require_reference and reference_config_id not in config_ids:
        raise ValueError("calibrator dataset does not contain the reference config")
    data = CalibratorDataset(
        reference_config_id, tuple(workloads), np.asarray(intervals, dtype=np.int64), tuple(config_ids), np.stack(configs),
        np.stack(proxies), np.stack(labels), configs_names, proxies_names, outputs,
    )
    if require_reference:
        _validate_reference_coverage(data)
    if output_path is not None:
        _write_dataset(data, Path(output_path))
    return data


def train_calibrator(
    dataset: CalibratorDataset | str | Path, *, output_path: str | Path | None = None,
    epochs: int = DEFAULT_EPOCHS, learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY, sinkhorn_regularization: float = DEFAULT_SINKHORN_REGULARIZATION,
    sinkhorn_weight: float = DEFAULT_SINKHORN_WEIGHT, view_weight: float = DEFAULT_VIEW_WEIGHT,
    seed: int = 19,
) -> FittedCalibrator:
    data = _coerce_dataset(dataset)
    if epochs < 1 or learning_rate <= 0.0 or weight_decay < 0.0 or sinkhorn_regularization <= 0.0:
        raise ValueError("invalid calibrator training parameters")
    if sinkhorn_weight < 0.0 or view_weight < 0.0:
        raise ValueError("calibrator loss weights must be nonnegative")
    proxy_target = _target_proxy(data)
    features = np.concatenate((data.configs, np.log1p(data.proxy_metrics)), axis=1)
    feature_mean, feature_scale = standardize(features)
    output_mean, output_scale = standardize(np.log1p(data.labels))
    torch.manual_seed(seed)
    model = MultiHeadPeregrineModel(features.shape[1], (32, 16), data.output_names)
    for head in model.heads.values():
        torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x = torch.from_numpy(scale(features, feature_mean, feature_scale))
    base = torch.from_numpy(np.log1p(proxy_target))
    target = torch.from_numpy(np.log1p(data.labels))
    target_normalized = (target - torch.from_numpy(output_mean)) / torch.from_numpy(output_scale)
    reference = torch.as_tensor([config_id == data.reference_config_id for config_id in data.config_ids], dtype=torch.bool)
    groups = _groups(data)
    for _ in range(epochs):
        optimizer.zero_grad()
        predicted_log = base + model(x)
        anchor_loss = F.smooth_l1_loss(predicted_log[reference], target[reference])
        predicted_normalized = (predicted_log - torch.from_numpy(output_mean)) / torch.from_numpy(output_scale)
        sinkhorn_loss = _mean_sinkhorn(predicted_normalized, target_normalized, data.workload_ids, sinkhorn_regularization)
        view_loss = _view_consistency(predicted_log, groups)
        (anchor_loss + sinkhorn_weight * sinkhorn_loss + view_weight * view_loss).backward()
        optimizer.step()
    model.eval()
    fitted = FittedCalibrator(
        model, feature_mean, feature_scale, output_mean, output_scale,
        data.config_names, data.proxy_names, data.output_names, epochs,
    )
    if output_path is not None:
        _save(fitted, Path(output_path))
    return fitted


def predict_calibrator(
    fitted: FittedCalibrator | str | Path, dataset: CalibratorDataset | str | Path,
) -> tuple[CalibratorPrediction, ...]:
    model, data = _coerce_fitted(fitted), _coerce_dataset(dataset)
    if (model.config_names, model.proxy_names, model.output_names) != (data.config_names, data.proxy_names, data.output_names):
        raise ValueError("calibrator model and dataset inputs differ")
    features = np.concatenate((data.configs, np.log1p(data.proxy_metrics)), axis=1)
    with torch.inference_mode():
        residual = model.model(torch.from_numpy(scale(features, model.feature_mean, model.feature_scale))).numpy()
    prediction = np.maximum(np.expm1(np.log1p(_target_proxy(data)) + residual), 0.0).astype(np.float32)
    return tuple(
        CalibratorPrediction(workload, int(interval), config_id, value)
        for workload, interval, config_id, value in zip(data.workload_ids, data.interval_indices, data.config_ids, prediction, strict=True)
    )


def evaluate_calibrator(dataset: CalibratorDataset | str | Path) -> dict[str, Any]:
    data = _coerce_dataset(dataset)
    return {
        "workload_ood": _evaluate_axis(data, axis="workload"),
        "config_ood": _evaluate_axis(data, axis="config"),
    }


def _evaluate_axis(data: CalibratorDataset, *, axis: str) -> dict[str, Any]:
    groups = tuple(sorted(set(data.workload_ids if axis == "workload" else (item for item in data.config_ids if item != data.reference_config_id))))
    if len(groups) < 2:
        raise ValueError(f"calibrator {axis} evaluation needs at least two held-out groups")
    proxy_reports, calibrated_reports = [], []
    values = data.workload_ids if axis == "workload" else data.config_ids
    for held_out in groups:
        test = np.asarray([value == held_out for value in values])
        train = ~test
        train_data, test_data = _subset(data, train), _subset(data, test, require_reference=False)
        fitted = train_calibrator(train_data)
        prediction = np.stack([item.prediction for item in predict_calibrator(fitted, test_data)])
        proxy_reports.append(regression_error_report(data.output_names, test_data.labels, _target_proxy(test_data)))
        calibrated_reports.append(regression_error_report(data.output_names, test_data.labels, prediction))
    return {"folds": len(groups), "proxy": _mean_reports(proxy_reports), "calibrated": _mean_reports(calibrated_reports)}


def _mean_sinkhorn(predicted: torch.Tensor, target: torch.Tensor, workload_ids: tuple[str, ...], regularization: float) -> torch.Tensor:
    values = []
    for workload_id in sorted(set(workload_ids)):
        mask = torch.as_tensor([item == workload_id for item in workload_ids], dtype=torch.bool)
        values.append(_sinkhorn_divergence(predicted[mask], target[mask], regularization))
    return torch.stack(values).mean()


def _sinkhorn_divergence(source: torch.Tensor, target: torch.Tensor, regularization: float) -> torch.Tensor:
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1] or not len(source) or not len(target):
        raise ValueError("Sinkhorn inputs must be nonempty matrices with equal widths")
    def cost(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        weights_lhs = torch.full((len(lhs),), 1.0 / len(lhs), dtype=lhs.dtype, device=lhs.device)
        weights_rhs = torch.full((len(rhs),), 1.0 / len(rhs), dtype=rhs.dtype, device=rhs.device)
        return ot.sinkhorn2(weights_lhs, weights_rhs, torch.cdist(lhs, rhs).square(), regularization, method="sinkhorn_log")
    return cost(source, target) - 0.5 * cost(source, source) - 0.5 * cost(target, target)


def _view_consistency(predicted_log: torch.Tensor, groups: tuple[tuple[int, ...], ...]) -> torch.Tensor:
    values = [predicted_log[list(indices)].var(dim=0, unbiased=False).mean() for indices in groups if len(indices) > 1]
    return torch.stack(values).mean() if values else predicted_log.new_zeros(())


def _groups(data: CalibratorDataset) -> tuple[tuple[int, ...], ...]:
    grouped: dict[tuple[str, int], list[int]] = {}
    for index, identity in enumerate(zip(data.workload_ids, data.interval_indices, strict=True)):
        grouped.setdefault((identity[0], int(identity[1])), []).append(index)
    return tuple(tuple(indices) for _, indices in sorted(grouped.items()))


def _target_proxy(data: CalibratorDataset) -> np.ndarray:
    positions = tuple(data.proxy_names.index(name) for name in data.output_names)
    return data.proxy_metrics[:, positions]


def _validate_reference_coverage(data: CalibratorDataset) -> None:
    expected = {(workload, int(interval)) for workload, interval in zip(data.workload_ids, data.interval_indices, strict=True)}
    observed = {(workload, int(interval)) for workload, interval, config_id in zip(data.workload_ids, data.interval_indices, data.config_ids, strict=True) if config_id == data.reference_config_id}
    if observed != expected:
        raise ValueError("every calibrator workload/window requires one reference proxy point")


def _write_dataset(data: CalibratorDataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"workload_id": workload, "interval_index": int(interval), "config_id": config_id,
         "config": config.tolist(), "proxy_metrics": proxy.tolist(), "label": label.tolist()}
        for workload, interval, config_id, config, proxy, label in zip(
            data.workload_ids, data.interval_indices, data.config_ids, data.configs, data.proxy_metrics, data.labels, strict=True
        )
    ]
    table = pa.Table.from_pylist(rows).replace_schema_metadata({
        b"calibrator_config": ",".join(data.config_names).encode(),
        b"calibrator_proxy": ",".join(data.proxy_names).encode(),
        b"calibrator_outputs": ",".join(data.output_names).encode(),
        b"calibrator_reference_config_id": data.reference_config_id.encode(),
    })
    pq.write_table(table, path, compression="zstd")


def _read_dataset(path: Path) -> CalibratorDataset:
    table = pq.read_table(path); metadata = table.schema.metadata or {}
    required = (b"calibrator_config", b"calibrator_proxy", b"calibrator_outputs", b"calibrator_reference_config_id")
    if any(not metadata.get(key) for key in required):
        raise ValueError("Parquet file is not a calibrator dataset")
    return build_calibrator_dataset(
        table.to_pylist(), config_names=tuple(metadata[b"calibrator_config"].decode().split(",")),
        proxy_names=tuple(metadata[b"calibrator_proxy"].decode().split(",")),
        output_names=tuple(metadata[b"calibrator_outputs"].decode().split(",")),
        reference_config_id=metadata[b"calibrator_reference_config_id"].decode(),
    )


def _save(fitted: FittedCalibrator, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": fitted.model.state_dict(), "input_size": fitted.model.trunk[0].in_features,
                "config_names": fitted.config_names, "proxy_names": fitted.proxy_names, "outputs": fitted.output_names,
                "feature_mean": fitted.feature_mean, "feature_scale": fitted.feature_scale,
                "output_mean": fitted.output_mean, "output_scale": fitted.output_scale, "epochs": fitted.epochs}, path)


def _load(path: Path) -> FittedCalibrator:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    outputs = tuple(saved["outputs"])
    model = MultiHeadPeregrineModel(int(saved["input_size"]), (32, 16), outputs)
    model.load_state_dict(saved["state"]); model.eval()
    return FittedCalibrator(model, saved["feature_mean"], saved["feature_scale"], saved["output_mean"], saved["output_scale"], tuple(saved["config_names"]), tuple(saved["proxy_names"]), outputs, int(saved["epochs"]))


def _coerce_dataset(value: CalibratorDataset | str | Path) -> CalibratorDataset:
    return value if isinstance(value, CalibratorDataset) else _read_dataset(Path(value))


def _coerce_fitted(value: FittedCalibrator | str | Path) -> FittedCalibrator:
    return value if isinstance(value, FittedCalibrator) else _load(Path(value))


def _subset(data: CalibratorDataset, mask: np.ndarray, *, require_reference: bool = True) -> CalibratorDataset:
    rows = [
        {"workload_id": workload, "interval_index": int(interval), "config_id": config_id,
         "config": config, "proxy_metrics": proxy, "label": label}
        for workload, interval, config_id, config, proxy, label, keep in zip(
            data.workload_ids, data.interval_indices, data.config_ids, data.configs, data.proxy_metrics, data.labels, mask, strict=True
        ) if keep
    ]
    return build_calibrator_dataset(
        rows, config_names=data.config_names, proxy_names=data.proxy_names,
        output_names=data.output_names, reference_config_id=data.reference_config_id, require_reference=require_reference,
    )


def _mean_reports(reports: Sequence[dict[str, dict[str, float | int | None]]]) -> dict[str, dict[str, float | None]]:
    return {name: {metric: float(np.mean([report[name][metric] for report in reports if report[name][metric] is not None])) if any(report[name][metric] is not None for report in reports) else None for metric in reports[0][name] if metric != "samples"} for name in reports[0]}
