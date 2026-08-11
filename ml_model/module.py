"""The canonical training, checkpoint, and inference implementation."""

from __future__ import annotations

from collections.abc import Sequence

import lightning as L
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from .error_metrics import regression_metrics
from .model import BOUNDED_HEAD, CONSTANT_ZERO_HEAD, POSITIVE_HEAD, ZERO_INFLATED_HEAD, SurrogateNetwork
from .tasks import TargetSpec

_EPSILON = 1e-6


def fit_normalization(features: np.ndarray, labels: np.ndarray, targets: Sequence[TargetSpec]) -> dict[str, np.ndarray]:
    """Fit preprocessing on a training partition only."""
    if features.ndim != 2 or labels.shape != (len(features), len(targets)):
        raise ValueError("surrogate normalization received incompatible matrices")
    target_mean, target_scale = _target_normalization(labels, targets)
    feature = StandardScaler().fit(features)
    return {
        "feature_mean": np.asarray(feature.mean_, dtype=np.float32),
        "feature_scale": np.asarray(feature.scale_, dtype=np.float32),
        "target_mean": target_mean, "target_scale": target_scale,
    }


def _target_normalization(labels: np.ndarray, targets: Sequence[TargetSpec]) -> tuple[np.ndarray, np.ndarray]:
    mean, scale = np.zeros(len(targets), dtype=np.float32), np.ones(len(targets), dtype=np.float32)
    for index, spec in enumerate(targets):
        if spec.head_kind == CONSTANT_ZERO_HEAD:
            continue
        values = _encode_numpy(labels[:, index], spec.head_kind)
        if spec.head_kind == ZERO_INFLATED_HEAD:
            values = values[labels[:, index] > 0.0]
        if len(values):
            fitted = StandardScaler().fit(values.reshape(-1, 1))
            mean[index], scale[index] = np.float32(fitted.mean_[0]), np.float32(fitted.scale_[0])
    return mean, scale


class SurrogateModule(L.LightningModule):
    def __init__(self, *, feature_columns: Sequence[str], targets: Sequence[TargetSpec | dict[str, str]], hidden_dims: Sequence[int], learning_rate: float, weight_decay: float, feature_mean: Sequence[float], feature_scale: Sequence[float], target_mean: Sequence[float], target_scale: Sequence[float], dropout: float = 0.0) -> None:
        super().__init__()
        feature_columns = tuple(feature_columns)
        hidden_dims = tuple(int(value) for value in hidden_dims)
        feature_mean = tuple(float(value) for value in feature_mean)
        feature_scale = tuple(float(value) for value in feature_scale)
        target_mean = tuple(float(value) for value in target_mean)
        target_scale = tuple(float(value) for value in target_scale)
        targets = tuple({"metric": item.metric if isinstance(item, TargetSpec) else item["metric"], "label_column": item.label_column if isinstance(item, TargetSpec) else item["label_column"], "head_kind": item.head_kind if isinstance(item, TargetSpec) else item["head_kind"], "primary_metric": item.primary_metric if isinstance(item, TargetSpec) else item["primary_metric"]} for item in targets)
        self.save_hyperparameters()
        self.feature_columns = feature_columns
        self.targets = tuple(TargetSpec(**item) for item in targets)
        if not self.feature_columns or not self.targets:
            raise ValueError("surrogate module requires features and targets")
        feature_width = len(feature_mean)
        self.network = SurrogateNetwork(
            feature_width,
            hidden_dims, {item.metric: item.head_kind for item in self.targets}, dropout=float(dropout),
        )
        self.register_buffer("feature_mean", _vector(feature_mean, feature_width, "feature_mean"))
        self.register_buffer("feature_scale", _positive_vector(feature_scale, feature_width, "feature_scale"))
        self.register_buffer("target_mean", _vector(target_mean, len(self.targets), "target_mean"))
        self.register_buffer("target_scale", _positive_vector(target_scale, len(self.targets), "target_scale"))
        self.learning_rate, self.weight_decay = float(learning_rate), float(weight_decay)
        self.validation_metrics = nn.ModuleDict({target.metric: regression_metrics(target.metric) for target in self.targets})

    @property
    def output_metrics(self) -> tuple[str, ...]: return tuple(item.metric for item in self.targets)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != len(self.feature_mean):
            raise ValueError("surrogate feature matrix has incompatible columns")
        latent = self._latent(features)
        return torch.cat([self._physical_prediction(spec, latent[spec.metric], index) for index, spec in enumerate(self.targets)], dim=1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        features, labels = batch; loss = self._loss(features, labels)
        self.log("train/loss", loss, on_epoch=True, batch_size=len(features)); return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> None:
        features, labels = batch; loss = self._loss(features, labels); prediction = self(features)
        for index, target in enumerate(self.targets): self.validation_metrics[target.metric].update(prediction[:, index], labels[:, index])
        self.log("val/loss", loss, on_epoch=True, batch_size=len(features))

    def on_validation_epoch_end(self) -> None:
        for target in self.targets:
            for name, value in self.validation_metrics[target.metric].compute().items(): self.log(f"val/{target.metric}/{name}", value)
            self.validation_metrics[target.metric].reset()

    def configure_optimizers(self) -> torch.optim.Optimizer: return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def _loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        latent, losses = self._latent(features), []
        for index, spec in enumerate(self.targets):
            target, output = labels[:, index:index + 1], latent[spec.metric]
            if spec.head_kind == CONSTANT_ZERO_HEAD: continue
            if spec.head_kind == ZERO_INFLATED_HEAD:
                event, magnitude = output  # type: ignore[misc]
                event_loss = F.binary_cross_entropy_with_logits(event, (target > 0).to(target.dtype)); positive = target.reshape(-1) > 0
                magnitude_loss = F.smooth_l1_loss(magnitude.reshape(-1)[positive], self._normalize(_encode_tensor(target.reshape(-1)[positive], spec.head_kind), index).reshape(-1)) if positive.any() else event_loss.new_zeros(())
                losses.append(0.5 * (event_loss + magnitude_loss))
            else: losses.append(F.smooth_l1_loss(output, self._normalize(_encode_tensor(target, spec.head_kind), index)))  # type: ignore[arg-type]
        return torch.stack(losses).mean() if losses else features.sum() * 0.0

    def _physical_prediction(self, spec: TargetSpec, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor], index: int) -> torch.Tensor:
        if spec.head_kind == CONSTANT_ZERO_HEAD: return torch.zeros_like(output)  # type: ignore[arg-type]
        if spec.head_kind == ZERO_INFLATED_HEAD:
            event, magnitude = output  # type: ignore[misc]
            return torch.sigmoid(event) * torch.exp(torch.clamp(self._denormalize(magnitude, index), max=20.0))
        decoded = self._denormalize(output, index)  # type: ignore[arg-type]
        if spec.head_kind == POSITIVE_HEAD: return torch.exp(torch.clamp(decoded, max=20.0))
        if spec.head_kind == BOUNDED_HEAD: return torch.sigmoid(decoded)
        raise ValueError(f"unknown surrogate head: {spec.head_kind}")

    def _latent(self, features: torch.Tensor) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        return self.network((features - self.feature_mean) / self.feature_scale)

    def _normalize(self, values: torch.Tensor, index: int) -> torch.Tensor: return (values - self.target_mean[index]) / self.target_scale[index]
    def _denormalize(self, values: torch.Tensor, index: int) -> torch.Tensor: return values * self.target_scale[index] + self.target_mean[index]


def _encode_numpy(values: np.ndarray, kind: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if kind in (POSITIVE_HEAD, ZERO_INFLATED_HEAD):
        if np.any(values < 0): raise ValueError("positive surrogate target is negative")
        return np.log(np.clip(values, _EPSILON, None))
    if kind == BOUNDED_HEAD:
        clipped = np.clip(values, _EPSILON, 1 - _EPSILON); return np.log(clipped / (1 - clipped))
    raise ValueError(f"unknown surrogate head: {kind}")


def _encode_tensor(values: torch.Tensor, kind: str) -> torch.Tensor:
    if kind in (POSITIVE_HEAD, ZERO_INFLATED_HEAD): return torch.log(values.clamp_min(_EPSILON))
    if kind == BOUNDED_HEAD: return torch.logit(values.clamp(_EPSILON, 1 - _EPSILON))
    raise ValueError(f"unknown surrogate head: {kind}")


def _vector(values: Sequence[float], size: int, name: str) -> torch.Tensor:
    result = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    if len(result) != size or not torch.isfinite(result).all(): raise ValueError(f"invalid {name}")
    return result


def _positive_vector(values: Sequence[float], size: int, name: str) -> torch.Tensor:
    result = _vector(values, size, name)
    if torch.any(result <= 0): raise ValueError(f"{name} must be positive")
    return result
