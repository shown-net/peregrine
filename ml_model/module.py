"""The one training, checkpoint, and inference implementation for surrogate metrics."""

from __future__ import annotations

from collections.abc import Sequence

import lightning as L
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from .model import BOUNDED_HEAD, POSITIVE_HEAD, ZERO_INFLATED_HEAD, SurrogateNetwork
from .error_metrics import regression_metrics
from .tasks import TargetSpec

_EPSILON = 1e-6


def fit_normalization(features: np.ndarray, labels: np.ndarray, targets: Sequence[TargetSpec]) -> dict[str, np.ndarray]:
    """Fit the only preprocessing used by a training partition."""
    if features.ndim != 2 or labels.shape != (len(features), len(targets)):
        raise ValueError("surrogate normalization received incompatible matrices")
    feature = StandardScaler().fit(features)
    target_mean = np.zeros(len(targets), dtype=np.float32)
    target_scale = np.ones(len(targets), dtype=np.float32)
    for index, spec in enumerate(targets):
        values = _encode_numpy(labels[:, index], spec.head_kind)
        if spec.head_kind == ZERO_INFLATED_HEAD:
            values = values[labels[:, index] > 0.0]
            if not len(values):
                target_mean[index] = np.float32(0.0)
                target_scale[index] = np.float32(1.0)
                continue
        scaler = StandardScaler().fit(values.reshape(-1, 1))
        target_mean[index] = np.float32(scaler.mean_[0])
        target_scale[index] = np.float32(scaler.scale_[0])
    return {
        "feature_mean": np.asarray(feature.mean_, dtype=np.float32),
        "feature_scale": np.asarray(feature.scale_, dtype=np.float32),
        "target_mean": target_mean,
        "target_scale": target_scale,
    }


class SurrogateModule(L.LightningModule):
    """Shared surrogate network with fixed physical-domain prediction heads."""

    def __init__(
        self,
        *,
        feature_columns: Sequence[str],
        targets: Sequence[TargetSpec | dict[str, str]],
        hidden_dims: Sequence[int],
        learning_rate: float,
        weight_decay: float,
        feature_mean: Sequence[float],
        feature_scale: Sequence[float],
        target_mean: Sequence[float],
        target_scale: Sequence[float],
    ) -> None:
        super().__init__()
        feature_columns = tuple(feature_columns)
        targets = tuple({
            "metric": item.metric if isinstance(item, TargetSpec) else item["metric"],
            "label_column": item.label_column if isinstance(item, TargetSpec) else item["label_column"],
            "head_kind": item.head_kind if isinstance(item, TargetSpec) else item["head_kind"],
            "primary_metric": item.primary_metric if isinstance(item, TargetSpec) else item["primary_metric"],
        } for item in targets)
        hidden_dims = tuple(int(item) for item in hidden_dims)
        feature_mean = tuple(float(item) for item in feature_mean)
        feature_scale = tuple(float(item) for item in feature_scale)
        target_mean = tuple(float(item) for item in target_mean)
        target_scale = tuple(float(item) for item in target_scale)
        self.save_hyperparameters()
        self.feature_columns = feature_columns
        self.targets = tuple(TargetSpec(**item) for item in targets)
        if not self.feature_columns or not self.targets:
            raise ValueError("surrogate module requires features and targets")
        self.network = SurrogateNetwork(
            len(self.feature_columns), hidden_dims, {item.metric: item.head_kind for item in self.targets}
        )
        self.register_buffer("feature_mean", _vector(feature_mean, len(self.feature_columns), "feature_mean"))
        self.register_buffer("feature_scale", _positive_vector(feature_scale, len(self.feature_columns), "feature_scale"))
        self.register_buffer("target_mean", _vector(target_mean, len(self.targets), "target_mean"))
        self.register_buffer("target_scale", _positive_vector(target_scale, len(self.targets), "target_scale"))
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.validation_metrics = nn.ModuleDict({
            target.metric: regression_metrics(target.metric) for target in self.targets
        })

    @property
    def output_metrics(self) -> tuple[str, ...]:
        return tuple(item.metric for item in self.targets)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != len(self.feature_columns):
            raise ValueError("surrogate feature matrix has incompatible columns")
        latent = self.network((features - self.feature_mean) / self.feature_scale)
        values = [self._physical_prediction(spec, latent[spec.metric], index) for index, spec in enumerate(self.targets)]
        return torch.cat(values, dim=1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _batch_index: int) -> torch.Tensor:
        features, labels = batch
        loss = self._loss(features, labels)
        self.log("train/loss", loss, on_epoch=True, prog_bar=False, batch_size=len(features))
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _batch_index: int) -> None:
        features, labels = batch
        loss = self._loss(features, labels)
        prediction = self(features)
        for index, target in enumerate(self.targets):
            self.validation_metrics[target.metric].update(prediction[:, index], labels[:, index])
        self.log("val/loss", loss, on_epoch=True, prog_bar=False, batch_size=len(features))

    def on_validation_epoch_end(self) -> None:
        for target in self.targets:
            values = self.validation_metrics[target.metric].compute()
            for name, value in values.items():
                self.log(f"val/{target.metric}/{name}", value, prog_bar=False)
            self.validation_metrics[target.metric].reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def _loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        latent = self.network((features - self.feature_mean) / self.feature_scale)
        losses: list[torch.Tensor] = []
        for index, spec in enumerate(self.targets):
            target = labels[:, index:index + 1]
            output = latent[spec.metric]
            if spec.head_kind == ZERO_INFLATED_HEAD:
                event, magnitude = output  # type: ignore[misc]
                event_loss = F.binary_cross_entropy_with_logits(event, (target > 0.0).to(target.dtype))
                positive = target.reshape(-1) > 0.0
                if positive.any():
                    transformed = self._normalize(_encode_tensor(target[positive], spec.head_kind), index).reshape(-1)
                    magnitude_loss = F.smooth_l1_loss(magnitude.reshape(-1)[positive], transformed)
                else:
                    magnitude_loss = event_loss.new_zeros(())
                losses.append(event_loss + magnitude_loss)
            else:
                transformed = self._normalize(_encode_tensor(target, spec.head_kind), index)
                losses.append(F.smooth_l1_loss(output, transformed))  # type: ignore[arg-type]
        return torch.stack(losses).mean()

    def _physical_prediction(
        self, spec: TargetSpec, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor], index: int
    ) -> torch.Tensor:
        if spec.head_kind == ZERO_INFLATED_HEAD:
            event, magnitude = output  # type: ignore[misc]
            decoded = torch.clamp(self._denormalize(magnitude, index), max=20.0)
            return torch.sigmoid(event) * torch.clamp_min(torch.expm1(decoded), 0.0)
        decoded = self._denormalize(output, index)  # type: ignore[arg-type]
        if spec.head_kind == POSITIVE_HEAD:
            return torch.clamp_min(torch.expm1(torch.clamp(decoded, max=20.0)), 0.0)
        if spec.head_kind == BOUNDED_HEAD:
            return torch.sigmoid(decoded)
        raise ValueError(f"unknown surrogate head: {spec.head_kind}")

    def _normalize(self, values: torch.Tensor, index: int) -> torch.Tensor:
        return (values - self.target_mean[index]) / self.target_scale[index]

    def _denormalize(self, values: torch.Tensor, index: int) -> torch.Tensor:
        return values * self.target_scale[index] + self.target_mean[index]


def _encode_numpy(values: np.ndarray, kind: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if kind in (POSITIVE_HEAD, ZERO_INFLATED_HEAD):
        if np.any(values < 0.0):
            raise ValueError("positive surrogate target is negative")
        return np.log1p(values)
    if kind == BOUNDED_HEAD:
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("bounded surrogate target is outside [0, 1]")
        clipped = np.clip(values, _EPSILON, 1.0 - _EPSILON)
        return np.log(clipped / (1.0 - clipped))
    raise ValueError(f"unknown surrogate head: {kind}")


def _encode_tensor(values: torch.Tensor, kind: str) -> torch.Tensor:
    if kind in (POSITIVE_HEAD, ZERO_INFLATED_HEAD):
        return torch.log1p(values)
    if kind == BOUNDED_HEAD:
        clipped = values.clamp(_EPSILON, 1.0 - _EPSILON)
        return torch.logit(clipped)
    raise ValueError(f"unknown surrogate head: {kind}")


def _vector(values: Sequence[float], size: int, name: str) -> torch.Tensor:
    vector = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    if len(vector) != size or not torch.isfinite(vector).all():
        raise ValueError(f"invalid {name}")
    return vector


def _positive_vector(values: Sequence[float], size: int, name: str) -> torch.Tensor:
    vector = _vector(values, size, name)
    if torch.any(vector <= 0.0):
        raise ValueError(f"{name} must be positive")
    return vector
