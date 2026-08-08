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
from .tasks import ScalarChannel, TargetSpec

_EPSILON = 1e-6


def fit_normalization(features: np.ndarray, labels: np.ndarray, targets: Sequence[TargetSpec], *, ple_bins: int = 0, channel_width: int | None = None) -> dict[str, np.ndarray]:
    """Fit preprocessing on a training partition only.

    Cross-domain stats are already normalized during materialization; PLE knots
    are fitted only from positive training features.
    """
    if features.ndim != 2 or labels.shape != (len(features), len(targets)) or ple_bins < 0:
        raise ValueError("surrogate normalization received incompatible matrices")
    target_mean, target_scale = _target_normalization(labels, targets)
    if channel_width is None:
        feature = StandardScaler().fit(features)
        return {
            "feature_mean": np.asarray(feature.mean_, dtype=np.float32),
            "feature_scale": np.asarray(feature.scale_, dtype=np.float32),
            "target_mean": target_mean, "target_scale": target_scale,
            "feature_ple_boundaries": np.empty((0, 0), dtype=np.float32),
        }
    if channel_width < 1 or features.shape[1] != channel_width:
        raise ValueError("stats feature normalization width differs from schema")
    boundaries = np.empty((channel_width, ple_bins + 1), dtype=np.float32)
    for index in range(channel_width):
        positive = np.log(features[features[:, index] > 0.0, index])
        if len(positive):
            boundaries[index] = np.quantile(positive, np.linspace(0.0, 1.0, ple_bins + 1)).astype(np.float32)
        else:
            boundaries[index].fill(0.0)
    return {
        "feature_mean": np.zeros(channel_width, dtype=np.float32),
        "feature_scale": np.ones(channel_width, dtype=np.float32),
        "target_mean": target_mean, "target_scale": target_scale,
        "feature_ple_boundaries": boundaries,
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
    def __init__(self, *, feature_columns: Sequence[str], targets: Sequence[TargetSpec | dict[str, str]], hidden_dims: Sequence[int], learning_rate: float, weight_decay: float, feature_mean: Sequence[float], feature_scale: Sequence[float], target_mean: Sequence[float], target_scale: Sequence[float], feature_ple_boundaries: Sequence[Sequence[float]] = (), channel_schema: Sequence[ScalarChannel | dict[str, object]] = (), active_channel_indices: Sequence[int] = (), ple_bins: int = 0, dropout: float = 0.0) -> None:
        super().__init__()
        feature_columns = tuple(feature_columns)
        hidden_dims = tuple(int(value) for value in hidden_dims)
        feature_mean = tuple(float(value) for value in feature_mean)
        feature_scale = tuple(float(value) for value in feature_scale)
        target_mean = tuple(float(value) for value in target_mean)
        target_scale = tuple(float(value) for value in target_scale)
        feature_ple_boundaries = tuple(tuple(float(value) for value in row) for row in feature_ple_boundaries)
        targets = tuple({"metric": item.metric if isinstance(item, TargetSpec) else item["metric"], "label_column": item.label_column if isinstance(item, TargetSpec) else item["label_column"], "head_kind": item.head_kind if isinstance(item, TargetSpec) else item["head_kind"], "primary_metric": item.primary_metric if isinstance(item, TargetSpec) else item["primary_metric"]} for item in targets)
        channel_schema = tuple({"source_id": int(item.source_id if isinstance(item, ScalarChannel) else item["source_id"]), "position_id": int(item.position_id if isinstance(item, ScalarChannel) else item["position_id"]), "source": str(item.source if isinstance(item, ScalarChannel) else item["source"]), "subname": item.subname if isinstance(item, ScalarChannel) else item.get("subname"), "unit": str(item.unit if isinstance(item, ScalarChannel) else item["unit"])} for item in channel_schema)
        active_channel_indices = tuple(int(index) for index in active_channel_indices) or tuple(range(len(channel_schema)))
        if len(set(active_channel_indices)) != len(active_channel_indices) or any(index < 0 or index >= len(channel_schema) for index in active_channel_indices):
            raise ValueError("active scalar channel indices are invalid")
        self.save_hyperparameters()
        self.feature_columns = feature_columns
        self.channel_schema = tuple(ScalarChannel(**item) for item in channel_schema)
        self.active_channel_indices = active_channel_indices
        self.active_channels = tuple(self.channel_schema[index] for index in active_channel_indices)
        self.targets = tuple(TargetSpec(**item) for item in targets)
        self.ple_bins = int(ple_bins)
        if not self.feature_columns or not self.targets:
            raise ValueError("surrogate module requires features and targets")
        raw_width = len(self.active_channels)
        feature_width = raw_width if self.channel_schema else len(feature_mean)
        self.network = SurrogateNetwork(
            raw_width * 5 if self.channel_schema else feature_width,
            hidden_dims, {item.metric: item.head_kind for item in self.targets}, dropout=float(dropout),
        )
        self.register_buffer("feature_mean", _vector(feature_mean, feature_width, "feature_mean"))
        self.register_buffer("feature_scale", _positive_vector(feature_scale, feature_width, "feature_scale"))
        self.register_buffer("target_mean", _vector(target_mean, len(self.targets), "target_mean"))
        self.register_buffer("target_scale", _positive_vector(target_scale, len(self.targets), "target_scale"))
        boundaries = torch.as_tensor(feature_ple_boundaries, dtype=torch.float32)
        expected = (raw_width, self.ple_bins + 1) if self.channel_schema else (0, 0)
        if boundaries.numel() == 0 and expected == (0, 0): boundaries = boundaries.reshape(0, 0)
        if tuple(boundaries.shape) != expected or not torch.isfinite(boundaries).all():
            raise ValueError("invalid feature_ple_boundaries")
        self.register_buffer("feature_ple_boundaries", boundaries)
        if self.channel_schema:
            self.register_buffer("channel_source_ids", torch.tensor([item.source_id for item in self.active_channels], dtype=torch.long))
            self.register_buffer("channel_position_ids", torch.tensor([item.position_id for item in self.active_channels], dtype=torch.long))
            self.source_embedding = nn.Embedding(max(item.source_id for item in self.channel_schema) + 1, 8)
            self.position_embedding = nn.Embedding(max(item.position_id for item in self.channel_schema) + 1, 8)
            self.scalar_encoder = nn.Sequential(nn.Linear(self.ple_bins + 16, 4), nn.GELU())
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
        if not self.channel_schema: return self.network((features - self.feature_mean) / self.feature_scale)
        return self.network(self._channel_encoding(features).reshape(len(features), -1))

    def _channel_encoding(self, features: torch.Tensor) -> torch.Tensor:
        raw_width = len(self.active_channels); raw = features[:, :raw_width]
        positive = raw > 0
        logged = torch.where(positive, torch.log(raw.clamp_min(_EPSILON)), torch.zeros_like(raw))
        lower, upper = self.feature_ple_boundaries[:, :-1], self.feature_ple_boundaries[:, 1:]
        bins = ((logged.unsqueeze(-1) - lower) / (upper - lower).clamp_min(_EPSILON)).clamp(0.0, 1.0)
        location = torch.cat((self.source_embedding(self.channel_source_ids), self.position_embedding(self.channel_position_ids)), dim=-1).unsqueeze(0).expand(len(raw), -1, -1)
        magnitude = self.scalar_encoder(torch.cat((bins * positive.unsqueeze(-1), location), dim=-1)) * positive.unsqueeze(-1)
        return torch.cat((positive.to(raw.dtype).unsqueeze(-1), magnitude), dim=-1)

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
