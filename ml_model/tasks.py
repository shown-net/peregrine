"""The canonical surrogate task contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from anamol.python.design_space import PeregrineConfig

from .model import BOUNDED_HEAD, POSITIVE_HEAD, ZERO_INFLATED_HEAD


@dataclass(frozen=True)
class TargetSpec:
    metric: str
    label_column: str
    head_kind: str | None
    primary_metric: str


@dataclass(frozen=True)
class ScalarChannel:
    source_id: int
    position_id: int
    source: str
    subname: str | None
    unit: str


@dataclass(frozen=True)
class SurrogateTask:
    identity_columns: tuple[str, ...]
    group_column: str
    feature_columns: tuple[str, ...]
    targets: tuple[TargetSpec, ...]
    hidden_dims: tuple[int, int]
    max_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    num_threads: int
    seed: int
    evaluation_folds: int
    validation_fraction: float
    channel_schema: tuple[ScalarChannel, ...] = ()
    active_channel_indices: tuple[int, ...] = ()
    ple_bins: int = 0
    dropout: float = 0.0
    checkpoint_name: str = "surrogate.ckpt"

    @property
    def label_columns(self) -> tuple[str, ...]:
        return tuple(item.label_column for item in self.targets)

    @property
    def output_metrics(self) -> tuple[str, ...]:
        return tuple(item.metric for item in self.targets)


def surrogate_task(config: PeregrineConfig) -> SurrogateTask:
    return SurrogateTask(
        identity_columns=("workload_id", "window_index", "config_id"), group_column="workload_id",
        feature_columns=tuple(config.feature_columns), targets=tuple(_target(label) for label in config.label_columns),
        hidden_dims=config.training.hidden_dims, max_epochs=config.training.max_epochs,
        batch_size=config.training.batch_size, learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay, early_stopping_patience=config.training.early_stopping_patience,
        num_threads=config.training.num_threads or 1, seed=config.training.seed,
        evaluation_folds=config.evaluation.config_folds, validation_fraction=config.training.paper_test_fraction,
    )


def cross_domain_task(metrics: object, config: PeregrineConfig, *, channel_schema: Sequence[object]) -> SurrogateTask:
    channels = tuple(ScalarChannel(int(item.source_id), int(item.position_id), str(item.source), item.subname, str(item.unit)) for item in channel_schema)
    if not channels:
        raise ValueError("cross-domain task needs a scalar channel schema")
    targets = tuple(
        TargetSpec(item.metric_id, item.metric_id, item.head, "mae")
        for item in (getattr(metrics, "pmu_targets")[metric_id] for metric_id in getattr(metrics, "metric_ids"))
    )
    return SurrogateTask(
        identity_columns=("workload_id", "interval_index"), group_column="workload_id",
        feature_columns=("stats_values",), targets=targets,
        hidden_dims=config.cross_domain.hidden_dims, max_epochs=config.training.max_epochs,
        batch_size=config.training.batch_size, learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay, early_stopping_patience=config.training.early_stopping_patience,
        num_threads=config.training.num_threads or 1, seed=config.training.seed,
        evaluation_folds=config.evaluation.config_folds, validation_fraction=config.training.paper_test_fraction,
        channel_schema=channels, ple_bins=config.cross_domain.ple_bins, dropout=config.cross_domain.dropout,
        checkpoint_name="cross_domain.ckpt",
    )


def _target(label_column: str) -> TargetSpec:
    metric = label_column.removeprefix("label_")
    if metric == "CPI":
        return TargetSpec(metric, label_column, POSITIVE_HEAD, "mape_pct")
    if metric == "BRANCH_RATE":
        return TargetSpec(metric, label_column, BOUNDED_HEAD, "mae")
    return TargetSpec(metric, label_column, ZERO_INFLATED_HEAD, "mae")
