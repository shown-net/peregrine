from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .feature_pipeline import analytical_feature_columns
from .gem5_stats import TraceLabelRegistry, load_label_registry
from .microarchitecture import MicroarchitectureConfig
from .run_config import RunConfig


WORKLOAD_CONTEXT_STATISTICS = ("mean", "p90", "std", "active_ratio")


@dataclass(frozen=True)
class AnalysisConfig:
    window_size: int


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int
    batch_size: int
    hidden_dims: tuple[int, int]
    paper_test_fraction: float
    seed: int
    learning_rate: float
    weight_decay: float
    num_threads: int | None
    early_stopping_patience: int


@dataclass(frozen=True)
class EvaluationConfig:
    config_folds: int


@dataclass(frozen=True)
class CrossDomainModelConfig:
    hidden_dims: tuple[int, int]
    dropout: float


@dataclass(frozen=True)
class CollectionSamplingConfig:
    seed: int
    configs_per_region: int


@dataclass(frozen=True)
class PeregrineConfig:
    microarchitecture: MicroarchitectureConfig
    analysis: AnalysisConfig
    collection_sampling: CollectionSamplingConfig
    labels: TraceLabelRegistry
    training: TrainingConfig
    evaluation: EvaluationConfig
    cross_domain: CrossDomainModelConfig

    @property
    def design_parameter_names(self) -> tuple[str, ...]:
        return self.microarchitecture.design_parameter_names

    @property
    def numeric_design_parameter_names(self) -> tuple[str, ...]:
        return self.microarchitecture.numeric_design_parameter_names

    @property
    def categorical_design_parameter_names(self) -> tuple[str, ...]:
        return self.microarchitecture.categorical_design_parameter_names

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return (
            *analytical_feature_columns(self.microarchitecture.mechanisms),
            *self.numeric_design_parameter_names,
            *self.categorical_feature_columns,
            *reciprocal_feature_columns(),
            *workload_context_feature_columns(self.microarchitecture.mechanisms),
        )

    @property
    def categorical_feature_columns(self) -> tuple[str, ...]:
        columns: list[str] = []
        for parameter_name in self.categorical_design_parameter_names:
            parameter = self.microarchitecture.parameters_by_name[parameter_name]
            columns.extend(
                f"{parameter_name}_{value}"
                for value in parameter.values
            )
        return tuple(columns)

    @property
    def label_columns(self) -> tuple[str, ...]:
        return tuple(label.label_column for label in self.labels.labels)

    def run_config_from_args(
        self,
        config_id: str,
        gem5_args: tuple[str, ...],
    ) -> RunConfig:
        return RunConfig(config_id, self.microarchitecture.parse_gem5_args(gem5_args), gem5_args)


def load_peregrine_config(path: str | Path, *, metrics_config: str | Path, microarchitecture: MicroarchitectureConfig) -> PeregrineConfig:
    resolved = Path(path).resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Peregrine config must be a mapping: {resolved}")
    analysis = payload.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("Peregrine config must define analysis")
    collection = payload.get("collection")
    if not isinstance(collection, dict):
        raise ValueError("Peregrine config must define collection")
    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("Peregrine config must define training")
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("Peregrine config must define evaluation")
    cross_domain = payload.get("cross_domain")
    if not isinstance(cross_domain, dict):
        raise ValueError("Peregrine config must define cross_domain")
    hidden_dims = tuple(int(value) for value in training["hidden_dims"])
    if len(hidden_dims) != 2:
        raise ValueError("training.hidden_dims must contain two values")
    num_threads = training.get("num_threads")
    labels = payload.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("Peregrine config must define labels")
    label_metric_set = str(labels["metric_set"])
    config = PeregrineConfig(
        microarchitecture=microarchitecture,
        analysis=AnalysisConfig(window_size=int(analysis["window_size"])),
        collection_sampling=CollectionSamplingConfig(
            seed=int(collection["seed"]),
            configs_per_region=int(collection["configs_per_region"]),
        ),
        labels=load_label_registry(metrics_config, metric_set_id=label_metric_set),
        training=TrainingConfig(
            max_epochs=int(training["max_epochs"]),
            batch_size=int(training["batch_size"]),
            hidden_dims=(hidden_dims[0], hidden_dims[1]),
            paper_test_fraction=float(training["paper_test_fraction"]),
            seed=int(training["seed"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            num_threads=None if num_threads is None else int(num_threads),
            early_stopping_patience=int(training["early_stopping_patience"]),
        ),
        evaluation=EvaluationConfig(
            config_folds=int(evaluation["config_folds"]),
        ),
        cross_domain=CrossDomainModelConfig(
            hidden_dims=tuple(int(value) for value in cross_domain["hidden_dims"]),
            dropout=float(cross_domain["dropout"]),
        ),
    )
    _validate_config(config)
    return config


def reciprocal_feature_columns() -> tuple[str, ...]:
    return ("inv_rob_size", "inv_lq_entries", "inv_sq_entries")


def workload_context_feature_columns(mechanisms) -> tuple[str, ...]:
    return tuple(
        f"workload_context_{statistic}__{mechanism.name}"
        for mechanism in mechanisms
        for statistic in WORKLOAD_CONTEXT_STATISTICS
    )


def _validate_config(config: PeregrineConfig) -> None:
    if config.analysis.window_size < 1:
        raise ValueError("analysis.window_size must be positive")
    if config.collection_sampling.configs_per_region < 1:
        raise ValueError("collection.configs_per_region must be positive")
    if config.training.max_epochs < 1:
        raise ValueError("training.max_epochs must be positive")
    if config.training.early_stopping_patience < 1:
        raise ValueError("training.early_stopping_patience must be positive")
    if config.training.weight_decay < 0.0:
        raise ValueError("training.weight_decay must not be negative")
    if config.evaluation.config_folds < 3:
        raise ValueError("evaluation.config_folds must be at least three")
    if (
        len(config.cross_domain.hidden_dims) != 2
        or min(config.cross_domain.hidden_dims) < 1
        or not 0.0 <= config.cross_domain.dropout < 1.0
    ):
        raise ValueError("cross_domain model configuration is invalid")
    for name, value in (("paper_test_fraction", config.training.paper_test_fraction),):
        if not 0.0 < value < 1.0:
            raise ValueError(f"training.{name} must be between zero and one")
