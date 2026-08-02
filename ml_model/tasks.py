"""Domain-owned feature providers expressed through the common prediction task API."""

from __future__ import annotations

from anamol.python.design_space import PeregrineConfig

from .multitask import MultiHeadTraining
from .prediction import FeatureSet
from .prediction import PredictionTask
from .real_anchor import P2_FEATURES
from .real_anchor import target_names


L1_TASK_ID = "l1-surrogate"
L3_TASK_ID = "l3-real-anchor"


def l1_task(config: PeregrineConfig) -> PredictionTask:
    features = tuple(config.feature_columns)
    return PredictionTask(
        task_id=L1_TASK_ID,
        identity_columns=("workload_id", "window_index", "config_id"),
        group_column="config_id",
        feature_set=FeatureSet("trace_design", features),
        label_columns=tuple(config.label_columns),
        output_metrics=tuple(label.removeprefix("label_") for label in config.label_columns),
        training=MultiHeadTraining(
            config.training.hidden_dims, config.training.max_epochs, config.training.batch_size,
            config.training.learning_rate, config.training.weight_decay,
            config.training.early_stopping_patience,
        ),
        num_threads=config.training.num_threads or 1, seed=config.training.seed,
        evaluation_folds=config.evaluation.config_folds,
    )


def l3_task() -> PredictionTask:
    targets = target_names()
    return PredictionTask(
        task_id=L3_TASK_ID,
        identity_columns=("workload_id", "interval_index"), group_column="workload_id",
        feature_set=FeatureSet("p2", P2_FEATURES),
        label_columns=tuple(f"label_{name}" for name in targets), output_metrics=targets,
        training=MultiHeadTraining((32, 16), 200, 64, 1e-3, 1e-3, 20),
        num_threads=24, seed=42,
        data_limitations=(
            "PMU metric groups are aligned by interval_index in the current historical corpus; "
            "they do not yet carry a shared collector-run sample identity.",
        ),
    )
