"""Domain-owned feature providers expressed through the common prediction task API."""

from __future__ import annotations

from anamol.python.design_space import PeregrineConfig

from .multitask import MultiHeadTraining
from .prediction import FeatureSet
from .prediction import PredictionTask
SURROGATE_TASK_ID = "surrogate"


def surrogate_task(config: PeregrineConfig) -> PredictionTask:
    features = tuple(config.feature_columns)
    return PredictionTask(
        task_id=SURROGATE_TASK_ID,
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
