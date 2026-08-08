from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch

from ml_model.inference import _validate_object_schema
from ml_model.model import POSITIVE_HEAD, ZERO_INFLATED_HEAD
from ml_model.module import SurrogateModule, fit_normalization
from ml_model.prediction import _fit_task, _predict_array
from ml_model.tasks import StatsFeatureObject, SurrogateTask, TargetSpec


def test_normalized_stats_ple_produces_physical_predictions() -> None:
    objects = (
        StatsFeatureObject("cycles", "point", (), (), "cycles_per_instruction"),
        StatsFeatureObject("classes", "vector", (2,), (("alu", "load"),), "per_ki"),
    )
    features = np.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    targets = (TargetSpec("CPI", "CPI", POSITIVE_HEAD, "mae"), TargetSpec("MISS", "MISS", ZERO_INFLATED_HEAD, "mae"))
    labels = np.asarray([[1.0, 0.0], [2.0, 1.0]], dtype=np.float32)
    module = SurrogateModule(
        feature_columns=("cycles", "classes"), targets=targets, hidden_dims=(8, 4),
        learning_rate=0.01, weight_decay=0.0, object_schema=objects, ple_bins=4,
        **fit_normalization(features, labels, targets, ple_bins=4, object_feature_width=3),
    )

    values = module(torch.from_numpy(features)).detach().numpy()

    assert values.shape == labels.shape
    assert np.isfinite(values).all()
    assert (values >= 0.0).all()


def test_stats_inference_rejects_schema_mismatch() -> None:
    objects = (StatsFeatureObject("cycles", "point", (), (), "cycles_per_instruction"),)
    target = (TargetSpec("CPI", "CPI", POSITIVE_HEAD, "mae"),)
    module = SurrogateModule(
        feature_columns=("cycles",), targets=target, hidden_dims=(8, 4), learning_rate=0.01,
        weight_decay=0.0, object_schema=objects, ple_bins=2,
        **fit_normalization(np.asarray([[1.0], [2.0]], dtype=np.float32), np.asarray([[1.0], [2.0]], dtype=np.float32), target, ple_bins=2, object_feature_width=1),
    )
    schema = pa.schema([pa.field("cycles", pa.float64(), metadata={
        b"stats_kind": b"point", b"stats_shape": b"[]", b"stats_fields": b"[]", b"stats_unit": b"per_ki",
    })])

    with pytest.raises(ValueError, match="schema differs"):
        _validate_object_schema(schema, module)


def test_task_preserves_configured_target_heads() -> None:
    task = SurrogateTask(
        identity_columns=("workload_id",), group_column="workload_id", feature_columns=("f",),
        targets=(TargetSpec("P", "P", POSITIVE_HEAD, "mae"), TargetSpec("Z", "Z", ZERO_INFLATED_HEAD, "mae")),
        hidden_dims=(8, 4), max_epochs=1, batch_size=2, learning_rate=0.01, weight_decay=0.0,
        early_stopping_patience=1, num_threads=1, seed=7, evaluation_folds=3, validation_fraction=0.25,
        object_schema=(StatsFeatureObject("f", "point", (), (), "per_ki"),), ple_bins=2,
    )
    fitted, _, _ = _fit_task(task, np.asarray([[1.0], [2.0]], dtype=np.float32), np.asarray([[1.0, 0.0], [2.0, 1.0]], dtype=np.float32), np.asarray([[1.0]], dtype=np.float32))

    assert tuple(target.head_kind for target in fitted.targets) == (POSITIVE_HEAD, ZERO_INFLATED_HEAD)


def test_prediction_array_rejects_non_finite_outputs() -> None:
    class BadModule(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.full((len(features), 1), float("nan"))

    with pytest.raises(ValueError, match="non-finite"):
        _predict_array(BadModule(), np.ones((2, 1), dtype=np.float32), batch_size=2)
