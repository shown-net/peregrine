from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
import torch

from ml_model.inference import _feature_matrix
from ml_model.model import POSITIVE_HEAD, ZERO_INFLATED_HEAD
from ml_model.module import SurrogateModule, fit_normalization
from ml_model.prediction import _fit_task, _predict_array
from ml_model.tasks import SurrogateTask, TargetSpec


def _target() -> tuple[TargetSpec, ...]:
    return (TargetSpec("CPI", "CPI", POSITIVE_HEAD, "mae"),)


def test_surrogate_module_consumes_named_scalar_columns() -> None:
    features = np.asarray([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    labels = np.asarray([[1.0], [2.0]], dtype=np.float32)
    module = SurrogateModule(
        feature_columns=("first", "second"),
        targets=_target(),
        hidden_dims=(8, 4),
        learning_rate=0.01,
        weight_decay=0.0,
        **fit_normalization(features, labels, _target()),
    )

    values = module(torch.from_numpy(features)).detach().numpy()

    assert values.shape == labels.shape
    assert np.isfinite(values).all()
    assert (values >= 0.0).all()


def test_inference_reads_checkpoint_feature_columns() -> None:
    module = SurrogateModule(
        feature_columns=("first", "second"),
        targets=_target(),
        hidden_dims=(8, 4),
        learning_rate=0.01,
        weight_decay=0.0,
        **fit_normalization(
            np.asarray([[1.0, 2.0]], dtype=np.float32),
            np.asarray([[1.0]], dtype=np.float32),
            _target(),
        ),
    )
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1.0]), pa.array([2.0])],
        names=["first", "second"],
    )

    assert np.array_equal(_feature_matrix(batch, module), [[1.0, 2.0]])


def test_task_preserves_configured_target_heads() -> None:
    task = SurrogateTask(
        identity_columns=("workload_id",),
        group_column="workload_id",
        feature_columns=("first", "constant"),
        targets=(
            TargetSpec("P", "P", POSITIVE_HEAD, "mae"),
            TargetSpec("Z", "Z", ZERO_INFLATED_HEAD, "mae"),
        ),
        hidden_dims=(8, 4),
        max_epochs=1,
        batch_size=2,
        learning_rate=0.01,
        weight_decay=0.0,
        early_stopping_patience=1,
        num_threads=1,
        seed=7,
        evaluation_folds=3,
        validation_fraction=0.25,
    )
    fitted, train_x, valid_x = _fit_task(
        task,
        np.asarray([[1.0, 5.0], [2.0, 5.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0], [2.0, 1.0]], dtype=np.float32),
        np.asarray([[3.0, 5.0]], dtype=np.float32),
    )
    assert tuple(target.head_kind for target in fitted.targets) == (
        POSITIVE_HEAD,
        ZERO_INFLATED_HEAD,
    )
    assert fitted.feature_columns == ("first",)
    assert train_x.shape[1] == valid_x.shape[1] == 1


def test_prediction_array_rejects_non_finite_outputs() -> None:
    class BadModule(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.full((len(features), 1), float("nan"))

    with pytest.raises(ValueError, match="non-finite"):
        _predict_array(BadModule(), np.ones((2, 1), dtype=np.float32), batch_size=2)
