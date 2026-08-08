from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pytest
import torch

from ml_model.inference import _feature_matrix, _validate_channel_schema
from ml_model.model import POSITIVE_HEAD, ZERO_INFLATED_HEAD
from ml_model.module import SurrogateModule, fit_normalization
from ml_model.prediction import _fit_task, _predict_array
from ml_model.tasks import ScalarChannel, SurrogateTask, TargetSpec


def _channels() -> tuple[ScalarChannel, ...]:
    return (
        ScalarChannel(0, 0, "cycles", None, "cycles_per_instruction"),
        ScalarChannel(1, 0, "classes", "alu", "per_ki"),
        ScalarChannel(1, 1, "classes", "load", "per_ki"),
    )


def test_scalar_channels_combine_value_and_location_encoding() -> None:
    channels = _channels()
    features = np.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    targets = (TargetSpec("CPI", "CPI", POSITIVE_HEAD, "mae"), TargetSpec("MISS", "MISS", ZERO_INFLATED_HEAD, "mae"))
    labels = np.asarray([[1.0, 0.0], [2.0, 1.0]], dtype=np.float32)
    module = SurrogateModule(
        feature_columns=("stats_values",), targets=targets, hidden_dims=(8, 4), learning_rate=0.01,
        weight_decay=0.0, channel_schema=channels, ple_bins=4,
        **fit_normalization(features, labels, targets, ple_bins=4, channel_width=len(channels)),
    )

    values = module(torch.from_numpy(features)).detach().numpy()

    assert values.shape == labels.shape
    assert np.isfinite(values).all()
    assert (values >= 0.0).all()


def test_zero_channel_has_presence_without_semantic_magnitude() -> None:
    channels = _channels()
    target = (TargetSpec("CPI", "CPI", POSITIVE_HEAD, "mae"),)
    features = np.asarray([[1.0, 0.0, 3.0], [2.0, 0.0, 4.0]], dtype=np.float32)
    module = SurrogateModule(
        feature_columns=("stats_values",), targets=target, hidden_dims=(8, 4), learning_rate=0.01,
        weight_decay=0.0, channel_schema=channels, active_channel_indices=(0, 1, 2), ple_bins=2,
        **fit_normalization(features, np.asarray([[1.0], [2.0]], dtype=np.float32), target, ple_bins=2, channel_width=3),
    )

    encoded = module._channel_encoding(torch.from_numpy(features))

    assert torch.equal(encoded[:, 1, 0], torch.zeros(2))
    assert torch.equal(encoded[:, 1, 1:], torch.zeros((2, 4)))


def test_scalar_inference_rejects_schema_mismatch() -> None:
    channels = _channels()
    target = (TargetSpec("CPI", "CPI", POSITIVE_HEAD, "mae"),)
    module = SurrogateModule(
        feature_columns=("stats_values",), targets=target, hidden_dims=(8, 4), learning_rate=0.01,
        weight_decay=0.0, channel_schema=channels, active_channel_indices=(0, 2), ple_bins=2,
        **fit_normalization(np.asarray([[1.0, 3.0]], dtype=np.float32), np.asarray([[1.0]], dtype=np.float32), target, ple_bins=2, channel_width=2),
    )
    raw = [item.__dict__ for item in channels]
    raw[-1] = {**raw[-1], "position_id": 2}
    schema = pa.schema([pa.field("stats_values", pa.list_(pa.float64(), len(channels)), metadata={b"cross_domain.channel_schema": json.dumps(raw, sort_keys=True).encode()})])

    with pytest.raises(ValueError, match="schema differs"):
        _validate_channel_schema(schema, module)


def test_inference_selects_checkpoint_active_channels() -> None:
    channels = _channels()
    target = (TargetSpec("CPI", "CPI", POSITIVE_HEAD, "mae"),)
    module = SurrogateModule(
        feature_columns=("stats_values",), targets=target, hidden_dims=(8, 4), learning_rate=0.01,
        weight_decay=0.0, channel_schema=channels, active_channel_indices=(0, 2), ple_bins=2,
        **fit_normalization(np.asarray([[1.0, 3.0]], dtype=np.float32), np.asarray([[1.0]], dtype=np.float32), target, ple_bins=2, channel_width=2),
    )
    batch = pa.RecordBatch.from_arrays([pa.array([[1.0, 2.0, 3.0]], type=pa.list_(pa.float64(), 3))], names=["stats_values"])

    assert np.array_equal(_feature_matrix(batch, module), [[1.0, 3.0]])


def test_task_preserves_configured_target_heads() -> None:
    task = SurrogateTask(
        identity_columns=("workload_id",), group_column="workload_id", feature_columns=("stats_values",),
        targets=(TargetSpec("P", "P", POSITIVE_HEAD, "mae"), TargetSpec("Z", "Z", ZERO_INFLATED_HEAD, "mae")),
        hidden_dims=(8, 4), max_epochs=1, batch_size=2, learning_rate=0.01, weight_decay=0.0,
        early_stopping_patience=1, num_threads=1, seed=7, evaluation_folds=3, validation_fraction=0.25,
        channel_schema=_channels(), ple_bins=2,
    )
    fitted, train_x, valid_x = _fit_task(task, np.asarray([[1.0, 0.0, 5.0], [2.0, 0.0, 5.0]], dtype=np.float32), np.asarray([[1.0, 0.0], [2.0, 1.0]], dtype=np.float32), np.asarray([[3.0, 0.0, 5.0]], dtype=np.float32))
    assert tuple(target.head_kind for target in fitted.targets) == (POSITIVE_HEAD, ZERO_INFLATED_HEAD)
    assert fitted.active_channel_indices == (0,)
    assert train_x.shape[1] == valid_x.shape[1] == 1


def test_prediction_array_rejects_non_finite_outputs() -> None:
    class BadModule(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.full((len(features), 1), float("nan"))

    with pytest.raises(ValueError, match="non-finite"):
        _predict_array(BadModule(), np.ones((2, 1), dtype=np.float32), batch_size=2)
