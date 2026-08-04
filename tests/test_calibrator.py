from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ml_model.calibrator import _GaussianCalibratorModel, _ot_covariances, _paired_w2_squared, _stabilize_covariances
from ml_model.calibrator import build_calibrator_dataset, build_calibrator_inputs, evaluate_calibrator, predict_calibrator, train_calibrator


def _records() -> list[dict[str, object]]:
    return [
        {"workload_id": workload, "interval_index": interval, "config_id": config_id,
         "config": np.asarray([config], np.float32), "proxy_metrics": np.asarray([1.0 + interval + offset], np.float32)}
        for workload, offset in (("a", 0.0), ("b", 0.2))
        for interval in range(2)
        for config_id, config in (("baseline", 0.0), ("near", 1.0))
    ]


def _targets() -> list[dict[str, object]]:
    return [
        {"workload_id": workload, "interval_index": interval,
         "mean": np.asarray([1.2 + offset + interval], np.float32),
         "sample_covariance": np.asarray([[0.03]], np.float32)}
        for workload, offset in (("a", 0.0), ("b", 0.2)) for interval in range(2)
    ]


def _write_target(path, records: list[dict[str, object]]) -> None:
    pq.write_table(
        pa.Table.from_pylist([
            {"workload_id": row["workload_id"], "interval_index": row["interval_index"],
             "mean": row["mean"].tolist(), "sample_covariance": row["sample_covariance"].tolist()}
            for row in records
        ]).replace_schema_metadata({b"calibrator_target_metrics": b"CPI"}),
        path,
    )


def test_gaussian_w2_is_zero_for_matching_moments_and_backpropagates() -> None:
    mean = torch.tensor([[1.0]], requires_grad=True)
    target_mean = torch.tensor([[1.0]])
    factor = torch.tensor([[[0.2]]], requires_grad=True)
    covariance = factor @ factor.transpose(-1, -2)
    target_covariance = torch.tensor([[[0.04]]])

    loss = _paired_w2_squared(mean, covariance, target_mean, target_covariance).sum()

    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(factor.grad).all()


def test_stabilized_covariance_makes_rank_deficient_target_safe_for_pot() -> None:
    covariance = torch.tensor([[[1.0, 0.0], [0.0, -1e-18]]])

    stabilized = _stabilize_covariances(covariance)

    assert torch.linalg.eigvalsh(stabilized).min() > 0.0


def test_ot_covariance_adds_positive_distinct_numerical_axes() -> None:
    adjusted = _ot_covariances(torch.zeros((1, 3, 3)))

    assert torch.linalg.eigvalsh(adjusted).min() > 0.0
    assert len(torch.unique(torch.diagonal(adjusted, dim1=-2, dim2=-1))) == 3


def test_initial_covariance_head_has_finite_bures_gradients() -> None:
    model = _GaussianCalibratorModel(1, 2)
    mean, covariance = model(torch.zeros((1, 1)), torch.zeros((1, 2)))

    _paired_w2_squared(
        mean, covariance, torch.zeros((1, 2)),
        _stabilize_covariances(torch.tensor([[[0.2, 0.0], [0.0, 0.4]]])),
    ).sum().backward()

    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_covariance_head_keeps_extreme_factor_logits_numerically_finite() -> None:
    model = _GaussianCalibratorModel(1, 2)
    with torch.no_grad():
        model.covariance_factor.bias.fill_(1e6)

    _, covariance = model(torch.zeros((1, 1)), torch.zeros((1, 2)))

    assert torch.isfinite(covariance).all()
    assert torch.linalg.eigvalsh(covariance).max() < 1e4


def test_calibrator_persists_interval_predictions(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_target(dataset / "target.parquet", _targets())
    data = build_calibrator_dataset(
        _records(), target_records=_targets(), config_names=("config",),
        proxy_names=("CPI",), output_names=("CPI",), reference_config_id="baseline", output_path=dataset,
    )

    fitted = train_calibrator(data, epochs=3, ot_weight=0.0, output_path=tmp_path / "model.pt")
    direct = predict_calibrator(fitted, data)
    reloaded = predict_calibrator(tmp_path / "model.pt", dataset)

    assert len(direct) == len(data.workload_ids)
    assert np.allclose(np.stack([item.mean for item in direct]), np.stack([item.mean for item in reloaded]))
    assert np.all(np.stack([item.lower for item in direct]) <= np.stack([item.mean for item in direct]))
    assert np.all(np.stack([item.mean for item in direct]) <= np.stack([item.upper for item in direct]))
    inputs = build_calibrator_inputs(_records(), config_names=("config",), proxy_names=("CPI",))
    assert len(predict_calibrator(fitted, inputs)) == len(direct)


def test_dataset_requires_sample_covariance_not_mean_covariance() -> None:
    target = _targets()
    target[0] = {"workload_id": "a", "interval_index": 0, "mean": np.asarray([1.2], np.float32), "covariance": np.asarray([[0.001]], np.float32)}

    try:
        build_calibrator_dataset(_records(), target_records=target, config_names=("config",), proxy_names=("CPI",), output_names=("CPI",), reference_config_id="baseline")
    except KeyError as error:
        assert "sample_covariance" in str(error)
    else:
        raise AssertionError("legacy mean covariance target was accepted")


def test_nonreference_gmm_ot_trains_with_finite_interval_outputs() -> None:
    data = build_calibrator_dataset(
        _records(), target_records=_targets(), config_names=("config",),
        proxy_names=("CPI",), output_names=("CPI",), reference_config_id="baseline",
    )

    fitted = train_calibrator(data, epochs=1, ot_weight=0.03)
    rows = predict_calibrator(fitted, data)

    assert np.isfinite(np.stack([row.mean for row in rows])).all()
    assert np.isfinite(np.stack([row.lower for row in rows])).all()
    assert np.isfinite(np.stack([row.upper for row in rows])).all()


def test_evaluation_reports_workload_config_and_joint_ood() -> None:
    records = [
        {**row, "config_id": config_id, "config": np.asarray([config], np.float32)}
        for row in _records()
        for config_id, config in (("baseline", 0.0), ("c0", 1.0), ("c1", 2.0))
        if row["config_id"] == "baseline"
    ]
    data = build_calibrator_dataset(
        records, target_records=_targets(), config_names=("config",),
        proxy_names=("CPI",), output_names=("CPI",), reference_config_id="baseline",
    )

    report = evaluate_calibrator(data, ot_weight=0.0, folds=2)

    assert {"workload_ood", "config_ood", "joint_ood"} <= set(report)
    assert "gaussian_w2" in report["joint_ood"]
