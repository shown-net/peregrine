from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ml_model.calibrator import build_calibrator_dataset
from ml_model.calibrator import build_calibrator_inputs
from ml_model.calibrator import heldout_calibrator_predictions
from ml_model.calibrator import load_calibrator
from ml_model.calibrator import load_calibrator_dataset
from ml_model.calibrator import predict_calibrator
from ml_model.calibrator import train_calibrator
from ml_model.calibrator import write_calibrator_source


def _dataset():
    source = []
    targets = []
    for workload_index, workload in enumerate(("work_a", "work_b", "work_c")):
        for interval in range(12):
            x = 1.0 + interval / 10.0
            source.append({"workload_id": workload, "interval_index": interval, "source_proxy_values": [x, .5 + workload_index, 3.0 + interval]})
            targets.append({"workload_id": workload, "interval_index": interval, "mean": [x + workload_index / 10.0, .4 + workload_index / 10.0], "measurement_covariance": [[.01, .002], [.002, .02]]})
    return build_calibrator_dataset(source, target_records=targets, source_names=("sim_cycles_per_inst", "sim_l1d_per_inst", "duplicate_free"), target_names=("CPU_CYCLES", "L1D_CACHE_REFILL_RD"), target_anchor_period=100_000)


def _sparse_dataset():
    source = []
    targets = []
    for workload_index, workload in enumerate(("work_a", "work_b", "work_c", "work_d")):
        for interval in range(24):
            signal = 1.0 + workload_index + interval / 24.0
            sparse = 0.0 if interval < 16 else signal / 10.0
            source.append({"workload_id": workload, "interval_index": interval, "source_proxy_values": [signal, 0.0, interval + 1.0]})
            targets.append({"workload_id": workload, "interval_index": interval, "mean": [signal, sparse], "measurement_covariance": [[.01, 0.0], [0.0, .02]]})
    return build_calibrator_dataset(source, target_records=targets, source_names=("signal", "constant_zero", "interval"), target_names=("DENSE_EVENT", "SPARSE_EVENT"), target_anchor_period=100_000)


def test_calibrator_persists_cross_run_mean_contract(tmp_path) -> None:
    data = _dataset()
    source_path = tmp_path / "dataset/source.parquet"; target_path = tmp_path / "target.parquet"
    write_calibrator_source(data, source_path)
    pq.write_table(pa.Table.from_pylist([{"workload_id": w, "interval_index": int(i), "mean": mean.tolist(), "measurement_covariance": covariance.tolist()} for w, i, mean, covariance in zip(data.target_workload_ids, data.target_interval_indices, data.target_means, data.target_measurement_covariances, strict=True)]).replace_schema_metadata({b"calibrator_target_events": b"CPU_CYCLES,L1D_CACHE_REFILL_RD", b"calibrator_target_anchor_period": b"100000", b"calibrator_target_provenance": b"cross_run_mean_vector"}), target_path)
    path = tmp_path / "calibrator.joblib"; fitted = train_calibrator(load_calibrator_dataset(source_path, target_path), output_path=path); loaded = load_calibrator(path)
    assert fitted.target_names == loaded.target_names
    assert len(predict_calibrator(loaded, data)) == len(data.source_workload_ids)


def test_sparse_torch_calibrator_marks_sparse_targets_and_drops_constant_sources() -> None:
    data = _sparse_dataset()
    fitted = train_calibrator(data)

    assert fitted.estimator.target_modes == ("dense", "sparse")
    assert fitted.estimator.active_source_names == ("signal", "interval")
    assert fitted.estimator.shrinkage == 0.0

    predictions = np.stack([row.mean for row in predict_calibrator(fitted, data)])
    assert predictions.shape == data.target_means.shape
    assert np.isfinite(predictions).all()
    assert np.all(predictions >= data.target_means.min(axis=0))
    assert np.all(predictions <= data.target_means.max(axis=0))


def test_dataset_aligns_target_rows_by_window_identity() -> None:
    source = [{"workload_id": "work", "interval_index": 0, "source_proxy_values": [1.0]}, {"workload_id": "work", "interval_index": 1, "source_proxy_values": [2.0]}]
    targets = [{"workload_id": "work", "interval_index": 1, "mean": [20.0], "measurement_covariance": [[.01]]}, {"workload_id": "work", "interval_index": 0, "mean": [10.0], "measurement_covariance": [[.01]]}]
    data = build_calibrator_dataset(source, target_records=targets, source_names=("source",), target_names=("target",), target_anchor_period=100_000)
    assert data.target_interval_indices.tolist() == [0, 1]
    assert data.target_means[:, 0].tolist() == [10.0, 20.0]


def test_evaluation_returns_lowo_event_predictions_for_domain_reporting() -> None:
    data = _dataset()
    heldout = heldout_calibrator_predictions(data)

    assert heldout.target_names == data.target_names
    assert heldout.workload_ids == data.target_workload_ids
    assert heldout.truth.shape == heldout.prediction.shape == heldout.baseline.shape
    assert np.isfinite(heldout.prediction).all()


def test_deployment_prediction_is_mean_only_and_bounded_by_training_targets() -> None:
    data = _dataset()
    prediction = predict_calibrator(train_calibrator(data), data)[0]
    assert prediction.mean.shape == (2,)
    assert np.all(np.isfinite(prediction.mean))
    assert np.all(prediction.mean >= data.target_means.min(axis=0))
    assert np.all(prediction.mean <= data.target_means.max(axis=0))


def test_prediction_rejects_a_different_source_schema() -> None:
    data = _dataset()
    inputs = build_calibrator_inputs([{"workload_id": "outside", "interval_index": 0, "source_proxy_values": [1.0, 1.0, 1.0]}], source_names=("wrong", "source", "schema"))
    with pytest.raises(ValueError, match="source proxy names differ"):
        predict_calibrator(train_calibrator(data), inputs)
