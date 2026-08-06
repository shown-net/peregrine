from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch

from ml_model.inference import predict_parquet
from ml_model.error_metrics import compute_regression_metrics
from ml_model.model import BOUNDED_HEAD, POSITIVE_HEAD, ZERO_INFLATED_HEAD
from ml_model.module import SurrogateModule, fit_normalization
from ml_model.prediction import (
    compare_oof_evaluations,
    evaluate_random_pair_surrogate,
    evaluate_surrogate,
    _predict_array,
    train_surrogate,
)
from ml_model.tasks import SurrogateTask, TargetSpec


def _task(*, epochs: int = 2) -> SurrogateTask:
    return SurrogateTask(
        identity_columns=("workload_id", "window_index", "config_id"), group_column="workload_id",
        feature_columns=("f0", "f1"),
        targets=(
            TargetSpec("CPI", "label_CPI", POSITIVE_HEAD, "mape_pct"),
            TargetSpec("BRANCH_RATE", "label_BRANCH_RATE", BOUNDED_HEAD, "mae"),
            TargetSpec("BRANCH_MPKI", "label_BRANCH_MPKI", ZERO_INFLATED_HEAD, "mae"),
        ),
        hidden_dims=(8, 4), max_epochs=epochs, batch_size=4, learning_rate=0.01,
        weight_decay=0.0, early_stopping_patience=1, num_threads=1, seed=7,
        evaluation_folds=3, validation_fraction=0.25,
    )


def _frame() -> pd.DataFrame:
    rows = []
    for workload in range(6):
        for window in range(3):
            rows.append({
                "workload_id": f"w{workload}", "window_index": window, "config_id": "c0",
                "f0": float(workload), "f1": float(window), "label_CPI": float(workload + window + 1),
                "label_BRANCH_RATE": float((workload + window + 1) / 10.0),
                "label_BRANCH_MPKI": float((workload + window) % 3),
            })
    return pd.DataFrame(rows)


def test_fixed_heads_decode_to_their_physical_domains() -> None:
    task = _task()
    frame = _frame()
    features = frame.loc[:, task.feature_columns].to_numpy(dtype=np.float32)
    labels = frame.loc[:, task.label_columns].to_numpy(dtype=np.float32)
    module = SurrogateModule(
        feature_columns=task.feature_columns, targets=task.targets, hidden_dims=task.hidden_dims,
        learning_rate=task.learning_rate, weight_decay=task.weight_decay,
        **fit_normalization(features, labels, task.targets),
    )
    values = module(torch.from_numpy(features.copy())).detach().numpy()
    assert (values[:, 0] >= 0.0).all()
    assert ((values[:, 1] >= 0.0) & (values[:, 1] <= 1.0)).all()
    assert (values[:, 2] >= 0.0).all()


def test_zero_inflated_head_accepts_all_zero_training_partition() -> None:
    task = SurrogateTask(
        identity_columns=("workload_id", "window_index", "config_id"), group_column="workload_id",
        feature_columns=("f0",),
        targets=(TargetSpec("TLB", "label_TLB", ZERO_INFLATED_HEAD, "mae"),),
        hidden_dims=(8, 4), max_epochs=1, batch_size=4, learning_rate=0.01,
        weight_decay=0.0, early_stopping_patience=1, num_threads=1, seed=7,
        evaluation_folds=3, validation_fraction=0.25,
    )
    features = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
    labels = np.zeros((3, 1), dtype=np.float32)

    module = SurrogateModule(
        feature_columns=task.feature_columns, targets=task.targets, hidden_dims=task.hidden_dims,
        learning_rate=task.learning_rate, weight_decay=task.weight_decay,
        **fit_normalization(features, labels, task.targets),
    )
    values = module(torch.from_numpy(features.copy())).detach().numpy()

    assert np.isfinite(values).all()
    assert (values[:, 0] >= 0.0).all()


def test_prediction_array_rejects_non_finite_outputs() -> None:
    class BadModule(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.full((len(features), 1), float("nan"))

    with pytest.raises(ValueError, match="non-finite"):
        _predict_array(BadModule(), np.ones((2, 1), dtype=np.float32), batch_size=2)


def test_train_checkpoint_drives_streaming_prediction(tmp_path: Path) -> None:
    task = _task(epochs=1)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _frame().to_parquet(dataset / "samples.parquet", index=False)
    training = train_surrogate(task=task, dataset_dir=dataset, output_dir=tmp_path / "model")
    output = tmp_path / "prediction.parquet"
    report = predict_parquet(
        checkpoint_path=training["checkpoint"], features_path=dataset / "samples.parquet", output_path=output,
        batch_size=2,
    )
    table = pq.read_table(output)
    assert report["rows"] == len(_frame())
    assert table.column_names == [
        "workload_id", "window_index", "config_id", "prediction_CPI",
        "prediction_BRANCH_RATE", "prediction_BRANCH_MPKI",
    ]


def test_grouped_oof_does_not_fit_heldout_workload_labels(tmp_path: Path) -> None:
    task = _task(epochs=1)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    baseline = _frame()
    changed = baseline.copy()
    changed.loc[changed.workload_id == "w0", "label_CPI"] += 1000.0
    baseline.to_parquet(first / "samples.parquet", index=False)
    changed.to_parquet(second / "samples.parquet", index=False)
    first_result = evaluate_surrogate(task=task, dataset_dir=first, output_dir=tmp_path / "first-evaluation")
    second_result = evaluate_surrogate(task=task, dataset_dir=second, output_dir=tmp_path / "second-evaluation")
    left = pd.read_parquet(first_result["oof_predictions"])
    right = pd.read_parquet(second_result["oof_predictions"])
    assert np.allclose(
        left.loc[left.workload_id == "w0", "prediction_CPI"],
        right.loc[right.workload_id == "w0", "prediction_CPI"],
    )


def test_official_metric_contract_is_target_aware_and_grouped(tmp_path: Path) -> None:
    task = _task(epochs=1)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _frame().to_parquet(dataset / "samples.parquet", index=False)
    result = evaluate_surrogate(task=task, dataset_dir=dataset, output_dir=tmp_path / "evaluation")
    report = json.loads(Path(result["evaluation"]).read_text())
    metrics = report["metrics"]
    assert set(metrics) == {"macro_workload", "roi_weighted", "per_workload"}
    assert "mape_pct" in metrics["roi_weighted"]["CPI"]
    assert "mape_pct" in metrics["roi_weighted"]["BRANCH_RATE"]
    assert "mape_pct" not in metrics["roi_weighted"]["BRANCH_MPKI"]
    assert "p90_absolute_error" not in metrics["roi_weighted"]["CPI"]
    predictions = pd.read_parquet(result["oof_predictions"])
    expected = compute_regression_metrics(
        "CPI",
        torch.tensor(predictions["prediction_CPI"].to_numpy()),
        torch.tensor(predictions["truth_CPI"].to_numpy()),
    )
    assert metrics["roi_weighted"]["CPI"] == expected


def test_oof_comparison_recomputes_all_aggregations(tmp_path: Path) -> None:
    task = _task(epochs=1)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    baseline = _frame()
    baseline.to_parquet(first / "samples.parquet", index=False)
    baseline.to_parquet(second / "samples.parquet", index=False)
    historical = evaluate_surrogate(task=task, dataset_dir=first, output_dir=tmp_path / "historical")
    current = evaluate_surrogate(task=task, dataset_dir=second, output_dir=tmp_path / "current")
    result = compare_oof_evaluations(
        task=task, historical_oof_path=historical["oof_predictions"], current_oof_path=current["oof_predictions"],
        output_path=tmp_path / "comparison.json",
    )
    comparison = json.loads(Path(result["comparison"]).read_text())
    assert comparison["samples"] == len(baseline)
    assert set(comparison["metrics"]) == {"macro_workload", "roi_weighted", "per_workload"}
    assert "mape_pct" not in comparison["metrics"]["roi_weighted"]["BRANCH_MPKI"]


def test_random_pair_evaluation_is_reproducible_and_reports_only_test_rows(tmp_path: Path) -> None:
    task = _task(epochs=1)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _frame().to_parquet(dataset / "samples.parquet", index=False)
    first = evaluate_random_pair_surrogate(task=task, dataset_dir=dataset, output_dir=tmp_path / "first")
    second = evaluate_random_pair_surrogate(task=task, dataset_dir=dataset, output_dir=tmp_path / "second")
    report = json.loads(Path(first["evaluation"]).read_text())
    left = pd.read_parquet(first["test_predictions"])
    right = pd.read_parquet(second["test_predictions"])
    assert report["protocol"] == "random_pair_split"
    assert report["generalization_scope"] == "in_distribution_random_pair"
    assert sum(report["split"][name] for name in ("train_rows", "validation_rows", "test_rows")) == len(_frame())
    assert len(left) == report["split"]["test_rows"]
    assert left.equals(right)
    assert set(report["metrics"]) == {"macro_workload", "roi_weighted", "per_workload"}
