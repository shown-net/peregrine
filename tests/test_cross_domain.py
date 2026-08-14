import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ml_model import cross_domain


def test_single_target_network_starts_at_zero_residual() -> None:
    network = cross_domain.CrossDomainNetwork(6, (8, 4), 0.0)

    output = network(torch.ones((3, 6)))

    assert output.shape == (3, 1)
    assert torch.count_nonzero(output) == 0


def test_positive_and_count_rate_corrections_round_trip() -> None:
    counts = np.asarray([100_000.0, 100_000.0], dtype=np.float32)
    positive = SimpleNamespace(residual="positive_ratio")
    count_rate = SimpleNamespace(residual="count_rate")
    for spec, anchor, truth in (
        (positive, np.asarray([2.0, 4.0], dtype=np.float32), np.asarray([3.0, 2.0], dtype=np.float32)),
        (count_rate, np.asarray([0.0, 2.0], dtype=np.float32), np.asarray([0.0, 4.0], dtype=np.float32)),
    ):
        correction = cross_domain._correction(spec, anchor, truth, counts)
        assert np.allclose(cross_domain._decode(spec, anchor, correction, counts), truth, rtol=1e-5, atol=1e-6)


def test_count_rate_anchor_feature_is_finite_at_zero() -> None:
    feature = cross_domain._anchor_feature(
        SimpleNamespace(residual="count_rate"),
        np.asarray([0.0, 2.0], dtype=np.float32),
        np.asarray([100_000.0, 100_000.0], dtype=np.float32),
    )

    assert np.isfinite(feature).all()


def test_evaluation_emits_workload_isolated_oof_and_diagnostic(tmp_path) -> None:
    workloads = np.repeat(np.asarray(["alpha", "beta", "gamma", "delta", "epsilon"]), 3)
    rows = len(workloads)
    features = [np.asarray([float(index % 3), float(index // 3)], dtype=np.float32) for index in range(rows)]
    anchors = [np.asarray([1.0, float(index % 2)], dtype=np.float32) for index in range(rows)]
    task = SimpleNamespace(
        targets=(
            SimpleNamespace(metric="CPI", residual="positive_ratio"),
            SimpleNamespace(metric="MEM_READ_PER_KI", residual="count_rate"),
        ),
        diagnostic_ids=("STALL_CYCLE_RATIO",),
        stats_width=2,
        hidden_dims=(4, 2),
        dropout=0.0,
        max_epochs=2,
        batch_size=8,
        learning_rate=0.01,
        weight_decay=0.0,
        num_threads=1,
        seed=7,
        evaluation_folds=5,
        early_stopping_patience=1,
    )
    dataset = tmp_path / "dataset.parquet"
    pq.write_table(pa.table({
        "workload_id": workloads,
        "prefix_index": np.tile(np.arange(1, 4), 5),
        "instruction_count": np.full(rows, 100_000),
        "stats_anchors": anchors,
        "stats_features": features,
        "CPI": 1.0 + np.arange(rows) / 100.0,
        "MEM_READ_PER_KI": np.arange(rows, dtype=float) % 3,
        "STALL_CYCLE_RATIO": np.full(rows, 0.25),
    }), dataset)

    result = cross_domain.evaluate_cross_domain(task=task, dataset_path=dataset, output_dir=tmp_path / "evaluation")
    oof = pq.read_table(result["oof_predictions"]).to_pandas()

    assert len(oof) == rows
    assert "truth_STALL_CYCLE_RATIO" in oof
    assert "anchor_CPI" in oof
    assert "mlp_CPI" in oof
    assert "selected_CPI" not in oof
    assert "ridge_CPI" not in oof


def test_evaluation_reports_canonical_continuous_and_sparse_summaries(tmp_path) -> None:
    workloads = np.repeat(np.asarray(["alpha", "beta", "gamma", "delta", "epsilon"]), 3)
    rows = len(workloads)
    features = [np.asarray([float(index % 3), float(index // 3)], dtype=np.float32) for index in range(rows)]
    anchors = [np.asarray([1.0, float(index % 2)], dtype=np.float32) for index in range(rows)]
    task = SimpleNamespace(
        targets=(
            SimpleNamespace(metric="CPI", residual="positive_ratio"),
            SimpleNamespace(metric="MEM_READ_PER_KI", residual="count_rate"),
        ),
        diagnostic_ids=(),
        stats_width=2,
        hidden_dims=(4, 2),
        dropout=0.0,
        max_epochs=2,
        batch_size=8,
        learning_rate=0.01,
        weight_decay=0.0,
        num_threads=1,
        seed=7,
        evaluation_folds=5,
        early_stopping_patience=1,
    )
    dataset = tmp_path / "dataset.parquet"
    pq.write_table(pa.table({
        "workload_id": workloads,
        "prefix_index": np.tile(np.arange(1, 4), 5),
        "instruction_count": np.full(rows, 100_000),
        "stats_anchors": anchors,
        "stats_features": features,
        "CPI": 1.0 + np.arange(rows) / 100.0,
        "MEM_READ_PER_KI": np.arange(rows, dtype=float) % 3,
    }), dataset)

    result = cross_domain.evaluate_cross_domain(task=task, dataset_path=dataset, output_dir=tmp_path / "evaluation")
    report = json.loads(Path(result["evaluation"]).read_text(encoding="utf-8"))

    cpi = report["metrics"]["CPI"]
    assert cpi["metric_kind"] == "continuous"
    assert set(cpi["models"]) == {"anchor", "mlp"}
    assert set(cpi["models"]["anchor"]) == {"smape_pct", "wape_pct"}
    assert set(cpi["models"]["mlp"]) == {"smape_pct", "wape_pct"}
    assert "smape_pct_delta" in cpi["mlp_vs_anchor"]
    sparse = report["metrics"]["MEM_READ_PER_KI"]
    assert sparse["metric_kind"] == "sparse"
    assert set(sparse["models"]) == {"anchor", "mlp"}
    assert set(sparse["models"]["anchor"]) == {
        "average_precision", "positive_wape_pct", "smape_pct", "wape_pct",
        "zero_pred_abs_p50", "zero_pred_abs_p95",
    }
    assert set(sparse["models"]["mlp"]) == set(sparse["models"]["anchor"])
    assert "positive_wape_pct_delta" in sparse["mlp_vs_anchor"]
    assert report["derived_metrics"]["IPC"]["metric_kind"] == "continuous"


def test_acceptance_reports_mlp_vs_anchor_improvement() -> None:
    continuous = SimpleNamespace(metric="CPI")
    sparse = SimpleNamespace(metric="MISS_MPKI")
    metrics = {
        "CPI": {
            "metric_kind": "continuous",
            "truth_degenerate": False,
            "models": {
                "anchor": {"smape_pct": 10.0, "wape_pct": 11.0},
                "mlp": {"smape_pct": 8.0, "wape_pct": 9.0},
            },
            "mlp_vs_anchor": {"smape_pct_delta": -2.0, "wape_pct_delta": -2.0},
        },
        "MISS_MPKI": {
            "metric_kind": "sparse",
            "truth_degenerate": False,
            "models": {
                "anchor": {"average_precision": 0.8, "positive_wape_pct": 50.0, "smape_pct": 60.0, "wape_pct": 55.0},
                "mlp": {"average_precision": 0.7, "positive_wape_pct": 30.0, "smape_pct": 70.0, "wape_pct": 35.0},
            },
            "mlp_vs_anchor": {"positive_wape_pct_delta": -20.0, "smape_pct_delta": 10.0, "wape_pct_delta": -20.0},
        },
    }
    fold_metrics = {
        "CPI": [metrics["CPI"]],
        "MISS_MPKI": [metrics["MISS_MPKI"]],
    }

    acceptance = cross_domain._acceptance(
        SimpleNamespace(targets=(continuous, sparse)),
        metrics,
        fold_metrics,
    )

    assert acceptance["improved_targets"] == ["CPI", "MISS_MPKI"]
    assert acceptance["regressed_targets"] == []
    assert acceptance["degenerate_targets"] == []
    assert acceptance["model"] == "mlp"
    assert acceptance["baseline"] == "anchor"
    assert acceptance["passed"] is True


def test_metric_summary_reports_mlp_vs_anchor_delta() -> None:
    summary = cross_domain._metric_summary(
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        {
            "anchor": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            "mlp": np.asarray([9.0, 18.0, 27.0], dtype=np.float32),
        },
        np.asarray(["a", "b", "c"]),
    )

    assert set(summary["models"]) == {"anchor", "mlp"}
    assert summary["mlp_vs_anchor"]["smape_pct_delta"] > 0.0


def test_acceptance_reports_degenerate_targets_without_counting_them() -> None:
    target = SimpleNamespace(metric="L3_ACCESSES_PER_KI")
    metrics = {
        "L3_ACCESSES_PER_KI": {
            "metric_kind": "continuous",
            "truth_degenerate": True,
            "models": {
                "anchor": {"smape_pct": 100.0, "wape_pct": 100.0},
                "mlp": {"smape_pct": 0.0, "wape_pct": 0.0},
            },
            "mlp_vs_anchor": {"smape_pct_delta": -100.0, "wape_pct_delta": -100.0},
        }
    }

    acceptance = cross_domain._acceptance(
        SimpleNamespace(targets=(target,)),
        metrics,
        {"L3_ACCESSES_PER_KI": [metrics["L3_ACCESSES_PER_KI"]]},
    )

    assert acceptance["degenerate_targets"] == ["L3_ACCESSES_PER_KI"]
    assert acceptance["improved_targets"] == []
    assert acceptance["passed"] is False
