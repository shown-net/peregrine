from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_model.plots import plot_cross_domain_model_comparison
from ml_model.plots import plot_surrogate_generalization_errors


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "workload_id": ["cache__base", "cache__small", "branch__base", "branch__small", "branch__small"],
            "config_id": ["config_b", "config_a", "config_b", "config_a", "config_a"],
            "window_index": [0, 1, 0, 0, 1],
            "truth_CPI": [1.0, 2.0, 1.0, 2.0, 2.0],
            "prediction_CPI": [1.5, 1.0, 1.0, 3.0, 2.0],
            "truth_BRANCH_RATE": [0.2, 0.4, 0.1, 0.2, 0.2],
            "prediction_BRANCH_RATE": [0.1, 0.4, 0.2, 0.2, 0.3],
        }
    )


def _write_evaluation(root: Path, *, protocol: str, predictions_name: str) -> None:
    root.mkdir()
    payload = {"protocol": protocol}
    if protocol == "random_pair_split":
        payload["generalization_scope"] = "in_distribution_random_pair"
    (root / "evaluation.json").write_text(json.dumps(payload), encoding="utf-8")
    _predictions().to_parquet(root / predictions_name, index=False)


def test_generalization_error_plots_are_split_by_evaluation_and_metric(tmp_path: Path) -> None:
    workload_ood = tmp_path / "workload_ood_input"
    random_pair = tmp_path / "random_pair_input"
    _write_evaluation(workload_ood, protocol="grouped_workload_kfold", predictions_name="oof_predictions.parquet")
    _write_evaluation(random_pair, protocol="random_pair_split", predictions_name="test_predictions.parquet")

    plots = plot_surrogate_generalization_errors(
        workload_ood_dir=workload_ood,
        random_pair_dir=random_pair,
        output_dir=tmp_path / "plots",
    )

    assert set(plots) == {"workload_ood", "random_pair"}
    for evaluation, metric_paths in plots.items():
        assert set(metric_paths) == {"BRANCH_RATE", "CPI"}
        for path in metric_paths.values():
            image = Path(path)
            assert image.parent == tmp_path / "plots" / evaluation
            assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_generalization_error_plots_allow_missing_workload_configuration_pairs(tmp_path: Path) -> None:
    workload_ood = tmp_path / "workload_ood_input"
    random_pair = tmp_path / "random_pair_input"
    _write_evaluation(workload_ood, protocol="grouped_workload_kfold", predictions_name="oof_predictions.parquet")
    _write_evaluation(random_pair, protocol="random_pair_split", predictions_name="test_predictions.parquet")

    plots = plot_surrogate_generalization_errors(
        workload_ood_dir=workload_ood,
        random_pair_dir=random_pair,
        output_dir=tmp_path / "plots",
    )

    assert Path(plots["workload_ood"]["CPI"]).is_file()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda root: (root / "evaluation.json").write_text(json.dumps({"protocol": "wrong"}), encoding="utf-8"), "protocol"),
        (
            lambda root: _predictions().drop(columns="config_id").to_parquet(root / "oof_predictions.parquet", index=False),
            "missing columns",
        ),
        (
            lambda root: _predictions().assign(truth_CPI=np.nan).to_parquet(root / "oof_predictions.parquet", index=False),
            "non-finite",
        ),
    ],
)
def test_generalization_error_plots_reject_invalid_workload_ood_artifacts(
    tmp_path: Path, mutate, message: str,
) -> None:
    workload_ood = tmp_path / "workload_ood_input"
    random_pair = tmp_path / "random_pair_input"
    _write_evaluation(workload_ood, protocol="grouped_workload_kfold", predictions_name="oof_predictions.parquet")
    _write_evaluation(random_pair, protocol="random_pair_split", predictions_name="test_predictions.parquet")
    mutate(workload_ood)

    with pytest.raises(ValueError, match=message):
        plot_surrogate_generalization_errors(
            workload_ood_dir=workload_ood,
            random_pair_dir=random_pair,
            output_dir=tmp_path / "plots",
        )


def _cross_domain_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "workload_id": ["alpha", "alpha", "beta", "beta"],
            "prefix_index": [1, 2, 1, 2],
            "truth_CPI": [1.0, 2.0, 3.0, 4.0],
            "anchor_CPI": [1.5, 2.5, 2.5, 50.0],
            "mlp_CPI": [0.9, 2.2, 2.8, 4.1],
            "truth_BRANCH_MPKI": [0.0, 0.0, 10.0, 1000.0],
            "anchor_BRANCH_MPKI": [0.0, 1.0, 12.0, 800.0],
            "mlp_BRANCH_MPKI": [0.0, 0.2, 11.0, 950.0],
            "truth_L2_REFILL_RATIO": [np.nan, 0.0, 0.5, 1.0],
            "anchor_L2_REFILL_RATIO": [0.0, 0.0, 0.6, 0.8],
            "mlp_L2_REFILL_RATIO": [0.0, 0.0, 0.5, 1.1],
            "truth_STALL_CYCLE_RATIO": [0.1, 0.2, 0.3, 0.4],
        }
    )


def _write_cross_domain_evaluation(root: Path) -> None:
    root.mkdir()
    payload = {
        "protocol": "workload_kfold",
        "samples": 4,
        "workloads": 2,
        "metrics": {
            "CPI": {
                "metric_kind": "continuous",
                "truth_degenerate": False,
                "models": {
                    "anchor": {"smape_pct": 20.0, "wape_pct": 20.0},
                    "mlp": {"smape_pct": 4.0, "wape_pct": 4.0},
                },
                "mlp_vs_anchor": {"smape_pct_delta": -16.0, "wape_pct_delta": -16.0},
            },
            "BRANCH_MPKI": {
                "metric_kind": "sparse",
                "truth_degenerate": False,
                "models": {
                    "anchor": {"average_precision": 0.7, "positive_wape_pct": 30.0, "smape_pct": 40.0, "wape_pct": 35.0},
                    "mlp": {"average_precision": 0.75, "positive_wape_pct": 25.0, "smape_pct": 35.0, "wape_pct": 30.0},
                },
                "mlp_vs_anchor": {"positive_wape_pct_delta": -5.0, "smape_pct_delta": -5.0, "wape_pct_delta": -5.0},
            },
        },
        "derived_metrics": {
            "L2_REFILL_RATIO": {
                "metric_kind": "sparse",
                "truth_degenerate": False,
                "models": {
                    "anchor": {"average_precision": 0.7, "positive_wape_pct": 30.0, "smape_pct": 40.0, "wape_pct": 35.0},
                    "mlp": {"average_precision": 0.85, "positive_wape_pct": 15.0, "smape_pct": 20.0, "wape_pct": 18.0},
                },
                "mlp_vs_anchor": {"positive_wape_pct_delta": -15.0, "smape_pct_delta": -20.0, "wape_pct_delta": -17.0},
            },
        },
    }
    (root / "evaluation.json").write_text(json.dumps(payload), encoding="utf-8")
    _cross_domain_predictions().to_parquet(root / "oof_predictions.parquet", index=False)


def test_cross_domain_model_comparison_plots_metric_pngs_and_manifest(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    _write_cross_domain_evaluation(evaluation)

    plots = plot_cross_domain_model_comparison(
        evaluation_dir=evaluation,
        output_dir=tmp_path / "plots",
    )

    assert set(plots) == {"BRANCH_MPKI", "CPI", "L2_REFILL_RATIO"}
    for path in plots.values():
        image = Path(path)
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    manifest = json.loads((tmp_path / "plots" / "plots.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"protocol", "plots"}
    assert manifest["protocol"] == "workload_kfold"
    assert manifest["plots"] == plots


def test_cross_domain_model_comparison_rejects_missing_model_columns(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    _write_cross_domain_evaluation(evaluation)
    _cross_domain_predictions().drop(columns="mlp_CPI").to_parquet(
        evaluation / "oof_predictions.parquet",
        index=False,
    )

    with pytest.raises(ValueError, match="required model columns"):
        plot_cross_domain_model_comparison(
            evaluation_dir=evaluation,
            output_dir=tmp_path / "plots",
        )


def test_cross_domain_model_comparison_handles_zero_inflated_long_tail_truth(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    rows = 20
    truth = np.array([0.0] * 8 + [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 10.0, 100.0, 1_000.0, 10_000.0])
    frame = pd.DataFrame({
        "workload_id": ["alpha"] * rows,
        "prefix_index": np.arange(1, rows + 1),
        "truth_CPI": truth,
        "anchor_CPI": truth + np.linspace(0.0, 10.0, rows),
        "mlp_CPI": truth,
    })
    (evaluation / "evaluation.json").write_text(
        json.dumps({
            "protocol": "workload_kfold",
            "samples": rows,
            "workloads": 1,
            "metrics": {
                "CPI": {
                    "metric_kind": "continuous",
                    "truth_degenerate": False,
                    "models": {
                        "anchor": {"smape_pct": 10.0, "wape_pct": 10.0},
                        "mlp": {"smape_pct": 0.0, "wape_pct": 0.0},
                    },
                    "mlp_vs_anchor": {"smape_pct_delta": -10.0, "wape_pct_delta": -10.0},
                },
            },
        }),
        encoding="utf-8",
    )
    frame.to_parquet(evaluation / "oof_predictions.parquet", index=False)

    plots = plot_cross_domain_model_comparison(evaluation_dir=evaluation, output_dir=tmp_path / "plots")

    assert Path(plots["CPI"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
