from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
