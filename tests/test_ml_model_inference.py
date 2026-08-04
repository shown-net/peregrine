from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch
import pytest

from anamol.python.design_space import AnalysisConfig
from anamol.python.design_space import CollectionSamplingConfig
from anamol.python.design_space import EvaluationConfig
from anamol.python.design_space import PeregrineConfig
from anamol.python.design_space import TrainingConfig
from anamol.python.design_space import load_peregrine_config
from anamol.python.microarchitecture import load_microarchitecture_config
from ml_model.inference import CpuMultiHeadPredictor
from ml_model.inference import predict_parquet
from ml_model.model import MultiHeadPeregrineModel
from ml_model.multitask import LOG1P_NONNEGATIVE_TARGET_TRANSFORM
from ml_model.multitask import MultiHeadTraining
from ml_model.multitask import inverse_transform_targets
from ml_model.multitask import regression_error_report
from ml_model.multitask import scale
from ml_model.multitask import standardize
from ml_model.plots import plot_surrogate_summary
from ml_model.prediction import FeatureSet
from ml_model.prediction import PredictionTask
from ml_model.prediction import _read_task_frame
from ml_model.prediction import evaluate_prediction_task
from ml_model.prediction import evaluate_random_roi_split_prediction_task
from ml_model.prediction import train_prediction_task
from ml_model.tasks import SURROGATE_TASK_ID
from ml_model.tasks import surrogate_task
from tests.helpers import METRICS_CONFIG
from tests.helpers import MICROARCHITECTURE_CONFIG
from tests.helpers import cpu_microarchitecture_root


def test_prediction_task_bundle_uses_paths_relative_to_its_own_directory(tmp_path: Path, monkeypatch) -> None:
    from ml_model.prediction import FeatureSet
    from ml_model.prediction import PredictionTask
    from ml_model.prediction import train_prediction_task

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    rows = []
    for workload_index, workload in enumerate(("w0", "w1", "w2")):
        for interval in range(2):
            value = float(workload_index + interval + 1)
            rows.append({
                "workload_id": workload,
                "interval_index": interval,
                "feature_raw": value,
                "feature_model": value,
                "label_raw": value,
                "label_model": value * 2.0,
            })
    pd.DataFrame(rows).to_parquet(dataset / "samples.parquet", index=False)
    task = PredictionTask(
        task_id="synthetic",
        identity_columns=("workload_id", "interval_index"),
        group_column="workload_id",
        feature_set=FeatureSet("core", ("feature_model",)),
        label_columns=("label_raw", "label_model"),
        output_metrics=("raw", "model"),
        training=MultiHeadTraining((4, 3), 2, 2, 0.01, 0.0, 1),
        num_threads=1,
        seed=7,
    )

    monkeypatch.chdir(tmp_path)
    report = train_prediction_task(task=task, dataset_dir=dataset, output_dir=Path("model"))

    training = json.loads(Path(report["training_report"]).read_text())
    assert training["protocol"] == "full_dataset_deployment_training"
    assert set(training["selected"]) == {"raw", "model"}
    assert Path(report["bundle"]).is_file()
    from ml_model.inference import PredictorBundle
    PredictorBundle(report["bundle"])


def test_predictor_bundle_combines_singlehead_outputs(tmp_path: Path) -> None:
    from ml_model.inference import PredictorBundle
    from ml_model.inference import predict_bundle_parquet

    checkpoint = tmp_path / "model.pt"
    model = MultiHeadPeregrineModel(1, (4, 3), ("label_raw",))
    torch.save({
        "state_dict": model.state_dict(), "feature_columns": ("feature_model",),
        "label_columns": ("label_raw",), "output_metrics": ("raw",),
        "hidden_dims": (4, 3), "num_threads": 1,
        "feature_mean": torch.zeros(1), "feature_scale": torch.ones(1),
        "label_mean": torch.zeros(1), "label_scale": torch.ones(1),
    }, checkpoint)
    model_output = tmp_path / "model-output.pt"
    model = MultiHeadPeregrineModel(1, (4, 3), ("label_model",))
    torch.save({
        "state_dict": model.state_dict(), "feature_columns": ("feature_model",),
        "label_columns": ("label_model",), "output_metrics": ("model",),
        "hidden_dims": (4, 3), "num_threads": 1,
        "feature_mean": torch.zeros(1), "feature_scale": torch.ones(1),
        "label_mean": torch.zeros(1), "label_scale": torch.ones(1),
    }, model_output)
    bundle_path = tmp_path / "predictor_bundle.json"
    bundle_path.write_text(json.dumps({
        "task_id": "synthetic", "identity_columns": ["workload_id", "interval_index"],
        "output_metrics": ["raw", "model"],
        "selected": {
            "raw": {"candidate_id": "singlehead:core:raw", "predictor_kind": "singlehead", "feature_set": "core"},
            "model": {"candidate_id": "singlehead:core:model", "predictor_kind": "singlehead", "feature_set": "core"},
        },
        "model_paths": {"singlehead:core:raw": str(checkpoint), "singlehead:core:model": str(model_output)},
    }))
    source = tmp_path / "features.parquet"
    pd.DataFrame({
        "workload_id": ["w0"], "interval_index": [0],
        "feature_raw": [3.0], "feature_model": [2.0],
    }).to_parquet(source, index=False)
    destination = tmp_path / "prediction.parquet"

    report = predict_bundle_parquet(
        predictor=PredictorBundle(bundle_path), features_path=source, output_path=destination,
    )

    table = pq.read_table(destination)
    assert report["rows"] == 1
    assert table.column_names == ["workload_id", "interval_index", "prediction_raw", "prediction_model"]


def test_checkpoint_drives_dynamic_prediction_outputs(tmp_path: Path) -> None:
    feature_columns = ("f0", "f1")
    label_columns = (
        "label_CPI",
        "label_BRANCH_RATE",
        "label_MEM_ACCESS_PER_KI",
        "label_BRANCH_MPKI",
        "label_ICACHE_MPKI",
        "label_L1D_MPKI",
        "label_L2_MPKI",
    )
    output_metrics = tuple(label.removeprefix("label_") for label in label_columns)
    model = MultiHeadPeregrineModel(len(feature_columns), (4, 3), label_columns)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "label_columns": label_columns,
            "output_metrics": output_metrics,
            "hidden_dims": (4, 3),
            "num_threads": 1,
            "feature_mean": torch.zeros(len(feature_columns)),
            "feature_scale": torch.ones(len(feature_columns)),
            "label_mean": torch.zeros(len(label_columns)),
            "label_scale": torch.ones(len(label_columns)),
        },
        checkpoint,
    )
    features = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "workload_id": ["w0", "w0", "w0"],
            "region_id": ["window_0", "window_1", "window_2"],
            "config_id": ["config_0", "config_0", "config_0"],
            "f0": [1.0, 2.0, 3.0],
            "f1": [4.0, 5.0, 6.0],
        }
    ).to_parquet(features, index=False)
    output = tmp_path / "predictions.parquet"

    report = predict_parquet(
        predictor=CpuMultiHeadPredictor(checkpoint),
        features_path=features,
        output_path=output,
        batch_size=2,
    )
    predictions = pq.read_table(output)

    assert report["rows"] == 3
    assert predictions.column_names == [
        "workload_id",
        "region_id",
        "config_id",
        *(f"prediction_{metric}" for metric in output_metrics),
    ]


def test_predict_array_is_batch_size_invariant(tmp_path: Path) -> None:
    feature_columns = ("f0", "f1")
    label_columns = ("label_CPI", "label_BRANCH_RATE")
    model = MultiHeadPeregrineModel(len(feature_columns), (4, 3), label_columns)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "label_columns": label_columns,
            "output_metrics": ("CPI", "BRANCH_RATE"),
            "hidden_dims": (4, 3),
            "num_threads": 1,
            "feature_mean": torch.zeros(len(feature_columns)),
            "feature_scale": torch.ones(len(feature_columns)),
            "label_mean": torch.zeros(len(label_columns)),
            "label_scale": torch.ones(len(label_columns)),
        },
        checkpoint,
    )
    predictor = CpuMultiHeadPredictor(checkpoint)
    assert torch.get_num_threads() == 1
    features = __import__("numpy").array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        dtype="float32",
    )

    small = predictor.predict_array(features, batch_size=2)
    large = predictor.predict_array(features, batch_size=16)

    assert __import__("numpy").allclose(small, large)
    assert small.shape == (5, 2)


def test_log1p_nonnegative_checkpoint_clamps_physical_predictions(tmp_path: Path) -> None:
    feature_columns = ("f0",)
    label_columns = ("label_CPI",)
    model = MultiHeadPeregrineModel(len(feature_columns), (4, 3), label_columns)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "label_columns": label_columns,
            "output_metrics": ("CPI",),
            "hidden_dims": (4, 3),
            "num_threads": 1,
            "feature_mean": torch.zeros(len(feature_columns)),
            "feature_scale": torch.ones(len(feature_columns)),
            "label_mean": torch.full((1,), -100.0),
            "label_scale": torch.ones(len(label_columns)),
            "target_transforms": (LOG1P_NONNEGATIVE_TARGET_TRANSFORM,),
        },
        checkpoint,
    )
    predictor = CpuMultiHeadPredictor(checkpoint)
    predictions = predictor.predict_array(__import__("numpy").array([[0.0], [1.0]], dtype="float32"))

    assert (predictions >= 0.0).all()
    assert predictor.target_transforms == (LOG1P_NONNEGATIVE_TARGET_TRANSFORM,)


def test_workload_grouped_evaluation_uses_shared_report_contract(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    pd.DataFrame(
        {
            "workload_id": ["w0", "w0", "w1", "w1", "w2", "w2"],
            "region_id": [f"window_{index}" for index in range(6)],
            "config_id": ["config_0"] * 6,
            "f0": [0.0, 0.1, 1.0, 1.1, 2.0, 2.1],
            "f1": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
            "label_CPI": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
        }
    ).to_parquet(dataset / "samples.parquet", index=False)
    task = _test_task(feature_columns=("f0", "f1"), label_columns=("label_CPI",), output_metrics=("CPI",))
    report = evaluate_prediction_task(task=task, dataset_dir=dataset, output_dir=tmp_path / "evaluation")
    evaluation = json.loads(Path(report["evaluation"]).read_text())
    assert evaluation["primary_metric"] == {
        "aggregation": "macro_workload", "metric": "mape_pct", "target": "CPI",
    }
    errors = evaluation["metrics"]["roi_weighted"]["CPI"]
    assert set(errors) == {
        "mae", "rmse", "mape_pct", "mape_nonzero_rows",
        "p90_absolute_error", "wape_pct", "smape_pct",
    }
    assert all(__import__("numpy").isfinite(errors[name]) for name in ("mae", "rmse", "mape_pct", "p90_absolute_error", "wape_pct", "smape_pct"))


def test_prediction_failure_does_not_replace_existing_output(tmp_path: Path) -> None:
    feature_columns = ("f0",)
    label_columns = ("label_CPI",)
    model = MultiHeadPeregrineModel(1, (4, 3), label_columns)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(), "feature_columns": feature_columns,
            "label_columns": label_columns, "output_metrics": ("CPI",),
            "hidden_dims": (4, 3), "feature_mean": torch.zeros(1),
            "num_threads": 1,
            "feature_scale": torch.ones(1), "label_mean": torch.zeros(1),
            "label_scale": torch.ones(1),
        },
        checkpoint,
    )
    features = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "workload_id": ["w0", "w0"], "region_id": ["r0", "r1"],
            "config_id": ["c0", "c1"], "f0": [1.0, float("nan")],
        }
    ).to_parquet(features, index=False)
    output = tmp_path / "predictions.parquet"
    output.write_bytes(b"previous-output")

    with pytest.raises(ValueError, match="invalid inference feature matrix"):
        predict_parquet(
            predictor=CpuMultiHeadPredictor(checkpoint),
            features_path=features,
            output_path=output,
            batch_size=1,
        )

    assert output.read_bytes() == b"previous-output"
    assert not output.with_suffix(".parquet.partial").exists()


def test_training_rejects_a_constant_label(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    pd.DataFrame(
        {
            "workload_id": ["w0", "w1"], "region_id": ["r0", "r1"],
            "config_id": ["c0", "c1"], "f0": [0.0, 1.0],
            "label_constant": [0.0, 0.0],
        }
    ).to_parquet(dataset / "samples.parquet", index=False)

    with pytest.raises(ValueError, match="zero-variance labels"):
        train_prediction_task(
            task=_test_task(feature_columns=("f0",), label_columns=("label_constant",), output_metrics=("constant",)),
            dataset_dir=dataset, output_dir=tmp_path / "model",
        )


def test_shared_error_report_uses_percentage_units_and_zero_safe_mape() -> None:
    report = regression_error_report(
        ("label_CPI",),
        __import__("numpy").array([[0.0], [2.0]], dtype="float32"),
        __import__("numpy").array([[1.0], [1.0]], dtype="float32"),
    )["CPI"]

    assert report["mae"] == 1.0
    assert report["rmse"] == 1.0
    assert report["mape_pct"] == 50.0
    assert report["mape_nonzero_rows"] == 1
    assert report["p90_absolute_error"] == 1.0
    assert report["wape_pct"] == 100.0
    assert report["smape_pct"] == pytest.approx(133.33333333333334)


def test_macro_workload_mape_does_not_follow_roi_sample_counts() -> None:
    from ml_model.prediction import _macro_workload_report

    task = _test_task(feature_columns=("f0",), label_columns=("label_CPI",), output_metrics=("CPI",))
    values = __import__("numpy").array(["small", "large", "large", "large"])
    truth = __import__("numpy").full((4, 1), 10.0, dtype="float32")
    prediction = __import__("numpy").array([[0.0], [10.0], [10.0], [10.0]], dtype="float32")

    macro = _macro_workload_report(task, values, truth, prediction)["CPI"]
    roi = regression_error_report(task.label_columns, truth, prediction)["CPI"]

    assert macro["mape_pct"] == 50.0
    assert roi["mape_pct"] == 25.0


def test_macro_workload_report_preserves_optional_percentage_metrics() -> None:
    from ml_model.prediction import _macro_workload_report

    task = _test_task(feature_columns=("f0",), label_columns=("label_CPI",), output_metrics=("CPI",))
    values = __import__("numpy").array(["zero", "nonzero"])
    truth = __import__("numpy").array([[0.0], [10.0]], dtype="float32")
    prediction = __import__("numpy").array([[1.0], [5.0]], dtype="float32")

    macro = _macro_workload_report(task, values, truth, prediction)["CPI"]

    assert macro["mae"] == 3.0
    assert macro["mape_pct"] == 50.0
    assert macro["mape_nonzero_rows"] == 1
    assert macro["wape_pct"] == 50.0


def test_log1p_inverse_transform_is_nonnegative() -> None:
    restored = inverse_transform_targets(
        __import__("numpy").array([[-100.0], [0.0], [100.0]], dtype="float32"),
        (LOG1P_NONNEGATIVE_TARGET_TRANSFORM,),
    )

    assert (restored >= 0.0).all()
    assert __import__("numpy").isfinite(restored).all()


def test_standardize_bounds_a_constant_training_feature_outside_support() -> None:
    mean, scale_value = standardize(__import__("numpy").array([[400.0], [400.0]], dtype="float32"))
    heldout = scale(__import__("numpy").array([[100.0]], dtype="float32"), mean, scale_value)

    assert scale_value.tolist() == [1.0]
    assert heldout.tolist() == [[-8.0]]


def test_surrogate_task_uses_one_full_feature_singlehead_protocol() -> None:
    task = surrogate_task(load_peregrine_config(
        cpu_microarchitecture_root() / "peregrine/configs/peregrine.yaml", metrics_config=METRICS_CONFIG,
        microarchitecture=load_microarchitecture_config(MICROARCHITECTURE_CONFIG),
    ))

    assert task.feature_set.feature_set_id == "trace_design"
    assert task.identity_columns == ("workload_id", "window_index", "config_id")
    assert task.group_column == "config_id"


def test_workload_grouped_evaluation_excludes_heldout_labels_from_its_prediction(tmp_path: Path) -> None:
    def write_dataset(path: Path, heldout_offset: float) -> None:
        path.mkdir()
        rows = []
        for workload_index, workload in enumerate(("w0", "w1", "w2", "w3")):
            for interval in range(3):
                rows.append({
                    "workload_id": workload, "region_id": f"{workload}_{interval}",
                    "config_id": f"c_{workload}_{interval}", "f0": float(workload_index + interval),
                    "label_CPI": float(workload_index + interval + 1 + (heldout_offset if workload == "w0" else 0.0)),
                })
        pd.DataFrame(rows).to_parquet(path / "samples.parquet", index=False)

    first, second = tmp_path / "first", tmp_path / "second"
    write_dataset(first, 0.0)
    write_dataset(second, 100.0)
    task = _test_task(feature_columns=("f0",), label_columns=("label_CPI",), output_metrics=("CPI",))

    first_report = evaluate_prediction_task(task=task, dataset_dir=first, output_dir=tmp_path / "first-evaluation")
    second_report = evaluate_prediction_task(task=task, dataset_dir=second, output_dir=tmp_path / "second-evaluation")

    first_oof = pd.read_parquet(first_report["oof_predictions"])
    second_oof = pd.read_parquet(second_report["oof_predictions"])
    first_w0 = first_oof.loc[first_oof.workload_id == "w0", "prediction_CPI"].to_numpy()
    second_w0 = second_oof.loc[second_oof.workload_id == "w0", "prediction_CPI"].to_numpy()
    evaluation = json.loads(Path(first_report["evaluation"]).read_text())
    assert __import__("numpy").allclose(first_w0, second_w0)
    assert evaluation["protocol"] == "leave_one_workload_out"
    assert evaluation["generalization_scope"] == "joint_program_microarchitecture_ood"
    assert not (tmp_path / "first-evaluation" / "predictor_bundle.json").exists()


def test_random_roi_evaluation_reports_the_historical_id_split(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    pd.DataFrame({
        "workload_id": [f"w{index % 3}" for index in range(20)],
        "region_id": [f"r{index}" for index in range(20)],
        "config_id": [f"c{index}" for index in range(20)],
        "f0": [float(index) for index in range(20)],
        "label_CPI": [float(index + 1) for index in range(20)],
    }).to_parquet(dataset / "samples.parquet", index=False)

    report = evaluate_random_roi_split_prediction_task(
        task=_test_task(feature_columns=("f0",), label_columns=("label_CPI",), output_metrics=("CPI",)),
        dataset_dir=dataset, output_dir=tmp_path / "evaluation",
    )

    evaluation = json.loads(Path(report["evaluation"]).read_text())
    split = evaluation["split"]
    assert evaluation["protocol"] == "random_roi_split"
    assert evaluation["primary_metric"] == {
        "aggregation": "roi_weighted", "metric": "mape_pct", "target": "CPI",
    }
    assert split["train_rows"] + split["validation_rows"] + split["test_rows"] == 20
    assert set(evaluation["metrics"]["roi_weighted"]) == {"CPI"}
    assert pq.read_table(report["test_predictions"]).num_rows == split["test_rows"]


def test_config_evaluation_keeps_every_config_out_of_fit_and_validation(tmp_path: Path) -> None:
    def write_dataset(path: Path, heldout_offset: float) -> None:
        path.mkdir()
        rows = []
        for config_id in ("baseline", *(f"c{index}" for index in range(6))):
            config_index = 0 if config_id == "baseline" else int(config_id[1:])
            for workload_index in range(2):
                for window_index in range(2):
                    rows.append({
                        "workload_id": f"w{workload_index}", "window_index": window_index,
                        "config_id": config_id, "f0": float(config_index + workload_index + window_index),
                        "label_CPI": float(config_index + workload_index + window_index + 1 + (heldout_offset if config_id == "c0" else 0.0)),
                    })
        pd.DataFrame(rows).to_parquet(path / "samples.parquet", index=False)

    first, second = tmp_path / "first-config", tmp_path / "second-config"
    write_dataset(first, 0.0)
    write_dataset(second, 100.0)
    task = replace(
        _test_task(feature_columns=("f0",), label_columns=("label_CPI",), output_metrics=("CPI",)),
        identity_columns=("workload_id", "window_index", "config_id"),
        group_column="config_id", evaluation_folds=3,
    )

    first_report = evaluate_prediction_task(task=task, dataset_dir=first, output_dir=tmp_path / "first-config-eval")
    second_report = evaluate_prediction_task(task=task, dataset_dir=second, output_dir=tmp_path / "second-config-eval")
    first_prediction = pd.read_parquet(first_report["oof_predictions"])
    second_prediction = pd.read_parquet(second_report["oof_predictions"])
    first_c0 = first_prediction.loc[first_prediction.config_id == "c0", "prediction_CPI"].to_numpy()
    second_c0 = second_prediction.loc[second_prediction.config_id == "c0", "prediction_CPI"].to_numpy()
    evaluation = json.loads(Path(first_report["evaluation"]).read_text())

    assert __import__("numpy").allclose(first_c0, second_c0)
    assert set(first_prediction.config_id) == {"baseline", *(f"c{index}" for index in range(6))}
    assert evaluation["metrics"]["per_metric_absolute"]["CPI"]["mae"] is not None
    assert evaluation["protocol"] == "grouped_config_kfold"
    assert evaluation["generalization_scope"] == "held_out_configuration"
    for fold in evaluation["outer_folds"]:
        assert not set(fold["heldout_groups"]) & set(fold["validation_groups"])
    assert "retrieval" not in evaluation


def test_l1_plot_summary_writes_three_core_figures(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    generalization = tmp_path / "config_generalization"
    random_roi = tmp_path / "evaluation"
    plots = tmp_path / "plots"
    dataset.mkdir()
    generalization.mkdir()
    random_roi.mkdir()
    metrics = (
        "CPI",
        "BRANCH_MPKI",
        "CHI_L1I_IFETCH_MPKI",
        "CHI_L1D_LD_MPKI",
        "CHI_L2_LD_MPKI",
    )
    rows = []
    predictions = []
    for workload_index, workload in enumerate(("w0", "w1", "w2")):
        for index in range(3):
            row = {"workload_id": workload, "region_id": f"{workload}_{index}", "config_id": f"c{index}"}
            pred = dict(row)
            for metric_index, metric in enumerate(metrics):
                truth = float((workload_index + 1) * (index + 1) * (metric_index + 1))
                row[f"label_{metric}"] = truth
                pred[f"truth_{metric}"] = truth
                pred[f"prediction_{metric}"] = truth * (1.0 + 0.1 * (workload_index + 1))
            rows.append(row)
            predictions.append(pred)
    pd.DataFrame(rows).to_parquet(dataset / "samples.parquet", index=False)
    pd.DataFrame(predictions).to_parquet(generalization / "oof_predictions.parquet", index=False)
    generalization_report = {
        "task_id": SURROGATE_TASK_ID,
        "generalization_scope": "held_out_configuration",
        "metrics": {
            "per_metric_absolute": {
                metric: {"smape_pct": 10.0 + offset, "wape_pct": 11.0 + offset}
                for offset, metric in enumerate(metrics)
            },
        },
    }
    (generalization / "evaluation.json").write_text(json.dumps(generalization_report), encoding="utf-8")
    random_report = {
        "task_id": SURROGATE_TASK_ID,
        "protocol": "random_roi_split",
        "metrics": {
            "roi_weighted": {
                metric: {"smape_pct": 2.0 + offset, "wape_pct": 3.0 + offset}
                for offset, metric in enumerate(metrics)
            },
        },
    }
    (random_roi / "evaluation.json").write_text(json.dumps(random_report), encoding="utf-8")

    report = plot_surrogate_summary(
        dataset_dir=dataset,
        config_generalization_dir=generalization,
        random_roi_dir=random_roi,
        output_dir=plots,
    )
    summary = json.loads(Path(report["summary"]).read_text(encoding="utf-8"))

    assert summary["task_id"] == SURROGATE_TASK_ID
    assert summary["metrics"] == list(metrics)
    assert summary["protocol_errors"]["CPI"]["gap_smape_pct"] == 8.0
    assert summary["workload_errors"]["CPI"]["w0"]["rows"] == 3
    for path in summary["plots"].values():
        assert Path(path).stat().st_size > 0


def test_l1_plot_summary_allows_missing_random_roi(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    generalization = tmp_path / "config_generalization"
    plots = tmp_path / "plots"
    dataset.mkdir()
    generalization.mkdir()
    metrics = (
        "CPI",
        "BRANCH_MPKI",
        "CHI_L1I_IFETCH_MPKI",
        "CHI_L1D_LD_MPKI",
        "CHI_L2_LD_MPKI",
    )
    row = {"workload_id": "w0", "region_id": "r0", "config_id": "c0"}
    prediction = dict(row)
    for metric in metrics:
        row[f"label_{metric}"] = 1.0
        prediction[f"truth_{metric}"] = 1.0
        prediction[f"prediction_{metric}"] = 1.0
    pd.DataFrame([row]).to_parquet(dataset / "samples.parquet", index=False)
    pd.DataFrame([prediction]).to_parquet(generalization / "oof_predictions.parquet", index=False)
    (generalization / "evaluation.json").write_text(json.dumps({
        "task_id": SURROGATE_TASK_ID,
        "generalization_scope": "held_out_configuration",
        "metrics": {
            "per_metric_absolute": {
                metric: {"smape_pct": 0.0, "wape_pct": 0.0}
                for metric in metrics
            },
        },
    }), encoding="utf-8")

    report = plot_surrogate_summary(
        dataset_dir=dataset,
        config_generalization_dir=generalization,
        random_roi_dir=tmp_path / "missing_random_roi",
        output_dir=plots,
    )

    assert report["random_roi_available"] is False
    assert report["protocol_errors"]["CPI"]["random_roi_smape_pct"] is None


def test_surrogate_plot_summary_requires_config_generalization_artifacts(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    pd.DataFrame({
        "label_CPI": [1.0],
        "label_BRANCH_MPKI": [1.0],
        "label_CHI_L1I_IFETCH_MPKI": [1.0],
        "label_CHI_L1D_LD_MPKI": [1.0],
        "label_CHI_L2_LD_MPKI": [1.0],
    }).to_parquet(dataset / "samples.parquet", index=False)

    with pytest.raises(FileNotFoundError, match="missing surrogate held-out-configuration artifacts"):
        plot_surrogate_summary(
            dataset_dir=dataset,
            config_generalization_dir=tmp_path / "missing_generalization",
            random_roi_dir=tmp_path / "missing_random_roi",
            output_dir=tmp_path / "plots",
        )


def test_workload_selection_is_independent_of_prediction_grouping(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    pd.DataFrame({
        "workload_id": ("work_a", "work_a", "work_b"),
        "window_index": (0, 1, 0),
        "config_id": ("config_a", "config_a", "config_a"),
        "feature": (1.0, 2.0, 3.0),
        "label": (1.0, 2.0, 3.0),
    }).to_parquet(dataset / "samples.parquet", index=False)
    task = PredictionTask(
        task_id="prediction-test",
        identity_columns=("workload_id", "window_index", "config_id"),
        group_column="config_id",
        feature_set=FeatureSet("trace_design", ("feature",)),
        label_columns=("label",),
        output_metrics=("metric",),
        training=MultiHeadTraining((4, 3), 2, 2, 0.01, 0.0, 1),
        num_threads=1,
        seed=7,
    )

    frame = _read_task_frame(task, dataset, ("work_a",))

    assert frame["workload_id"].tolist() == ["work_a", "work_a"]


def _training_config() -> PeregrineConfig:
    return PeregrineConfig(
        microarchitecture=load_microarchitecture_config(MICROARCHITECTURE_CONFIG),
        analysis=AnalysisConfig(window_size=1),
        collection_sampling=CollectionSamplingConfig(
            seed=7,
            configs_per_region=1,
        ),
        labels=(),
        training=TrainingConfig(
            max_epochs=2,
            batch_size=2,
            hidden_dims=(4, 3),
            paper_test_fraction=0.33,
            seed=7,
            learning_rate=0.01,
            weight_decay=0.0,
            num_threads=1,
            early_stopping_patience=2,
        ),
        evaluation=EvaluationConfig(config_folds=3),
    )


def _test_task(*, feature_columns: tuple[str, ...], label_columns: tuple[str, ...], output_metrics: tuple[str, ...]) -> PredictionTask:
    return PredictionTask(
        task_id="prediction-test", identity_columns=("workload_id", "region_id", "config_id"),
        group_column="workload_id", feature_set=FeatureSet("trace_design", feature_columns),
        label_columns=label_columns, output_metrics=output_metrics,
        training=MultiHeadTraining((4, 3), 2, 2, 0.01, 0.0, 1),
        num_threads=1, seed=7,
    )
