from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch
import pytest

from anamol.python.design_space import AnalysisConfig
from anamol.python.design_space import CollectionSamplingConfig
from anamol.python.design_space import PeregrineConfig
from anamol.python.design_space import TrainingConfig
from anamol.python.microarchitecture import load_microarchitecture_config
from ml_model.inference import CpuMultiHeadPredictor
from ml_model.inference import predict_parquet
from ml_model.model import MultiHeadPeregrineModel
from ml_model.train import evaluate_workload_ood
from ml_model.train import train_surrogate


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


def test_training_keeps_workload_ood_as_an_explicit_diagnostic(tmp_path: Path) -> None:
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
    config = _training_config()

    report = train_surrogate(
        config=config,
        dataset_dir=dataset,
        output_dir=tmp_path / "model",
        feature_columns=("f0", "f1"),
        label_columns=("label_CPI",),
        output_metrics=("CPI",),
    )
    errors = report["random_split"]
    for name in ("mae", "rmse", "wape", "smape"):
        assert set(errors[name]) == {"CPI"}
        assert __import__("numpy").isfinite(errors[name]["CPI"])
    ood = evaluate_workload_ood(
        config=config,
        dataset_dir=dataset,
        feature_columns=("f0", "f1"),
        label_columns=("label_CPI",),
    )
    assert {item["heldout_workload"] for item in ood["workload_ood"]} == {"w0", "w1", "w2"}


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
        train_surrogate(
            config=_training_config(),
            dataset_dir=dataset,
            output_dir=tmp_path / "model",
            feature_columns=("f0",),
            label_columns=("label_constant",),
            output_metrics=("constant",),
        )


def _training_config() -> PeregrineConfig:
    return PeregrineConfig(
        microarchitecture=load_microarchitecture_config(
            "../cpu_microarchitecture/configs/microarchitectures/zte_neoverse_n2.yaml"
        ),
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
    )
