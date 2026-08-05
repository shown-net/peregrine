from __future__ import annotations

from pathlib import Path

import numpy as np

from anamol.python.design_space import analytical_feature_columns
from anamol.python.design_space import workload_context_feature_columns
from anamol.python.dataset import _dataset_table_context
from anamol.python.dataset import _full_roi_feature_table
from anamol.python.dataset import _workload_context_values
from anamol.python.dataset import build_candidate_feature_table
from anamol.python.design_space import load_peregrine_config
from anamol.python.microarchitecture import load_microarchitecture_config
from tests.helpers import METRICS_CONFIG
from tests.helpers import MICROARCHITECTURE_CONFIG
from tests.helpers import cpu_microarchitecture_root


def _config():
    root = cpu_microarchitecture_root()
    return load_peregrine_config(
        root / "peregrine/configs/peregrine.yaml",
        metrics_config=METRICS_CONFIG,
        microarchitecture=load_microarchitecture_config(MICROARCHITECTURE_CONFIG),
    )


def test_workload_context_columns_append_global_trace_features() -> None:
    config = _config()
    analytical = analytical_feature_columns(config.microarchitecture.mechanisms)
    context = workload_context_feature_columns(config.microarchitecture.mechanisms)
    first_mechanism = config.microarchitecture.mechanisms[0].name

    assert context[:4] == (
        f"workload_context_mean__{first_mechanism}",
        f"workload_context_p90__{first_mechanism}",
        f"workload_context_std__{first_mechanism}",
        f"workload_context_active_ratio__{first_mechanism}",
    )
    assert len(context) == len(config.microarchitecture.mechanisms) * 4
    assert len(context) < len(analytical)
    assert config.feature_columns[-len(context):] == context


def test_workload_context_values_use_reference_config_mechanism_means_only() -> None:
    config = _config()
    table_context = _dataset_table_context(config)
    analytical = analytical_feature_columns(config.microarchitecture.mechanisms)
    analytical_count = len(analytical)
    first_mean = analytical.index("dynamic_rob_mean")
    second_mean = analytical.index("dynamic_load_queue_mean")
    features = np.zeros((2, 2, analytical_count), dtype=np.float64)
    features[:, 0, first_mean] = [1.0, 3.0]
    features[:, 1, first_mean] = [100.0, 300.0]
    features[:, 0, second_mean] = [0.0, 4.0]
    features[:, 0, analytical.index("dynamic_rob_raw_p1")] = [999.0, 999.0]

    context = _workload_context_values(
        features,
        reference_config_index=0,
        table_context=table_context,
    )

    assert context[:8].tolist() == [2.0, 2.8, 1.0, 1.0, 2.0, 3.6, 2.0, 0.5]


def test_full_roi_feature_table_repeats_workload_context_for_each_config() -> None:
    config = _config()
    table_context = _dataset_table_context(config)
    baseline = config.run_config_from_args(
        config.microarchitecture.config_id({}),
        config.microarchitecture.gem5_args(),
    )
    features = np.zeros((1, len(table_context.analytical_columns)), dtype=np.float64)
    features[0, 0] = 7.0
    workload_context = np.arange(len(table_context.workload_context_columns), dtype=np.float64)

    table = _full_roi_feature_table(
        workload_id="work",
        window_index=0,
        feature_values=features,
        workload_context=workload_context,
        configs=(baseline,),
        config=config,
        table_context=table_context,
    )

    assert table.column(table_context.analytical_columns[0]).to_pylist() == [7.0]
    assert table.column(table_context.workload_context_columns[0]).to_pylist() == [0.0]
    assert table.column(table_context.workload_context_columns[-1]).to_pylist() == [
        float(len(table_context.workload_context_columns) - 1)
    ]


def test_candidate_feature_table_uses_reference_config_for_context(monkeypatch, tmp_path: Path) -> None:
    config = _config()
    analytical = analytical_feature_columns(config.microarchitecture.mechanisms)
    candidate = config.run_config_from_args(
        config.microarchitecture.config_id({"l1d_size": 16}),
        config.microarchitecture.gem5_args({"l1d_size": 16}),
    )

    def fake_analyze_full_roi_windows(**kwargs):
        configs = kwargs["configs"]
        values = np.zeros((1, len(configs), len(analytical)), dtype=np.float64)
        values[:, 0, 0] = [3.0]
        values[:, 1, 0] = [200.0]
        values[:, 0, analytical.index("dynamic_rob_mean")] = [3.0]
        return values

    monkeypatch.setattr(
        "anamol.python.dataset.analyze_full_roi_windows",
        fake_analyze_full_roi_windows,
    )

    table = build_candidate_feature_table(
        config=config,
        workload_id="work",
        trace_path=tmp_path / "trace.pb.zst",
        window_count=2,
        run_configs=(candidate,),
        full_roi_window_size=100000,
    )

    context_columns = workload_context_feature_columns(config.microarchitecture.mechanisms)
    assert table.column("window_index").to_pylist() == [1]
    assert table.column(analytical[0]).to_pylist() == [200.0]
    assert table.column(context_columns[0]).to_pylist() == [3.0]
    assert table.column(context_columns[1]).to_pylist() == [3.0]
