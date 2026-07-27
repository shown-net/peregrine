import json
from pathlib import Path

import pyarrow.parquet as pq

from anamol.python.collection_plan import sample_run_configs
from anamol.python.collection_plan import is_legal
from anamol.python.collection_plan import sample_region_run_configs
from anamol.python import dataset as dataset_module
from anamol.python.dataset import _analytical_config
from anamol.python.feature_pipeline import FeatureBatch
from anamol.python.feature_pipeline import analytical_feature_columns
from anamol.python.feature_pipeline import _config_payload
from tests.helpers import load_test_config


def test_sample_run_configs_is_deterministic_and_uses_only_config_fields() -> None:
    config = load_test_config()

    first = sample_run_configs(config=config, count=3, seed=11)
    second = sample_run_configs(config=config, count=3, seed=11)

    assert first == second
    assert len(first) == 3
    assert len({item.config_id for item in first}) == 3
    for item in first:
        assert len(item.config_id) == len("config_") + 16
        assert item.gem5_args
        assert set(item.parameter_values) == set(config.microarchitecture.parameters_by_name)
        assert config.run_config_from_args(item.config_id, item.gem5_args).parameter_values == item.parameter_values


def test_region_sampling_is_deterministic_for_each_region() -> None:
    config = load_test_config()

    first = sample_region_run_configs(
        config=config,
        workload_id="work",
        region_id="simpoint_0",
    )
    repeated = sample_region_run_configs(
        config=config,
        workload_id="work",
        region_id="simpoint_0",
    )
    other_region = sample_region_run_configs(
        config=config,
        workload_id="work",
        region_id="simpoint_1",
    )

    assert first == repeated
    assert first != other_region
    assert len(first) == config.collection_sampling.configs_per_region
    assert len({item.config_id for item in first}) == len(first)
    assert first[0].parameter_values == config.microarchitecture.baseline_values
    assert all(is_legal(dict(item.parameter_values), config=config) for item in first)


def test_analytical_config_carries_canonical_cache_geometry() -> None:
    config = load_test_config()
    run = sample_run_configs(config=config, count=1, seed=1)[0]

    analytical = _analytical_config(run, config)

    assert analytical == config.microarchitecture.analytical_values(run.parameter_values)


def test_analytical_config_restores_legacy_icache_fills_component() -> None:
    config = load_test_config()
    run = sample_run_configs(config=config, count=1, seed=1)[0]

    analytical = _analytical_config(run, config)
    columns = analytical_feature_columns(config.microarchitecture.mechanisms)

    assert analytical["max_icache_fills"] > 0
    assert "dynamic_icache_fills_mean" in columns


def test_anamol_payload_carries_canonical_cache_geometry() -> None:
    config = load_test_config()
    run = sample_run_configs(config=config, count=1, seed=1)[0]

    payload = json.loads(_config_payload(
        (_analytical_config(run, config),),
        mechanisms=config.microarchitecture.mechanisms,
    ).decode())
    analytical = _analytical_config(run, config)
    assert payload == {
        "rob_size": analytical["rob_size"],
        "load_queue_size": analytical["lq_entries"],
        "store_queue_size": analytical["sq_entries"],
        "alu_issue_width": analytical["int_reg_issue_width"],
        "alu_mult_div_issue_width": analytical["int_mult_div_issue_width"],
        "fp_issue_width": analytical["fp_reg_issue_width"],
        "fp_mult_div_issue_width": analytical["fp_mult_div_issue_width"],
        "ls_issue_width": analytical["rdwr_port_issue_width"] + analytical["read_port_issue_width"],
        "num_ls_pipes": analytical["rdwr_port_issue_width"],
        "num_load_pipes": analytical["read_port_issue_width"],
        "max_icache_fills": analytical["max_icache_fills"],
    }


def test_dataset_build_uses_final_raw_samples_and_parallel_trace_groups(tmp_path: Path, monkeypatch) -> None:
    config = load_test_config()
    runs = sample_region_run_configs(
        config=config,
        workload_id="work",
        region_id="simpoint_0",
    )[:3]
    workload = tmp_path / "raw" / "workloads" / "work"
    region = workload / "regions" / "simpoint_0"
    trace_root = region / "trace"
    trace_root.mkdir(parents=True)
    (trace_root / dataset_module.TRACE_FILE).write_bytes(b"shared-trace")
    (trace_root / "baseline_stats.h5").write_bytes(b"stats")
    for run in runs[1:]:
        sample = region / "configs" / run.config_id
        sample.mkdir(parents=True)
        (sample / "stats.h5").write_bytes(b"stats")
        _write_run_log(sample, run.gem5_args)
    inflight = region / "configs" / f".{runs[-1].config_id}.inflight-abc"
    inflight.mkdir()
    (inflight / dataset_module.TRACE_FILE).write_bytes(b"unfinished")
    calls = []

    def fake_read_label_values(path, registry):
        if path.name == "baseline_stats.h5":
            index = 0
        else:
            index = [run.config_id for run in runs].index(path.parent.name)
        return __import__("numpy").array([[float(index + 1)] * len(config.label_columns)])

    def fake_iter_anamol_feature_batches(*, trace_path, configs, window_size, mechanisms):
        import numpy as np

        calls.append((Path(trace_path), len(configs)))
        values = np.arange(
            len(configs) * len(analytical_feature_columns(mechanisms)),
            dtype=np.float64,
        ).reshape(len(configs), len(analytical_feature_columns(mechanisms)))
        yield FeatureBatch(values=values)

    monkeypatch.setattr(dataset_module, "read_label_values", fake_read_label_values)
    monkeypatch.setattr(dataset_module, "iter_anamol_feature_batches", fake_iter_anamol_feature_batches)

    report = dataset_module.build_dataset_shards(
        config=config,
        raw_root=tmp_path / "raw",
        output_dir=tmp_path / "dataset",
        workload_ids=("work",),
        workers=2,
    )

    assert calls == [(trace_root / dataset_module.TRACE_FILE, 3)]
    assert report["shards"] == ["work.parquet"]
    assert report["workloads"] == [{"workload_id": "work", "rows": 3, "skipped": 0}]


def test_dataset_build_keeps_stats_labels_out_of_anamol_features(tmp_path: Path, monkeypatch) -> None:
    config = load_test_config()
    runs = sample_region_run_configs(
        config=config,
        workload_id="work",
        region_id="simpoint_0",
    )[:2]
    workload = tmp_path / "raw" / "workloads" / "work"
    region = workload / "regions" / "simpoint_0"
    trace_root = region / "trace"
    trace_root.mkdir(parents=True)
    (trace_root / dataset_module.TRACE_FILE).write_bytes(b"shared-trace")
    (trace_root / "baseline_stats.h5").write_bytes(b"baseline-stats")
    candidate = region / "configs" / runs[1].config_id
    candidate.mkdir(parents=True)
    (candidate / "stats.h5").write_bytes(b"candidate-stats")
    _write_run_log(candidate, runs[1].gem5_args)
    label_values = {
        "baseline_stats.h5": 101.0,
        "stats.h5": 202.0,
    }
    anamol_calls = []

    def fake_read_label_values(path, registry):
        import numpy as np

        return np.array([[label_values[Path(path).name]] * len(config.label_columns)])

    def fake_iter_anamol_feature_batches(*, trace_path, configs, window_size, mechanisms):
        import numpy as np

        anamol_calls.append((Path(trace_path), tuple(configs)))
        values = np.full(
            (len(configs), len(analytical_feature_columns(mechanisms))),
            7.0,
            dtype=np.float64,
        )
        yield FeatureBatch(values=values)

    monkeypatch.setattr(dataset_module, "read_label_values", fake_read_label_values)
    monkeypatch.setattr(dataset_module, "iter_anamol_feature_batches", fake_iter_anamol_feature_batches)

    dataset_module.build_dataset_shards(
        config=config,
        raw_root=tmp_path / "raw",
        output_dir=tmp_path / "dataset",
        workload_ids=("work",),
        workers=1,
    )

    analytical_columns = analytical_feature_columns(config.microarchitecture.mechanisms)
    frame = pq.read_table(tmp_path / "dataset" / "work.parquet").to_pandas()
    expected_runs = tuple(sorted(runs, key=lambda run: run.config_id))
    assert anamol_calls == [(
        trace_root / dataset_module.TRACE_FILE,
        tuple(_analytical_config(run, config) for run in expected_runs),
    )]
    assert frame[["workload_id", "region_id", "config_id"]].to_dict("records") == [
        {"workload_id": "work", "region_id": "simpoint_0", "config_id": run.config_id}
        for run in expected_runs
    ]
    assert frame[list(analytical_columns)].eq(7.0).all().all()
    assert frame[list(config.label_columns)].iloc[:, 0].tolist() == [
        101.0 if run.config_id == runs[0].config_id else 202.0
        for run in expected_runs
    ]


def test_dataset_build_uses_partial_raw_samples_and_rejects_invalid_run_config(tmp_path: Path, monkeypatch) -> None:
    config = load_test_config()
    runs = sample_region_run_configs(
        config=config,
        workload_id="work",
        region_id="simpoint_0",
    )[:2]
    region = tmp_path / "raw" / "workloads" / "work" / "regions" / "simpoint_0"
    trace_root = region / "trace"
    trace_root.mkdir(parents=True)
    (trace_root / dataset_module.TRACE_FILE).write_bytes(b"trace")
    (trace_root / "baseline_stats.h5").write_bytes(b"stats")
    valid = region / "configs" / runs[1].config_id
    valid.mkdir(parents=True)
    (valid / "stats.h5").write_bytes(b"stats")
    _write_run_log(valid, runs[1].gem5_args)
    def fake_read_label_values(path, *_):
        if not Path(path).is_file():
            raise OSError(path)
        return __import__("numpy").array([[1.0] * len(config.label_columns)])

    monkeypatch.setattr(dataset_module, "read_label_values", fake_read_label_values)
    monkeypatch.setattr(
        dataset_module,
        "iter_anamol_feature_batches",
        lambda *, configs, mechanisms, **__: iter((
            FeatureBatch(values=__import__("numpy").ones((len(configs), len(analytical_feature_columns(mechanisms)))),),
        )),
    )

    report = dataset_module.build_dataset_shards(
        config=config,
        raw_root=tmp_path / "raw",
        output_dir=tmp_path / "dataset",
        workload_ids=("work",),
    )

    assert report["workloads"] == [{"workload_id": "work", "rows": 2, "skipped": 0}]

    missing = region / "configs" / "config_ffffffffffffffff"
    missing.mkdir()
    _write_run_log(missing, runs[1].gem5_args)
    incomplete_report = dataset_module.build_dataset_shards(
        config=config,
        raw_root=tmp_path / "raw",
        output_dir=tmp_path / "dataset",
        workload_ids=("work",),
    )
    assert incomplete_report["workloads"] == [{"workload_id": "work", "rows": 2, "skipped": 1}]

    unknown = region / "configs" / "config_0000000000000000"
    unknown.mkdir()
    (unknown / "stats.h5").write_bytes(b"stats")
    invalid_args = tuple(
        "--fetch-width=999" if arg.startswith("--fetch-width=") else arg
        for arg in runs[0].gem5_args
    )
    _write_run_log(unknown, invalid_args)
    with __import__("pytest").raises(ValueError, match="outside its domain"):
        dataset_module.build_dataset_shards(
            config=config,
            raw_root=tmp_path / "raw",
            output_dir=tmp_path / "dataset",
            workload_ids=("work",),
        )


def _write_run_log(sample: Path, gem5_args: tuple[str, ...]) -> None:
    (sample / "run.log").write_text(
        "command line: gem5.fast sim.py " + " ".join(gem5_args) + "\n",
        encoding="utf-8",
    )
