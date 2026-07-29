from pathlib import Path
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
import zstandard as zstd

import peregrine_cli
from anamol.python.collection_plan import sample_run_configs
from anamol.python.collection_plan import is_legal
from anamol.python.collection_plan import sample_region_run_configs
from anamol.python import dataset as dataset_module
from anamol.python.dataset import _analytical_config
from anamol.python.feature_pipeline import FeatureBatch
from anamol.python.feature_pipeline import _analysis_module
from anamol.python.feature_pipeline import analytical_feature_columns
from anamol.python.feature_pipeline import mechanism_bindings
from anamol.python.feature_pipeline import trace_section_counts
from anamol.python.microarchitecture import load_microarchitecture_config
from anamol.python.run_config import RunConfig
from tests.helpers import load_test_config


def test_active_anamol_interfaces_do_not_expose_legacy_csv_or_registry() -> None:
    root = Path("anamol")

    parser_header = (root / "include/parser.h").read_text(encoding="utf-8")
    models_source = (root / "src/models.cpp").read_text(encoding="utf-8")
    feature_pipeline = (root / "python/feature_pipeline.py").read_text(encoding="utf-8")
    makefile = (root / "Makefile").read_text(encoding="utf-8")

    assert "parse_csv" not in parser_header
    assert "parse_and_convert" not in parser_header
    assert "resource_registry" not in models_source
    assert "RESOURCE_REGISTRY" not in models_source
    assert "subprocess" not in feature_pipeline
    assert "ANAMOL_BIN" not in feature_pipeline
    assert "$(ANAMOL_ROOT_PYEXT): $(OBJS_PYEXT)" in makefile
    pyext_rule = makefile.split("$(OBJDIR)/pyext_%.o:", 1)[1].split("\n\n", 1)[0]
    assert "$(GENERATED_HEADERS)" not in pyext_rule


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


def test_analytical_config_carries_configured_inputs_and_component_params() -> None:
    config = load_test_config()
    run = sample_run_configs(config=config, count=1, seed=1)[0]

    analytical = _analytical_config(run, config)
    columns = analytical_feature_columns(config.microarchitecture.mechanisms)

    assert config.microarchitecture.analysis.inputs["line_bytes"] == "cacheline_size"
    assert config.microarchitecture.analysis.inputs["l3_size_kb"] == "hnf_size"
    assert {"line_bytes", "l3_size_kb", "dram_latency_cycles"} <= set(analytical)
    assert analytical["max_icache_fills"] > 0
    assert "dynamic_icache_fills_mean" in columns


def test_components_define_analysis_models_and_columns() -> None:
    config = load_test_config()

    columns = analytical_feature_columns(config.microarchitecture.mechanisms)
    bindings = mechanism_bindings(config.microarchitecture.mechanisms)

    assert "dynamic_rob_mean" in columns
    assert "dynamic_load_store_ports_mean" in columns
    assert "dynamic_fetch_width_mean" in columns
    assert "dynamic_commit_width_mean" in columns
    assert all("model" in binding for binding in bindings)
    assert {binding["model"] for binding in bindings} >= {"width_bound", "icache_fill_slots_bound"}
    assert "formula" not in bindings[0]
    assert "gem5" not in bindings[0]


def test_analytical_columns_follow_canonical_mechanisms() -> None:
    config = load_test_config()

    columns = analytical_feature_columns(config.microarchitecture.mechanisms)
    bindings = mechanism_bindings(config.microarchitecture.mechanisms)

    assert columns[:3] == (
        "dynamic_rob_raw_p1",
        "dynamic_rob_raw_p3",
        "dynamic_rob_raw_p5",
    )
    assert "dynamic_int_reg_issue_mean" in columns
    assert "dynamic_alu_issue_mean" not in columns
    assert [item["name"] for item in bindings] == [
        mechanism.name for mechanism in config.microarchitecture.mechanisms
    ]
    assert bindings[1]["name"] == "load_queue"
    assert bindings[1]["params"] == ("lq_entries",)
    assert bindings[1]["model"] == "load_queue_capacity_latency_bound"


def test_analysis_extension_reports_configured_feature_count() -> None:
    config = load_test_config()
    engine = _analysis_module()
    bindings = mechanism_bindings(config.microarchitecture.mechanisms)

    assert engine.feature_count_for_bindings(list(bindings)) == len(
        analytical_feature_columns(config.microarchitecture.mechanisms)
    )


def test_anamol_binding_carries_canonical_cache_geometry() -> None:
    config = load_test_config()
    run = sample_run_configs(config=config, count=1, seed=1)[0]

    analytical = _analytical_config(run, config)
    bindings = mechanism_bindings(config.microarchitecture.mechanisms)

    assert bindings[-1]["name"] == "icache_fills"
    assert bindings[-1]["params"] == ("max_icache_fills",)
    assert bindings[-1]["model"] == "icache_fill_slots_bound"
    component_params = {
        param
        for binding in bindings
        for param in binding["params"]
    }
    assert component_params <= set(analytical)
    assert "load_queue_size" not in analytical
    assert "alu_issue_width" not in analytical
    assert "fp_issue_width" not in analytical


def test_microarchitecture_loader_rejects_retired_top_level_keys(tmp_path: Path) -> None:
    source = Path("../cpu_microarchitecture/configs/microarchitectures/zte_neoverse_n2.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["analysis_bindings"] = {"mechanisms": []}
    path = tmp_path / "microarchitecture.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="retired keys"):
        load_microarchitecture_config(path)


def test_microarchitecture_loader_rejects_retired_component_fields(tmp_path: Path) -> None:
    source = Path("../cpu_microarchitecture/configs/microarchitectures/zte_neoverse_n2.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["analysis"]["components"][0]["formula"] = payload["analysis"]["components"][0]["model"]
    path = tmp_path / "microarchitecture.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="retired keys"):
        load_microarchitecture_config(path)


def test_trace_section_counts_uses_in_process_protobuf_engine(tmp_path: Path) -> None:
    trace = tmp_path / "bad.pb.zst"
    trace.write_bytes(zstd.ZstdCompressor().compress(b"not a delimited protobuf"))

    with pytest.raises(RuntimeError, match="protobuf|trace"):
        trace_section_counts(trace)


def test_in_process_engine_rejects_missing_canonical_component_param(tmp_path: Path) -> None:
    engine = _analysis_module()
    trace = _minimal_trace_path(tmp_path)

    with pytest.raises(RuntimeError, match="missing parameter: rob_size"):
        engine.analyze_trace(
            str(trace),
            400,
            [_cache_config()],
            [{"name": "rob", "model": "rob_capacity_latency_bound", "params": ("rob_size",)}],
        )


def test_in_process_engine_accepts_canonical_numeric_subset(tmp_path: Path) -> None:
    engine = _analysis_module()
    trace = _minimal_trace_path(tmp_path)

    values = engine.analyze_trace(
        str(trace),
        400,
        [{"int_reg_issue_width": 3, "branch_predictor": "tage"}],
        [{"name": "renamed_int_issue", "model": "issue_width_count_bound", "params": ("int_reg_issue_width",)}],
    )

    assert values.shape == (1, 101)
    assert values[0, -1] > 0.0


def test_icache_fills_uses_configured_fetch_latency(tmp_path: Path) -> None:
    engine = _analysis_module()
    trace = _minimal_trace_path(tmp_path)
    base_config = {
        "line_bytes": 64,
        "l1i_size": 64,
        "l1d_size": 64,
        "l1_associativity": 4,
        "l1i_data_latency": 1,
        "l1d_data_latency": 1,
        "l2_size": 1024,
        "l2_associativity": 8,
        "l2_data_latency": 6,
        "l3_size_kb": 4096,
        "l3_associativity": 16,
        "hnf_data_latency": 12,
        "dram_latency_cycles": 70,
        "max_icache_fills": 1,
    }

    values = engine.analyze_trace(
        str(trace),
        400,
        [
            base_config,
            {**base_config, "l1i_data_latency": 9},
        ],
        [{"name": "icache_fills", "model": "icache_fill_slots_bound", "params": ("max_icache_fills",)}],
    )

    assert values.shape == (2, 101)
    assert values[0, -1] != values[1, -1]


def test_width_and_load_store_port_components_use_canonical_parameters(tmp_path: Path) -> None:
    engine = _analysis_module()
    trace = _minimal_trace_path(tmp_path)
    config = {
        "fetch_width": 4,
        "decode_width": 5,
        "rename_width": 6,
        "commit_width": 7,
        "read_port_issue_width": 1,
        "rdwr_port_issue_width": 2,
    }

    values = engine.analyze_trace(
        str(trace),
        400,
        [config, {**config, "fetch_width": 8, "rdwr_port_issue_width": 4}],
        [
            {"name": "renamed_fetch", "model": "width_bound", "params": ("fetch_width",)},
            {"name": "decode_width", "model": "width_bound", "params": ("decode_width",)},
            {"name": "rename_width", "model": "width_bound", "params": ("rename_width",)},
            {"name": "commit_width", "model": "width_bound", "params": ("commit_width",)},
            {
                "name": "load_store_ports",
                "model": "load_store_port_combined_bound",
                "params": ("rdwr_port_issue_width", "read_port_issue_width"),
            },
        ],
    )

    assert values.shape == (2, 5 * 101)
    fetch_mean = 100
    ls_mean = 4 * 101 + 100
    assert values[0, fetch_mean] != values[1, fetch_mean]
    assert values[0, ls_mean] != values[1, ls_mean]


def test_dataset_build_cli_passes_explicit_manifest_path(tmp_path: Path, monkeypatch) -> None:
    calls = []
    sentinel_config = object()
    sentinel_microarchitecture = object()
    monkeypatch.setattr(peregrine_cli, "load_microarchitecture_config", lambda _path: sentinel_microarchitecture)
    monkeypatch.setattr(
        peregrine_cli,
        "load_peregrine_config",
        lambda _path, *, metrics_config, microarchitecture: sentinel_config,
    )
    monkeypatch.setattr(
        peregrine_cli,
        "build_dataset_shards",
        lambda **kwargs: calls.append(kwargs) or {"dataset": str(tmp_path / "dataset"), "shards": [], "workloads": []},
    )
    args = peregrine_cli.build_parser().parse_args(
        [
            "dataset",
            "build",
            "--metrics-config",
            "metrics.yaml",
            "--microarchitecture-config",
            "micro.yaml",
            "--raw-root",
            str(tmp_path / "raw"),
            "--manifest",
            str(tmp_path / "raw" / "manifest.json"),
            "--output-dir",
            str(tmp_path / "dataset"),
        ]
    )

    assert peregrine_cli._dataset_build(args) == 0

    assert calls[0]["manifest_path"] == tmp_path / "raw" / "manifest.json"


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
    _write_sampling_manifest(tmp_path / "raw", "work", "simpoint_0", runs)
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


def test_dataset_build_consumes_sampling_manifest_and_ignores_stale_raw_samples(tmp_path: Path, monkeypatch) -> None:
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
    stale = region / "configs" / "config_ffffffffffffffff"
    stale.mkdir(parents=True)
    (stale / "stats.h5").write_bytes(b"stale-stats")
    (tmp_path / "raw" / "sampling_manifest.json").write_text(
        json.dumps(
            {
                "trace_file": dataset_module.TRACE_FILE,
                "workloads": [
                    {
                        "workload_id": "work",
                        "regions": [
                            {
                                "region_id": "simpoint_0",
                                "samples": [
                                    {
                                        "config_id": runs[0].config_id,
                                        "role": "trace_producer",
                                        "trace_path": "workloads/work/regions/simpoint_0/trace/peregrine.trace.pb.zst",
                                        "stats_path": "workloads/work/regions/simpoint_0/trace/baseline_stats.h5",
                                        "parameter_values": dict(runs[0].parameter_values),
                                    },
                                    {
                                        "config_id": runs[1].config_id,
                                        "role": "candidate",
                                        "trace_path": "workloads/work/regions/simpoint_0/trace/peregrine.trace.pb.zst",
                                        "stats_path": f"workloads/work/regions/simpoint_0/configs/{runs[1].config_id}/stats.h5",
                                        "parameter_values": dict(runs[1].parameter_values),
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    label_values = {
        "baseline_stats.h5": 101.0,
        "stats.h5": 202.0,
    }
    calls = []

    def fake_read_label_values(path, registry):
        import numpy as np

        assert "ffffffff" not in str(path)
        return np.array([[label_values[Path(path).name]] * len(config.label_columns)])

    def fake_iter_anamol_feature_batches(*, trace_path, configs, window_size, mechanisms):
        import numpy as np

        calls.append((Path(trace_path), tuple(configs)))
        yield FeatureBatch(
            values=np.ones((len(configs), len(analytical_feature_columns(mechanisms))))
        )

    monkeypatch.setattr(dataset_module, "read_label_values", fake_read_label_values)
    monkeypatch.setattr(dataset_module, "iter_anamol_feature_batches", fake_iter_anamol_feature_batches)

    report = dataset_module.build_dataset_shards(
        config=config,
        raw_root=tmp_path / "raw",
        output_dir=tmp_path / "dataset",
        workload_ids=("work",),
        workers=1,
    )

    frame = pq.read_table(tmp_path / "dataset" / "work.parquet").to_pandas()
    assert report["workloads"] == [{"workload_id": "work", "rows": 2, "skipped": 0}]
    assert frame["config_id"].astype(str).tolist() == [run.config_id for run in sorted(runs, key=lambda run: run.config_id)]
    assert calls == [(
        trace_root / dataset_module.TRACE_FILE,
        tuple(_analytical_config(run, config) for run in sorted(runs, key=lambda run: run.config_id)),
    )]


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
    _write_sampling_manifest(tmp_path / "raw", "work", "simpoint_0", runs)
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
    _write_sampling_manifest(tmp_path / "raw", "work", "simpoint_0", runs)
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
    _write_sampling_manifest(
        tmp_path / "raw",
        "work",
        "simpoint_0",
        (*runs, RunConfig("config_ffffffffffffffff", runs[1].parameter_values, runs[1].gem5_args)),
    )
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
    _write_sampling_manifest(
        tmp_path / "raw",
        "work",
        "simpoint_0",
        (
            runs[0],
            RunConfig(
                "config_0000000000000000",
                {**dict(runs[0].parameter_values), "fetch_width": 999},
                (),
            ),
        ),
    )
    with __import__("pytest").raises(ValueError, match="outside its domain"):
        dataset_module.build_dataset_shards(
            config=config,
            raw_root=tmp_path / "raw",
            output_dir=tmp_path / "dataset",
            workload_ids=("work",),
        )


def test_dataset_build_skips_manifest_sample_with_missing_label_stats(tmp_path: Path, monkeypatch) -> None:
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
    candidate = region / "configs" / runs[1].config_id
    candidate.mkdir(parents=True)
    (candidate / "stats.h5").write_bytes(b"stats")
    _write_sampling_manifest(tmp_path / "raw", "work", "simpoint_0", runs)

    def fake_read_label_values(path, *_):
        if Path(path).name == "stats.h5":
            raise KeyError("missing stat")
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

    assert report["workloads"] == [{"workload_id": "work", "rows": 1, "skipped": 1}]


def test_dataset_shard_validation_reads_only_identity_columns(tmp_path: Path, monkeypatch) -> None:
    config = load_test_config()
    shard = tmp_path / "work.parquet"
    expected_columns = [
        *dataset_module.IDENTITY_COLUMNS,
        *config.feature_columns,
        *config.label_columns,
    ]
    values = {
        "workload_id": pa.array(["work"], type=pa.string()),
        "region_id": pa.array(["simpoint_0"], type=pa.string()),
        "config_id": pa.array(["config_0000000000000000"], type=pa.string()),
    }
    values.update(
        {column: pa.array([1.0], type=pa.float64()) for column in expected_columns if column not in values}
    )
    pq.write_table(pa.table(values, schema=dataset_module._dataset_schema(config)), shard)
    requested_columns = []
    original_read_table = pq.read_table

    def recording_read_table(path, *, columns=None, **kwargs):
        requested_columns.append(tuple(columns or ()))
        return original_read_table(path, columns=columns, **kwargs)

    monkeypatch.setattr(dataset_module.pq, "read_table", recording_read_table)

    dataset_module._validate_dataset_shard(
        shard,
        workload_id="work",
        config=config,
        expected_rows=(("simpoint_0", "config_0000000000000000"),),
    )

    assert requested_columns == [dataset_module.IDENTITY_COLUMNS]


def test_dataset_build_rejects_nonfinite_generated_values_before_final_shard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_test_config()
    runs = sample_region_run_configs(
        config=config,
        workload_id="work",
        region_id="simpoint_0",
    )[:1]
    trace_root = tmp_path / "raw" / "workloads" / "work" / "regions" / "simpoint_0" / "trace"
    trace_root.mkdir(parents=True)
    (trace_root / dataset_module.TRACE_FILE).write_bytes(b"trace")
    (trace_root / "baseline_stats.h5").write_bytes(b"stats")
    _write_sampling_manifest(tmp_path / "raw", "work", "simpoint_0", runs)

    def fake_read_label_values(path, *_):
        return __import__("numpy").array([[1.0] * len(config.label_columns)])

    def fake_iter_anamol_feature_batches(*, configs, mechanisms, **__):
        import numpy as np

        values = np.ones((len(configs), len(analytical_feature_columns(mechanisms))), dtype=np.float64)
        values[0, 0] = float("nan")
        yield FeatureBatch(values=values)

    monkeypatch.setattr(dataset_module, "read_label_values", fake_read_label_values)
    monkeypatch.setattr(dataset_module, "iter_anamol_feature_batches", fake_iter_anamol_feature_batches)

    with pytest.raises(ValueError, match="non-finite"):
        dataset_module.build_dataset_shards(
            config=config,
            raw_root=tmp_path / "raw",
            output_dir=tmp_path / "dataset",
            workload_ids=("work",),
        )

    assert not (tmp_path / "dataset" / "work.parquet").exists()
    assert not (tmp_path / "dataset" / "work.parquet.partial").exists()


def _write_sampling_manifest(root: Path, workload_id: str, region_id: str, runs) -> None:
    samples = []
    for index, run in enumerate(runs):
        trace_path = f"workloads/{workload_id}/regions/{region_id}/trace/{dataset_module.TRACE_FILE}"
        stats_path = (
            f"workloads/{workload_id}/regions/{region_id}/trace/baseline_stats.h5"
            if index == 0
            else f"workloads/{workload_id}/regions/{region_id}/configs/{run.config_id}/stats.h5"
        )
        samples.append(
            {
                "config_id": run.config_id,
                "role": "trace_producer" if index == 0 else "candidate",
                "trace_path": trace_path,
                "stats_path": stats_path,
                "parameter_values": dict(run.parameter_values),
            }
        )
    (root / "sampling_manifest.json").write_text(
        json.dumps(
            {
                "trace_file": dataset_module.TRACE_FILE,
                "workloads": [
                    {
                        "workload_id": workload_id,
                        "regions": [{"region_id": region_id, "samples": samples}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _cache_config() -> dict[str, int]:
    return {
        "line_bytes": 64,
        "l1i_size": 64,
        "l1d_size": 64,
        "l1_associativity": 4,
        "l1i_data_latency": 1,
        "l1d_data_latency": 1,
        "l2_size": 1024,
        "l2_associativity": 8,
        "l2_data_latency": 6,
        "l3_size_kb": 4096,
        "l3_associativity": 16,
        "hnf_data_latency": 12,
        "dram_latency_cycles": 70,
    }


def _minimal_trace_path(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    trace = root / "minimal_anamol.pb.zst"
    if trace.exists():
        return trace
    message = b"".join(
        (
            _packed_varints(1, (0, 1)),
            _packed_varints(2, (0x1000, 0x1080)),
            _packed_varints(3, (33, 65)),
            _packed_varints(4, (0, 0)),
            _packed_varints(5, (0, 0)),
            _packed_varints(6, (0, 0)),
            _packed_varints(7, (0, 0, 0)),
            _packed_varints(9, (0, 1, 1)),
            _packed_varints(10, (0x2000,)),
            _packed_varints(11, (8,)),
            _packed_varints(12, (0, 0, 1)),
            _packed_varints(13, (0x3000,)),
            _packed_varints(14, (8,)),
            _packed_varints(19, (1, 1)),
        )
    )
    trace.write_bytes(zstd.ZstdCompressor().compress(_varint(len(message)) + message))
    return trace


def _packed_varints(field_number: int, values: tuple[int, ...]) -> bytes:
    payload = b"".join(_varint(value) for value in values)
    return _varint((field_number << 3) | 2) + _varint(len(payload)) + payload


def _varint(value: int) -> bytes:
    chunks = []
    while value >= 0x80:
        chunks.append((value & 0x7F) | 0x80)
        value >>= 7
    chunks.append(value)
    return bytes(chunks)
