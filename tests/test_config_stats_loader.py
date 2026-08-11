from __future__ import annotations

from pathlib import Path

import h5py

from anamol.python.dataset import load_config_stats_workload
from anamol.python.design_space import load_peregrine_config
from anamol.python.microarchitecture import load_microarchitecture_config
from tests.helpers import (
    METRICS_CONFIG,
    MICROARCHITECTURE_CONFIG,
    cpu_microarchitecture_root,
)


def _write_stats(path: Path, windows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        core = handle.require_group("board/processor/cores/core")
        core.create_dataset("numCycles", data=[100.0] * windows)
        thread = core.require_group("thread_0")
        thread.create_dataset("numInsts", data=[100.0] * windows)


def test_config_stats_loader_ignores_inflight_directories(
    monkeypatch, tmp_path: Path
) -> None:
    config = load_peregrine_config(
        cpu_microarchitecture_root() / "peregrine/configs/peregrine.yaml",
        metrics_config=METRICS_CONFIG,
        microarchitecture=load_microarchitecture_config(MICROARCHITECTURE_CONFIG),
    )
    reference = config.microarchitecture.config_id({})
    root = tmp_path / "workloads/work/configs"
    reference_root = root / reference
    reference_root.mkdir(parents=True)
    (reference_root / "peregrine.trace.pb.zst").write_bytes(b"trace")
    _write_stats(reference_root / "stats.h5", 2)
    inflight = root / ".config_deadbeefdeadbeef.inflight-test"
    inflight.mkdir()
    (inflight / "stats.h5").write_bytes(b"broken")

    monkeypatch.setattr(
        "anamol.python.dataset.read_label_values",
        lambda path, _labels: __import__("numpy").ones((2, 1))
        if path.parent == reference_root
        else (_ for _ in ()).throw(AssertionError("read inflight artifact")),
    )

    workload = load_config_stats_workload(
        config=config, raw_root=tmp_path, workload_id="work", parse_configs=False
    )

    assert [sample.config_id for sample in workload.samples] == [reference]
