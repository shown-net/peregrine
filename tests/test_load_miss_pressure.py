from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import zstandard

from anamol.python.feature_pipeline import analyze_full_roi_windows
from anamol.python.design_space import analytical_feature_columns
from anamol.python.design_space import load_peregrine_config
from anamol.python.microarchitecture import load_microarchitecture_config


def _trace_module(tmp_path: Path):
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    source = Path("/data00/xuhaoen/gem5/src/proto/peregrine_trace.proto")
    subprocess.run(
        ("protoc", f"--python_out={tmp_path}", f"-I{source.parent}", str(source)),
        check=True,
    )
    spec = importlib.util.spec_from_file_location("peregrine_trace_pb2", tmp_path / "peregrine_trace_pb2.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        spec.loader.exec_module(module)
    return module


def _write_trace(tmp_path: Path) -> Path:
    proto = _trace_module(tmp_path).AnamolTraceChunk()
    addresses = (0x100, 0x100, 0x200, 0x200)
    proto.ids.extend(range(len(addresses)))
    proto.ips.extend(0x1000 + 4 * index for index in range(len(addresses)))
    proto.class_flags.extend([32] * len(addresses))
    proto.branch_types.extend([0] * len(addresses))
    proto.branch_taken.extend([False] * len(addresses))
    proto.branch_target_addrs.extend([0] * len(addresses))
    proto.dep_offsets.extend([0] * (len(addresses) + 1))
    proto.read_offsets.extend([0] * (len(addresses) + 1))
    proto.write_offsets.extend([0] * (len(addresses) + 1))
    proto.micro_op_offsets.extend(range(len(addresses) + 1))
    proto.micro_op_ids.extend(range(1, len(addresses) + 1))
    proto.micro_op_class_flags.extend([32] * len(addresses))
    proto.micro_op_fixed_execution_latencies.extend([1] * len(addresses))
    proto.micro_op_dep_offsets.extend([0] * (len(addresses) + 1))
    proto.micro_op_read_offsets.extend(range(len(addresses) + 1))
    proto.micro_op_read_addresses.extend(addresses)
    proto.micro_op_read_sizes.extend([4] * len(addresses))
    proto.micro_op_write_offsets.extend([0] * (len(addresses) + 1))
    payload = proto.SerializeToString()
    encoded_size = bytearray()
    size = len(payload)
    while size >= 0x80:
        encoded_size.append((size & 0x7F) | 0x80)
        size >>= 7
    encoded_size.append(size)
    framed = bytes(encoded_size) + payload
    path = tmp_path / "trace.pb.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress(framed))
    return path


def test_load_miss_pressure_keeps_cache_state_across_warmup(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_peregrine_config(
        root / "peregrine/configs/peregrine.yaml",
        metrics_config=root / "configs/metrics.yaml",
        microarchitecture=load_microarchitecture_config(
            root / "configs/microarchitectures/zte_neoverse_n2.yaml"
        ),
    )
    columns = analytical_feature_columns(config.microarchitecture.mechanisms)
    features = analyze_full_roi_windows(
        trace_path=_write_trace(tmp_path),
        configs=(dict(config.microarchitecture.analytical_values(
            config.microarchitecture.parameter_values()
        )),),
        full_roi_window_size=2,
        analysis_window_size=1,
        window_count=2,
        mechanisms=config.microarchitecture.mechanisms,
    )

    assert features.shape == (1, 1, len(columns))
    assert np.isclose(features[0, 0, columns.index("dynamic_l1d_load_miss_pressure_mean")], 500.0)
    assert np.isclose(features[0, 0, columns.index("dynamic_l2_load_miss_pressure_mean")], 500.0)


def test_config_parallelism_preserves_feature_order_and_values(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_peregrine_config(
        root / "peregrine/configs/peregrine.yaml",
        metrics_config=root / "configs/metrics.yaml",
        microarchitecture=load_microarchitecture_config(root / "configs/microarchitectures/zte_neoverse_n2.yaml"),
    )
    values = dict(config.microarchitecture.analytical_values(config.microarchitecture.parameter_values()))
    common = {
        "trace_path": _write_trace(tmp_path), "configs": (values, values),
        "full_roi_window_size": 2, "analysis_window_size": 1, "window_count": 2,
        "mechanisms": config.microarchitecture.mechanisms,
    }
    serial = analyze_full_roi_windows(**common, config_threads=1)
    parallel = analyze_full_roi_windows(**common, config_threads=2)
    assert serial.shape == parallel.shape
    assert np.allclose(serial, parallel)
