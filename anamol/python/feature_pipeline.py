from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Iterator, Mapping

import numpy as np

from .distribution_features import distribution_feature_columns
from .microarchitecture import MechanismDef


ANAMOL_ROOT = Path(__file__).resolve().parent.parent
ANAMOL_BIN = ANAMOL_ROOT / "build/main"
ANAMOL_RESOURCE_NAMES = (
    "rob",
    "load_queue",
    "store_queue",
    "alu_issue",
    "alu_mult_div_issue",
    "fp_issue",
    "fp_mult_div_issue",
    "ls_issue",
    "load_ls_pipes_lower",
    "load_ls_pipes_upper",
    "icache_fills",
)
ANAMOL_REQUIRED_PARAMS = (
    "rob_size",
    "load_queue_size",
    "store_queue_size",
    "alu_issue_width",
    "alu_mult_div_issue_width",
    "fp_issue_width",
    "fp_mult_div_issue_width",
    "ls_issue_width",
    "num_ls_pipes",
    "num_load_pipes",
    "max_icache_fills",
)


@dataclass(frozen=True)
class FeatureBatch:
    values: np.ndarray


def analytical_feature_columns(mechanisms: tuple[MechanismDef, ...]) -> tuple[str, ...]:
    del mechanisms
    return tuple(
        column
        for resource_name in ANAMOL_RESOURCE_NAMES
        for column in distribution_feature_columns(f"dynamic_{resource_name}")
    )


def trace_section_counts(
    trace_path: str | Path,
    *,
    anamol_bin: str | Path = ANAMOL_BIN,
) -> tuple[int, ...]:
    trace = Path(trace_path).resolve()
    if trace.suffixes[-2:] != [".pb", ".zst"]:
        raise ValueError(f"Anamol requires a protobuf trace: {trace}")
    result = subprocess.run(
        [
            str(Path(anamol_bin).resolve()),
            "--trace-proto",
            str(trace),
            "--validate-trace",
        ],
        cwd=ANAMOL_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Anamol trace validation failed (exit {result.returncode}): "
            f"{stderr[-4000:]}"
        )
    record = np.dtype([("instruction_count", "<u8")])
    if len(result.stdout) % record.itemsize:
        raise ValueError("Anamol validation stream ends inside a section record")
    rows = np.frombuffer(result.stdout, dtype=record)
    if not len(rows):
        raise ValueError("Anamol validation stream contains no sections")
    if len(rows) != 1:
        raise ValueError("Anamol trace file must contain exactly one region")
    counts = tuple(int(value) for value in rows["instruction_count"])
    if any(value <= 0 for value in counts):
        raise ValueError("Anamol validation stream contains an empty section")
    return counts


def iter_anamol_feature_batches(
    *,
    trace_path: str | Path,
    configs: tuple[Mapping[str, int | float], ...],
    window_size: int,
    mechanisms: tuple[MechanismDef, ...],
    anamol_bin: str | Path = ANAMOL_BIN,
) -> Iterator[FeatureBatch]:
    if not configs:
        raise ValueError("Anamol configs must not be empty")
    trace = Path(trace_path).resolve()
    if trace.suffixes[-2:] != [".pb", ".zst"]:
        raise ValueError(f"Anamol requires a protobuf trace: {trace}")
    payload = _config_payload(configs, mechanisms=mechanisms)
    process = subprocess.Popen(
        [
            str(Path(anamol_bin).resolve()),
            "--window",
            str(window_size),
            "--configs-stdin",
            "--trace-proto",
            str(trace),
        ],
        cwd=ANAMOL_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        process.stdin.write(payload)
        process.stdin.close()
        feature_count = len(analytical_feature_columns(mechanisms))
        block_bytes = len(configs) * feature_count * np.dtype("<f8").itemsize
        raw = _read_exact(process.stdout, block_bytes)
        values = np.frombuffer(raw, dtype="<f8").reshape(
            (len(configs), feature_count)
        )
        if not np.isfinite(values).all():
            raise ValueError("Anamol feature stream contains non-finite features")
        yield FeatureBatch(values=values)
        if process.stdout.read(1):
            raise ValueError("Anamol emitted more than one region")
        return_code = process.wait()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        if return_code != 0:
            raise RuntimeError(
                f"Anamol failed (exit {return_code}): {stderr[-4000:]}"
            )
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def _config_payload(configs: tuple[Mapping[str, int | float | str], ...], *, mechanisms: tuple[MechanismDef, ...]) -> bytes:
    del mechanisms
    rows: list[bytes] = []
    for index, config in enumerate(configs):
        row = _anamol_config(config)
        missing = sorted(set(ANAMOL_REQUIRED_PARAMS) - set(row))
        if missing:
            raise ValueError(f"Anamol config {index} is missing parameters: {missing}")
        rows.append((json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))
    return b"".join(rows)


def _anamol_config(config: Mapping[str, int | float | str]) -> dict[str, int]:
    rdwr = int(config["rdwr_port_issue_width"])
    read = int(config["read_port_issue_width"])
    return {
        "rob_size": int(config["rob_size"]),
        "load_queue_size": int(config["lq_entries"]),
        "store_queue_size": int(config["sq_entries"]),
        "alu_issue_width": int(config["int_reg_issue_width"]),
        "alu_mult_div_issue_width": int(config["int_mult_div_issue_width"]),
        "fp_issue_width": int(config["fp_reg_issue_width"]),
        "fp_mult_div_issue_width": int(config["fp_mult_div_issue_width"]),
        "ls_issue_width": rdwr + read,
        "num_ls_pipes": rdwr,
        "num_load_pipes": read,
        "max_icache_fills": int(config.get("max_icache_fills", 8)),
    }


def _anamol_value(name: str, value: int | float | str) -> int:
    if name == "branch_predictor":
        return {"local": 0, "tage": 1}[str(value)]
    return int(value)


def _read_block_or_eof(stream, size: int) -> bytes | None:
    first = stream.read(size)
    if not first:
        return None
    if len(first) == size:
        return first
    return first + _read_exact(stream, size - len(first))


def _read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise RuntimeError("Anamol feature stream ended inside a section block")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
