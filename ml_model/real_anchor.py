"""Peregrine-owned real-anchored cross-domain dataset and multi-head training."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from anamol.python.formula import evaluate_formula, formula_name
from anamol.python.gem5_stats import label_registry_from_metrics, read_label_values



IDENTITY_COLUMNS = ("workload_id", "interval_index")
TARGET_SET = "real_anchor_core"
FEATURE_SET = "gem5_surrogate_core"
P1_FEATURES = tuple(f"feature_gem5_{name}" for name in ("CPI", "BRANCH_MPKI", "CHI_L1I_IFETCH_MPKI", "CHI_L1D_LD_MPKI", "CHI_L2_LD_MPKI"))
P2_FEATURES = (*P1_FEATURES, "feature_gem5_BRANCH_RATE", "feature_gem5_MEM_READ_PER_KI", "feature_gem5_MEM_WRITE_PER_KI")


def build_real_anchor_dataset(*, raw_root: str | Path, metrics_config: str | Path, output_dir: str | Path, workload_ids: tuple[str, ...] | None = None) -> dict[str, Any]:
    raw = Path(raw_root)
    metrics = (yaml.safe_load(Path(metrics_config).read_text(encoding="utf-8")) or {})["metrics"]
    targets = label_registry_from_metrics(metrics, metric_set_id=TARGET_SET)
    features = label_registry_from_metrics(metrics, metric_set_id=FEATURE_SET)
    target_names = tuple(label.metric_id for label in targets.labels)
    feature_names = tuple(label.metric_id for label in features.labels)
    selected = workload_ids or tuple(sorted(path.name for path in (raw / "gem5_full_roi" / "workloads").iterdir() if path.is_dir()))
    if not selected:
        raise ValueError("real-anchor workload selection is empty")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports, rejected = [], Counter()
    for workload_id in selected:
        try:
            gem5 = read_label_values(raw / "gem5_full_roi" / "workloads" / workload_id / "baseline" / "baseline_stats.h5", features)
            real, tails = _pmu_targets(raw / "pmu" / "workloads" / workload_id / "pmu_groups", metrics, target_names)
        except (FileNotFoundError, KeyError, ValueError, TypeError) as error:
            raise ValueError(f"cannot build real-anchor dataset for {workload_id}: {error}") from error
        rows = []
        for index in range(max(len(gem5), max(real, default=-1) + 1)):
            if index in tails:
                rejected["tail"] += 1
                continue
            if index >= len(gem5):
                rejected["missing_gem5"] += 1
                continue
            if index not in real:
                rejected["missing_pmu_group"] += 1
                continue
            row: dict[str, object] = {"workload_id": workload_id, "interval_index": index}
            row.update({f"feature_gem5_{name}": float(value) for name, value in zip(feature_names, gem5[index], strict=True)})
            row.update({f"label_{name}": float(real[index][name]) for name in target_names})
            rows.append(row)
        if not rows:
            raise ValueError(f"real-anchor dataset has no admitted rows: {workload_id}")
        pq.write_table(pa.Table.from_pylist(rows), out / f"{workload_id}.parquet", compression="zstd")
        reports.append({"workload_id": workload_id, "rows": len(rows)})
    return {"dataset_dir": str(out), "workloads": reports, "rejected": dict(sorted(rejected.items()))}


def _pmu_targets(root: Path, metrics: dict[str, Any], target_names: tuple[str, ...]) -> tuple[dict[int, dict[str, float]], set[int]]:
    rows: dict[int, dict[str, float]] = {}
    tails: set[int] = set()
    for metric in target_names:
        frame = pd.read_csv(root / metric / "segments.csv")
        required = {"segment_index", "is_tail_segment", "pmu_running_pct", *metrics["definitions"][metric]["pmu"]["events"]}
        if not required <= set(frame):
            raise ValueError(f"PMU group lacks required columns: {metric}")
        definition = metrics["definitions"][metric]["pmu"]
        aliases = {formula_name(str(event)): str(event) for event in definition["events"]}
        for item in frame.itertuples(index=False):
            index = int(item.segment_index)
            if str(item.is_tail_segment).strip().lower() in {"1", "true", "yes"}:
                tails.add(index)
                continue
            if float(item.pmu_running_pct) != 100.0:
                continue
            inputs = {alias: float(getattr(item, event)) for alias, event in aliases.items()}
            value = float(evaluate_formula(str(definition["formula"]), inputs))
            if not math.isfinite(value):
                continue
            rows.setdefault(index, {})[metric] = value
    return {index: values for index, values in rows.items() if set(values) == set(target_names)}, tails


def target_names() -> tuple[str, ...]:
    return tuple(name.removeprefix("feature_gem5_") for name in P1_FEATURES)
