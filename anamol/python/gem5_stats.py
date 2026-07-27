from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml

from .formula import evaluate_formula
from .formula import formula_name


@dataclass(frozen=True)
class TraceLabel:
    metric_id: str
    stats: tuple[str, ...]
    formula: str
    formula_inputs: tuple[tuple[str, str], ...]

    @property
    def label_column(self) -> str:
        return f"label_{self.metric_id}"


@dataclass(frozen=True)
class TraceLabelRegistry:
    labels: tuple[TraceLabel, ...]

    @property
    def wanted_stats(self) -> tuple[str, ...]:
        out: list[str] = []
        for label in self.labels:
            for stat in label.stats:
                if stat not in out:
                    out.append(stat)
        return tuple(out)


def load_label_registry(path: str | Path, *, metric_set_id: str) -> TraceLabelRegistry:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return label_registry_from_metrics(payload.get("metrics"), metric_set_id=metric_set_id)


def label_registry_from_metrics(metrics: Any, *, metric_set_id: str) -> TraceLabelRegistry:
    if not isinstance(metrics, dict):
        raise ValueError("metrics config must be a mapping")
    definitions = metrics.get("definitions")
    metric_sets = metrics.get("metric_sets")
    if not isinstance(metric_sets, dict) or metric_set_id not in metric_sets:
        raise ValueError(f"unknown label metric set: {metric_set_id}")
    metric_ids = tuple(str(item) for item in metric_sets[metric_set_id])
    if not isinstance(definitions, dict) or not metric_ids:
        raise ValueError("metrics config must define target metric ids and definitions")
    labels = []
    for metric_id in metric_ids:
        node = definitions[metric_id]
        gem5 = node["gem5"]
        formula_inputs = tuple(
            (formula_name(str(stat)), str(stat))
            for stat in gem5["stats"]
        )
        aliases = [alias for alias, _stat in formula_inputs]
        if len(set(aliases)) != len(aliases):
            raise ValueError(f"gem5 formula aliases collide for metric_id={metric_id}")
        labels.append(
            TraceLabel(
                metric_id=metric_id,
                stats=tuple(str(stat) for stat in gem5["stats"]),
                formula=str(gem5["formula"]),
                formula_inputs=formula_inputs,
            )
        )
    return TraceLabelRegistry(tuple(labels))


def read_label_frame(path: str | Path, registry: TraceLabelRegistry) -> pd.DataFrame:
    values = read_label_values(path, registry)
    return pd.DataFrame(
        values,
        columns=[label.label_column for label in registry.labels],
    )


def read_label_values(path: str | Path, registry: TraceLabelRegistry) -> np.ndarray:
    stats_path = Path(path)
    if not stats_path.exists():
        raise FileNotFoundError(f"missing gem5 stats.h5: {stats_path}")
    resolved: dict[str, np.ndarray] = {}
    datasets: dict[str, np.ndarray] = {}
    with h5py.File(stats_path, "r") as handle:
        section_count = 0
        for stat in registry.wanted_stats:
            dataset = _resolve_dataset(handle, stat)
            if dataset is None:
                continue
            values = datasets.get(dataset.name)
            if values is None:
                values = np.asarray(dataset[()], dtype=np.float64)
                datasets[dataset.name] = values
            column = _dataset_values(values, dataset.attrs, stat)
            if column.size == 0:
                continue
            resolved[stat] = column
            section_count = max(section_count, len(column))
    missing = [stat for stat in registry.wanted_stats if stat not in resolved]
    if missing:
        raise KeyError(f"missing stats in {stats_path}: {missing}")
    lengths = {stat: len(values) for stat, values in resolved.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"gem5 stats section counts differ in {stats_path}: {lengths}")
    columns: list[np.ndarray] = []
    for label in registry.labels:
        inputs = {
            alias: resolved[stat].astype(np.float64, copy=False)
            for alias, stat in label.formula_inputs
        }
        columns.append(np.asarray(evaluate_formula(label.formula, inputs), dtype=np.float64))
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise ValueError(f"gem5 metrics contain non-finite values: {stats_path}")
    return matrix

def _resolve_dataset(handle: h5py.File, stat: str) -> h5py.Dataset | None:
    node: h5py.Group | h5py.Dataset = handle
    remainder = stat.rsplit("::", 1)[0]
    while remainder:
        if isinstance(node, h5py.Dataset):
            return None
        if remainder in node:
            candidate = node[remainder]
            return candidate if isinstance(candidate, h5py.Dataset) else None
        head, sep, tail = remainder.partition(".")
        if not sep:
            return None
        if head in node and isinstance(node[head], h5py.Group):
            node = node[head]
            remainder = tail
            continue
        if remainder in node and isinstance(node[remainder], h5py.Dataset):
            return node[remainder]
        return None
    return None


def _dataset_values(
    values: np.ndarray,
    attrs: h5py.AttributeManager,
    stat: str,
) -> np.ndarray:
    if "::" not in stat:
        return values.reshape(-1)
    subname = stat.rsplit("::", 1)[1]
    if subname == "total":
        return values.reshape((values.shape[0], -1)).sum(axis=1)
    for attr_name in ("subnames", "y_subnames", "x_subnames"):
        raw = attrs.get(attr_name)
        if raw is None:
            continue
        names = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in raw]
        if subname not in names:
            continue
        index = names.index(subname)
        if values.ndim == 2:
            return values[:, index].reshape(-1)
        if values.ndim == 3:
            selected = values[:, index, :] if attr_name == "x_subnames" else values[:, :, index]
            return selected.reshape((selected.shape[0], -1)).sum(axis=1)
    return np.asarray([], dtype=np.float64)
