from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dataset_io import read_dataset_shards
from .tasks import L1_TASK_ID


L1_CORE_METRICS = (
    "CPI",
    "BRANCH_MPKI",
    "CHI_L1I_IFETCH_MPKI",
    "CHI_L1D_LD_MPKI",
    "CHI_L2_LD_MPKI",
)


def plot_l1_summary(
    *,
    dataset_dir: str | Path,
    workload_ood_dir: str | Path,
    random_roi_dir: str | Path,
    output_dir: str | Path,
    metrics: tuple[str, ...] = L1_CORE_METRICS,
) -> dict[str, Any]:
    dataset_root = Path(dataset_dir)
    ood_root = Path(workload_ood_dir)
    random_root = Path(random_roi_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    labels = tuple(f"label_{metric}" for metric in metrics)
    label_frame = read_dataset_shards(dataset_root, columns=labels)
    ood_evaluation_path = ood_root / "evaluation.json"
    ood_predictions_path = ood_root / "oof_predictions.parquet"
    if not ood_evaluation_path.is_file() or not ood_predictions_path.is_file():
        raise FileNotFoundError(
            "missing L1 workload-OOD artifacts; run "
            "`peregrine model evaluate --task l1-surrogate --protocol workload-ood "
            "--dataset-dir <dataset> --output-dir <workload_ood_dir>`"
        )
    ood_evaluation = _read_json(ood_evaluation_path)
    if ood_evaluation.get("task_id") != L1_TASK_ID:
        raise ValueError(f"workload-OOD evaluation is not L1: {ood_evaluation_path}")
    ood_predictions = pd.read_parquet(ood_predictions_path)
    _require_prediction_columns(ood_predictions, metrics)

    random_evaluation_path = random_root / "evaluation.json"
    random_evaluation = _read_json(random_evaluation_path) if random_evaluation_path.is_file() else None
    if random_evaluation is not None and random_evaluation.get("task_id") != L1_TASK_ID:
        raise ValueError(f"random-ROI evaluation is not L1: {random_evaluation_path}")

    label_plot = output / "l1_label_distributions.png"
    protocol_plot = output / "l1_generalization_protocol_errors.png"
    workload_plot = output / "l1_workload_error_points.png"
    _plot_label_distributions(label_frame, metrics, label_plot)
    _plot_protocol_errors(ood_evaluation, random_evaluation, metrics, protocol_plot)
    _plot_workload_error_points(ood_predictions, metrics, workload_plot)

    report = {
        "task_id": L1_TASK_ID,
        "metrics": list(metrics),
        "artifact_inputs": {
            "dataset_dir": str(dataset_root),
            "workload_ood_evaluation": str(ood_evaluation_path),
            "workload_ood_predictions": str(ood_predictions_path),
            "random_roi_evaluation": str(random_evaluation_path) if random_evaluation else None,
        },
        "plots": {
            "label_distributions": str(label_plot),
            "generalization_protocol_errors": str(protocol_plot),
            "workload_error_points": str(workload_plot),
        },
        "label_distribution": {
            metric: _distribution(label_frame[f"label_{metric}"].to_numpy(dtype=np.float64))
            for metric in metrics
        },
        "protocol_errors": _protocol_summary(ood_evaluation, random_evaluation, metrics),
        "workload_errors": _workload_summary(ood_predictions, metrics),
        "rows": {
            "dataset": int(len(label_frame)),
            "workload_ood": int(len(ood_predictions)),
        },
        "random_roi_available": random_evaluation is not None,
    }
    summary_path = output / "l1_plot_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": str(summary_path), **report}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_prediction_columns(frame: pd.DataFrame, metrics: tuple[str, ...]) -> None:
    missing = [
        column
        for metric in metrics
        for column in (f"truth_{metric}", f"prediction_{metric}")
        if column not in frame
    ]
    if missing:
        raise ValueError(f"L1 prediction artifact is missing columns: {missing}")
    if "workload_id" not in frame:
        raise ValueError("L1 prediction artifact is missing workload_id")


def _distribution(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("cannot summarize empty values")
    return {
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _smape_percent(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    denominator = np.abs(truth) + np.abs(prediction)
    return np.divide(
        200.0 * np.abs(truth - prediction),
        denominator,
        out=np.zeros_like(truth, dtype=np.float64),
        where=denominator != 0.0,
    )


def _error_summary(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    absolute = np.abs(truth - prediction).astype(np.float64)
    magnitude = float(np.abs(truth).sum())
    nonzero = truth != 0.0
    return {
        "rows": int(len(truth)),
        "mae": float(np.mean(absolute)),
        "median_smape_pct": float(np.percentile(_smape_percent(truth, prediction), 50)),
        "p90_smape_pct": float(np.percentile(_smape_percent(truth, prediction), 90)),
        "mean_smape_pct": float(np.mean(_smape_percent(truth, prediction))),
        "wape_pct": float(absolute.sum() / magnitude * 100.0) if magnitude else None,
        "mape_pct": float((absolute[nonzero] / np.abs(truth[nonzero])).mean() * 100.0)
        if nonzero.any() else None,
    }


def _metric_report(evaluation: dict[str, Any] | None, aggregation: str, metric: str) -> dict[str, Any] | None:
    if evaluation is None:
        return None
    return (((evaluation.get("metrics") or {}).get(aggregation) or {}).get(metric))


def _protocol_summary(
    ood_evaluation: dict[str, Any],
    random_evaluation: dict[str, Any] | None,
    metrics: tuple[str, ...],
) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for metric in metrics:
        ood = _metric_report(ood_evaluation, "macro_workload", metric)
        random = _metric_report(random_evaluation, "roi_weighted", metric)
        if ood is None:
            raise ValueError(f"workload-OOD evaluation lacks macro_workload metric: {metric}")
        ood_value = float(ood["smape_pct"])
        random_value = None if random is None else float(random["smape_pct"])
        summary[metric] = {
            "workload_ood_smape_pct": ood_value,
            "random_roi_smape_pct": random_value,
            "gap_smape_pct": None if random_value is None else ood_value - random_value,
            "workload_ood_wape_pct": float(ood["wape_pct"]),
            "random_roi_wape_pct": None if random is None else float(random["wape_pct"]),
        }
    return summary


def _workload_summary(frame: pd.DataFrame, metrics: tuple[str, ...]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    report: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for metric in metrics:
        metric_report = {}
        for workload, group in frame.groupby("workload_id", sort=True):
            truth = group[f"truth_{metric}"].to_numpy(dtype=np.float64)
            prediction = group[f"prediction_{metric}"].to_numpy(dtype=np.float64)
            metric_report[str(workload)] = _error_summary(truth, prediction)
        report[metric] = metric_report
    return report


def _grid(metrics: tuple[str, ...], *, width: float = 4.8, height: float = 3.4) -> tuple[plt.Figure, np.ndarray]:
    columns = min(3, len(metrics))
    rows = int(np.ceil(len(metrics) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows))
    return figure, np.atleast_1d(axes).reshape(rows, columns)


def _hide_unused_axes(axes: np.ndarray, used: int) -> None:
    for axis in axes.ravel()[used:]:
        axis.set_visible(False)


def _plot_label_distributions(frame: pd.DataFrame, metrics: tuple[str, ...], path: Path) -> None:
    figure, axes = _grid(metrics)
    for axis, metric in zip(axes.ravel(), metrics, strict=False):
        values = frame[f"label_{metric}"].to_numpy(dtype=np.float64)
        stats = _distribution(values)
        axis.hist(values, bins=36, color="#4C78A8", alpha=0.85)
        axis.axvline(stats["p50"], color="#222222", linestyle="-", linewidth=1.0, label="p50")
        axis.axvline(stats["p90"], color="#E45756", linestyle="--", linewidth=1.0, label="p90")
        axis.set_title(metric)
        axis.set_xlabel("Label value")
        axis.set_ylabel("Rows")
        axis.grid(True, axis="y", linestyle="--", alpha=0.3)
    _hide_unused_axes(axes, len(metrics))
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_protocol_errors(
    ood_evaluation: dict[str, Any],
    random_evaluation: dict[str, Any] | None,
    metrics: tuple[str, ...],
    path: Path,
) -> None:
    x = np.arange(len(metrics))
    width = 0.34 if random_evaluation else 0.5
    ood = [float(_metric_report(ood_evaluation, "macro_workload", metric)["smape_pct"]) for metric in metrics]
    random = [
        None if random_evaluation is None else float(_metric_report(random_evaluation, "roi_weighted", metric)["smape_pct"])
        for metric in metrics
    ]
    figure, axis = plt.subplots(figsize=(max(9, len(metrics) * 1.8), 5.2))
    axis.bar(x - (width / 2 if random_evaluation else 0), ood, width, label="workload-OOD macro", color="#4C78A8")
    if random_evaluation:
        axis.bar(x + width / 2, random, width, label="random-ROI test", color="#F58518")
        for idx, (ood_value, random_value) in enumerate(zip(ood, random, strict=True)):
            axis.plot([idx - width / 2, idx + width / 2], [ood_value, random_value], color="#666666", linewidth=0.9, alpha=0.5)
    axis.set_xticks(x)
    axis.set_xticklabels(metrics, rotation=25, ha="right")
    axis.set_ylabel("SMAPE%")
    axis.set_title("L1 Generalization Error by Protocol")
    axis.grid(True, axis="y", linestyle="--", alpha=0.35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_workload_error_points(frame: pd.DataFrame, metrics: tuple[str, ...], path: Path) -> None:
    figure, axes = _grid(metrics, width=5.2, height=3.8)
    rng = np.random.default_rng(12345)
    for axis, metric in zip(axes.ravel(), metrics, strict=False):
        rows = []
        for workload, group in frame.groupby("workload_id", sort=True):
            truth = group[f"truth_{metric}"].to_numpy(dtype=np.float64)
            prediction = group[f"prediction_{metric}"].to_numpy(dtype=np.float64)
            values = _smape_percent(truth, prediction)
            rows.append((str(workload), values, float(np.median(values)), float(np.mean(values))))
        rows.sort(key=lambda item: item[2], reverse=True)
        labels = [item[0] for item in rows]
        for index, (_workload, values, median, mean) in enumerate(rows):
            jitter = rng.uniform(-0.18, 0.18, size=len(values))
            axis.scatter(np.full(len(values), index) + jitter, values, s=10, alpha=0.35, color="#4C78A8")
            axis.scatter(index, median, s=35, color="#E45756", marker="D", zorder=3)
            axis.scatter(index, mean, s=35, color="#222222", marker="_", zorder=3)
        axis.set_title(metric)
        axis.set_ylabel("Sample SMAPE%")
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
        axis.grid(True, axis="y", linestyle="--", alpha=0.3)
    _hide_unused_axes(axes, len(metrics))
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
