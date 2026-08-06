from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

from .error_metrics import compute_regression_metrics


def plot_surrogate_generalization_errors(
    *,
    workload_ood_dir: str | Path,
    random_pair_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, dict[str, str]]:
    """Render per-metric workload error plots for the two surrogate evaluations."""
    output = Path(output_dir)
    evaluations = (
        (
            "workload_ood",
            Path(workload_ood_dir),
            "grouped_workload_kfold",
            None,
            "oof_predictions.parquet",
            "Test-level aggregate: workload-macro (equal weight per workload)",
        ),
        (
            "random_pair",
            Path(random_pair_dir),
            "random_pair_split",
            "in_distribution_random_pair",
            "test_predictions.parquet",
            "Test-level aggregate: ROI-weighted (all test ROIs pooled)",
        ),
    )
    plots: dict[str, dict[str, str]] = {}
    for name, root, protocol, scope, predictions_name, aggregation_label in evaluations:
        frame, metrics = _load_predictions(
            root=root,
            protocol=protocol,
            generalization_scope=scope,
            predictions_name=predictions_name,
        )
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        plots[name] = {
            metric: str(
                _plot_metric_errors(
                    frame,
                    metric,
                    destination / f"{_safe_name(metric)}_smape.png",
                    aggregation_label,
                )
            )
            for metric in metrics
        }
    return plots


def _load_predictions(
    *,
    root: Path,
    protocol: str,
    generalization_scope: str | None,
    predictions_name: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    evaluation_path = root / "evaluation.json"
    predictions_path = root / predictions_name
    if not evaluation_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(f"missing evaluation artifacts under {root}")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("protocol") != protocol:
        raise ValueError(f"unexpected evaluation protocol in {evaluation_path}: {evaluation.get('protocol')!r}")
    if generalization_scope is not None and evaluation.get("generalization_scope") != generalization_scope:
        raise ValueError(
            f"unexpected evaluation scope in {evaluation_path}: {evaluation.get('generalization_scope')!r}"
        )
    frame = pd.read_parquet(predictions_path)
    metrics = _prediction_metrics(frame)
    _validate_predictions(frame, metrics, predictions_path)
    return frame, metrics


def _prediction_metrics(frame: pd.DataFrame) -> tuple[str, ...]:
    metrics = tuple(sorted(column.removeprefix("truth_") for column in frame if column.startswith("truth_")))
    if not metrics:
        raise ValueError("surrogate prediction artifact has no truth_<metric> columns")
    missing = [f"prediction_{metric}" for metric in metrics if f"prediction_{metric}" not in frame]
    if missing:
        raise ValueError(f"surrogate prediction artifact is missing paired columns: {missing}")
    return metrics


def _validate_predictions(frame: pd.DataFrame, metrics: tuple[str, ...], path: Path) -> None:
    required = ("workload_id", "config_id")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"surrogate prediction artifact is missing columns: {missing}")
    if frame.empty or frame.loc[:, list(required)].isnull().any().any():
        raise ValueError(f"surrogate prediction artifact has empty or incomplete identities: {path}")
    values = frame.loc[:, [f"{kind}_{metric}" for metric in metrics for kind in ("truth", "prediction")]]
    try:
        numeric = values.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"surrogate prediction artifact has non-numeric values: {path}") from exc
    if not np.isfinite(numeric).all():
        raise ValueError(f"surrogate prediction artifact has non-finite values: {path}")


def _plot_metric_errors(frame: pd.DataFrame, metric: str, path: Path, aggregation_label: str) -> Path:
    values = frame.loc[:, ["workload_id", "config_id", f"truth_{metric}", f"prediction_{metric}"]].copy()
    families_variants = values["workload_id"].map(_family_variant)
    values["workload_family"] = [item[0] for item in families_variants]
    values["variant"] = [item[1] for item in families_variants]
    summary = pd.DataFrame(
        [
            {
                "workload_family": family,
                "variant": variant,
                "config_id": config,
                "smape_pct": compute_regression_metrics(
                    metric,
                    torch.from_numpy(group[f"prediction_{metric}"].to_numpy(dtype=np.float32, copy=True)),
                    torch.from_numpy(group[f"truth_{metric}"].to_numpy(dtype=np.float32, copy=True)),
                )["smape_pct"],
            }
            for (family, variant, config), group in values.groupby(
                ["workload_family", "variant", "config_id"], sort=True,
            )
        ]
    )
    samples = values.groupby("workload_family", sort=True).size()
    family_order = (
        summary.groupby("workload_family", sort=True)["smape_pct"].max().sort_values(ascending=False, kind="stable").index.tolist()
    )
    configs = sorted(summary["config_id"].astype(str).unique())
    variants = _ordered_variants(summary["variant"].astype(str).unique())
    config_labels = {config: f"C{index}" for index, config in enumerate(configs, start=1)}

    figure, axis = plt.subplots(figsize=(12.4, max(4.5, len(family_order) * 0.48 + 2.0)))
    colors = plt.get_cmap("tab10")
    variant_markers = ("o", "s", "^", "D", "P", "X")
    offsets = np.linspace(-0.18, 0.18, num=len(variants))
    for index, family in enumerate(family_order):
        family_rows = summary[summary["workload_family"] == family]
        for variant_index, variant in enumerate(variants):
            rows = family_rows[family_rows["variant"] == variant].set_index("config_id")
            present = [config for config in configs if config in rows.index]
            x = [float(rows.loc[config, "smape_pct"]) for config in present]
            y = index + offsets[variant_index]
            axis.plot(x, [y] * len(x), color="#9A9A9A", linewidth=0.8, zorder=1)
            for config, value in zip(present, x, strict=True):
                axis.scatter(
                    value,
                    y,
                    color=colors(configs.index(config)),
                    marker=variant_markers[variant_index],
                    s=34,
                    zorder=2,
                )
    config_handles = [
        Line2D([], [], color=colors(index), marker="o", linestyle="None", label=config_labels[config])
        for index, config in enumerate(configs)
    ]
    variant_handles = [
        Line2D([], [], color="#555555", marker=variant_markers[index], linestyle="None", label=variant)
        for index, variant in enumerate(variants)
    ]

    axis.set_yticks(range(len(family_order)))
    axis.set_yticklabels([f"{_display_family(family)} (n={samples[family]})" for family in family_order], fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Mean SMAPE%")
    axis.set_title(f"{metric}: microbench family error by configuration and variant\n{aggregation_label}", fontsize=10)
    axis.grid(True, axis="x", linestyle="--", alpha=0.35)
    configuration_legend = axis.legend(handles=config_handles, title="Configuration", loc="lower right")
    axis.add_artist(configuration_legend)
    axis.legend(handles=variant_handles, title="Variant", loc="upper right")
    figure.text(
        0.01,
        0.01,
        " | ".join(f"{label}={config}" for config, label in config_labels.items()),
        ha="left",
        va="bottom",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _safe_name(metric: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", metric).strip("_").lower()


def _family_variant(workload_id: object) -> tuple[str, str]:
    family, separator, variant = str(workload_id).rpartition("__")
    if not separator or not family or not variant:
        raise ValueError(f"workload_id must end with a variant: {workload_id!r}")
    return family, variant


def _ordered_variants(values: np.ndarray) -> tuple[str, ...]:
    preferred = ("base", "small", "large")
    present = set(values)
    return tuple(variant for variant in preferred if variant in present) + tuple(sorted(present - set(preferred)))


def _display_family(family: str) -> str:
    return family.removeprefix("microbench_")
