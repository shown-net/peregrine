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

_CROSS_DOMAIN_MODELS = ("anchor", "mlp")
_CROSS_DOMAIN_COLORS = {"anchor": "#4C78A8", "mlp": "#54A24B"}

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


def plot_cross_domain_model_comparison(
    *,
    evaluation_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    """Render one truth/model comparison plot per cross-domain metric."""
    root = Path(evaluation_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evaluation, frame, metrics = _load_cross_domain_predictions(root)
    plots: dict[str, str] = {}
    for metric in metrics:
        plot_path = _plot_cross_domain_metric(
            frame,
            metric,
            evaluation,
            output / f"{_safe_name(metric)}_comparison.png",
        )
        plots[metric] = str(plot_path)
    manifest = {
        "protocol": evaluation.get("protocol"),
        "plots": plots,
    }
    (output / "plots.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plots


def _load_cross_domain_predictions(root: Path) -> tuple[dict[str, object], pd.DataFrame, tuple[str, ...]]:
    evaluation_path = root / "evaluation.json"
    predictions_path = root / "oof_predictions.parquet"
    if not evaluation_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(f"missing cross-domain evaluation artifacts under {root}")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("protocol") != "workload_kfold":
        raise ValueError(f"unexpected cross-domain evaluation protocol: {evaluation.get('protocol')!r}")
    frame = pd.read_parquet(predictions_path)
    metrics = _cross_domain_metrics(frame, evaluation)
    _validate_cross_domain_predictions(frame, metrics, predictions_path)
    return evaluation, frame, metrics


def _cross_domain_metrics(frame: pd.DataFrame, evaluation: dict[str, object]) -> tuple[str, ...]:
    configured = []
    for section in ("metrics", "derived_metrics"):
        values = evaluation.get(section) or {}
        if isinstance(values, dict):
            configured.extend(str(metric) for metric in values)
    truth_metrics = tuple(
        metric for metric in configured
        if f"truth_{metric}" in frame
    )
    partial = {
        metric: [f"{model}_{metric}" for model in _CROSS_DOMAIN_MODELS if f"{model}_{metric}" not in frame]
        for metric in truth_metrics
    }
    missing = {metric: columns for metric, columns in partial.items() if columns}
    if missing:
        raise ValueError(f"cross-domain OOF artifact is missing required model columns: {missing}")
    metrics = truth_metrics
    if not metrics:
        raise ValueError("cross-domain OOF artifact has no plottable truth/model metric columns")
    return metrics


def _validate_cross_domain_predictions(frame: pd.DataFrame, metrics: tuple[str, ...], path: Path) -> None:
    required = ("workload_id", "prefix_index")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"cross-domain OOF artifact is missing columns: {missing}")
    if frame.empty or frame.loc[:, list(required)].isnull().any().any():
        raise ValueError(f"cross-domain OOF artifact has empty or incomplete identities: {path}")
    value_columns = [
        f"{kind}_{metric}"
        for metric in metrics
        for kind in ("truth", *_CROSS_DOMAIN_MODELS)
    ]
    try:
        numeric = frame.loc[:, value_columns].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cross-domain OOF artifact has non-numeric values: {path}") from exc
    for metric in metrics:
        columns = [f"{kind}_{metric}" for kind in ("truth", *_CROSS_DOMAIN_MODELS)]
        if not np.isfinite(frame.loc[:, columns].to_numpy(dtype=np.float64)).any():
            raise ValueError(f"cross-domain OOF artifact has no finite values for metric={metric}: {path}")


def _plot_cross_domain_metric(
    frame: pd.DataFrame, metric: str, evaluation: dict[str, object], path: Path,
) -> Path:
    columns = ["workload_id", f"truth_{metric}", *[f"{model}_{metric}" for model in _CROSS_DOMAIN_MODELS]]
    values = frame.loc[:, columns].copy()
    finite = np.isfinite(values.loc[:, columns[1:]].to_numpy(dtype=np.float64)).all(axis=1)
    values = values.loc[finite].copy()
    if values.empty:
        raise ValueError(f"cross-domain OOF artifact has no jointly finite rows for metric={metric}")
    truth = values[f"truth_{metric}"].to_numpy(dtype=np.float64)
    predictions = {
        model: values[f"{model}_{metric}"].to_numpy(dtype=np.float64)
        for model in _CROSS_DOMAIN_MODELS
    }
    metric_report = ((evaluation.get("metrics") or {}).get(metric)
                     or (evaluation.get("derived_metrics") or {}).get(metric)
                     or {})
    if not isinstance(metric_report, dict):
        metric_report = {}

    figure = plt.figure(figsize=(13.8, 8.4))
    grid = figure.add_gridspec(
        2, 2, height_ratios=(1.0, 1.25), wspace=0.14, hspace=0.28,
    )
    distribution_axis = figure.add_subplot(grid[0, 0])
    density_axis = figure.add_subplot(grid[0, 1])
    residual_axis = figure.add_subplot(grid[1, :])
    _plot_truth_distribution(distribution_axis, truth)
    _plot_truth_density(density_axis, truth)
    _plot_relative_residuals(residual_axis, truth, predictions)

    kind = metric_report.get("metric_kind", "n/a")
    degenerate = metric_report.get("truth_degenerate", False)
    figure.suptitle(
        f"{metric}: PMU sample distribution and residuals | anchor baseline vs MLP | kind={kind} | degenerate={degenerate}",
        fontsize=11,
    )
    figure.text(
        0.01,
        0.01,
        f"OOF rows={len(values)}/{len(frame)} | workloads={values['workload_id'].nunique()} | {_metric_summary_label(metric_report)}",
        ha="left",
        va="bottom",
        fontsize=8,
    )
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.09, top=0.86)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _plot_truth_distribution(axis: plt.Axes, truth: np.ndarray) -> None:
    positive = truth[truth > 0.0]
    zero_count = int((truth == 0.0).sum())
    if positive.size == 0:
        axis.text(
            0.5, 0.5, f"All {len(truth)} PMU samples are zero",
            ha="center", va="center", transform=axis.transAxes,
        )
        axis.set_axis_off()
        return
    axis.ecdf(positive, color="#222222", linewidth=1.4)
    quantiles = np.percentile(positive, [0, 5, 50, 95, 100])
    labels = ("min", "p5", "p50", "p95", "max")
    colors = ("#777777", "#999999", "#222222", "#999999", "#777777")
    for index, (value, label, color) in enumerate(zip(quantiles, labels, colors, strict=True)):
        axis.axvline(value, color=color, linewidth=0.9, linestyle="--", alpha=0.8)
        axis.annotate(
            label, (value, 0.02 + 0.055 * (index % 2)), rotation=90,
            fontsize=7, ha="right", va="bottom",
        )
    _set_distribution_scale(axis, positive)
    axis.set_title(f"PMU positive-value ECDF | zero={zero_count}/{len(truth)}")
    axis.set_xlabel("Raw PMU metric value")
    axis.set_ylabel("Cumulative share of positive samples")
    axis.grid(True, linestyle="--", alpha=0.3)


def _plot_truth_density(axis: plt.Axes, truth: np.ndarray) -> None:
    positive = truth[truth > 0.0]
    if positive.size == 0:
        axis.text(0.5, 0.5, "No positive PMU samples", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        return
    if _use_log_x(positive):
        edges = np.geomspace(
            positive.min(), positive.max(), min(40, max(8, int(np.sqrt(positive.size)))),
        )
    else:
        edges = np.histogram_bin_edges(positive, bins="auto")
    counts, edges = np.histogram(positive, bins=edges)
    axis.stairs(counts, edges, fill=True, color="#4C78A8", alpha=0.7)
    _set_distribution_scale(axis, positive)
    axis.set_title("Positive-value sample density")
    axis.set_xlabel("Raw PMU metric value")
    axis.set_ylabel("Windows per bin")
    axis.grid(True, axis="y", linestyle="--", alpha=0.3)


def _plot_relative_residuals(axis: plt.Axes, truth: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    positive = truth > 0.0
    zero = ~positive
    if not positive.any():
        axis.text(
            0.5, 0.5, "No positive PMU samples: relative residual is undefined",
            ha="center", va="center", transform=axis.transAxes,
        )
        axis.set_axis_off()
        return
    floor = float(np.percentile(truth[positive], 5))
    floor = floor if floor > 0.0 and np.isfinite(floor) else float(np.min(truth[positive]))
    denominator = np.maximum(np.abs(truth[positive]), floor)
    residuals = [
        (predictions[model][positive] - truth[positive]) / denominator
        for model in _CROSS_DOMAIN_MODELS
    ]
    violin = axis.violinplot(residuals, showextrema=False, showmedians=False)
    for body, model in zip(violin["bodies"], _CROSS_DOMAIN_MODELS, strict=True):
        body.set_facecolor(_CROSS_DOMAIN_COLORS[model])
        body.set_edgecolor("none")
        body.set_alpha(0.55)
    axis.boxplot(
        residuals, widths=0.22, showfliers=False, patch_artist=True,
        boxprops={"facecolor": "none", "edgecolor": "#222222"},
        medianprops={"color": "#111111", "linewidth": 1.4},
    )
    if _use_symlog(residuals):
        axis.set_yscale("symlog", linthresh=_linthresh(tuple(residuals)))
    axis.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--")
    axis.set_xticks(range(1, len(_CROSS_DOMAIN_MODELS) + 1))
    axis.set_xticklabels(_CROSS_DOMAIN_MODELS)
    axis.set_title("Signed relative residual on positive PMU samples")
    axis.set_ylabel("(prediction - PMU) / max(|PMU|, PMU positive p5)")
    axis.grid(True, axis="y", linestyle="--", alpha=0.3)
    correlations = []
    zero_summary = []
    for model, residual in zip(_CROSS_DOMAIN_MODELS, residuals, strict=True):
        pearson, spearman = _correlations(truth[positive], predictions[model][positive])
        correlations.append(f"{model}: r={pearson:.3f}, ρ={spearman:.3f}")
        if zero.any():
            absolute = np.abs(predictions[model][zero])
            zero_summary.append(
                f"{model} zero-pred |abs| p50/p95="
                f"{np.median(absolute):.3g}/{np.percentile(absolute, 95):.3g}"
            )
    axis.text(
        0.01, 0.98, "\n".join(correlations + zero_summary), transform=axis.transAxes,
        ha="left", va="top", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.86, "pad": 3},
    )


def _use_log_x(values: np.ndarray) -> bool:
    if values.size < 2 or values.min() <= 0.0:
        return False
    low, median, high = np.percentile(values, [5, 50, 95])
    return (
        (low > 0.0 and high / low >= 100.0)
        or (median > 0.0 and values.max() / median >= 50.0)
    )


def _set_distribution_scale(axis: plt.Axes, values: np.ndarray) -> None:
    if _use_log_x(values):
        axis.set_xscale("log")


def _correlations(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    if truth.size < 2 or np.ptp(truth) == 0.0 or np.ptp(prediction) == 0.0:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(truth, prediction)[0, 1])
    ranks_truth = pd.Series(truth).rank(method="average").to_numpy(dtype=np.float64)
    ranks_prediction = pd.Series(prediction).rank(method="average").to_numpy(dtype=np.float64)
    spearman = float(np.corrcoef(ranks_truth, ranks_prediction)[0, 1])
    return pearson, spearman


def _metric_summary_label(metric_report: dict[str, object]) -> str:
    models = metric_report.get("models")
    if not isinstance(models, dict):
        return "no evaluation summary"
    parts = []
    for model in _CROSS_DOMAIN_MODELS:
        values = models.get(model)
        if not isinstance(values, dict):
            continue
        if "smape_pct" in values:
            parts.append(f"{model} sMAPE={float(values['smape_pct']):.2f}%")
        elif "average_precision" in values and "positive_wape_pct" in values:
            parts.append(
                f"{model} AP={float(values['average_precision']):.3f} +WAPE={float(values['positive_wape_pct']):.1f}%"
            )
    return " | ".join(parts)


def _use_symlog(series: list[np.ndarray]) -> bool:
    values = np.concatenate([np.asarray(item, dtype=np.float64).reshape(-1) for item in series])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    zeros = np.mean(finite == 0.0)
    magnitudes = np.abs(finite[finite != 0.0])
    if zeros >= 0.2:
        return True
    if magnitudes.size < 2:
        return False
    low, high = np.percentile(magnitudes, [5, 95])
    return low > 0.0 and high / low >= 100.0


def _linthresh(series: tuple[np.ndarray, ...]) -> float:
    values = np.concatenate([np.abs(np.asarray(item, dtype=np.float64).reshape(-1)) for item in series])
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return 1.0
    return max(float(np.percentile(positive, 5)), 1e-6)


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
