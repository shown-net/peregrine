"""Workload-isolated residual evaluation for cross-domain PMU calibration."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class CrossDomainNetwork(nn.Module):
    def __init__(self, input_width: int, hidden_dims: tuple[int, int], dropout: float) -> None:
        super().__init__()
        first, second = hidden_dims
        self.network = nn.Sequential(
            nn.Linear(input_width, first), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(first, second), nn.GELU(), nn.Linear(second, 1),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _vector_column(values: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(row, dtype=np.float32) for row in values], axis=0)


def _is_count_rate(spec: object) -> bool:
    return str(spec.residual) == "count_rate"


def _correction(spec: object, anchor: np.ndarray, truth: np.ndarray, counts: np.ndarray) -> np.ndarray:
    if _is_count_rate(spec):
        quantum = 1000.0 / np.maximum(counts, 1.0)
        return (np.log1p(np.maximum(truth, 0.0) / quantum) - np.log1p(np.maximum(anchor, 0.0) / quantum)).astype(np.float32)
    return np.log(np.maximum(truth, 1e-8) / np.maximum(anchor, 1e-8)).astype(np.float32)


def _decode(spec: object, anchor: np.ndarray, correction: np.ndarray, counts: np.ndarray) -> np.ndarray:
    delta = np.clip(correction, -20.0, 20.0)
    if _is_count_rate(spec):
        quantum = 1000.0 / np.maximum(counts, 1.0)
        return (quantum * np.maximum((1.0 + np.maximum(anchor, 0.0) / quantum) * np.exp(delta) - 1.0, 0.0)).astype(np.float32)
    return (np.maximum(anchor, 0.0) * np.exp(delta)).astype(np.float32)


def _anchor_feature(spec: object, anchor: np.ndarray, counts: np.ndarray) -> np.ndarray:
    if _is_count_rate(spec):
        return np.log1p(np.maximum(anchor, 0.0) / (1000.0 / np.maximum(counts, 1.0)))
    return np.log(np.maximum(anchor, 1e-8))


def _validation_split(indices: np.ndarray, groups: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    selected = groups[indices]
    folds = min(5, len(np.unique(selected)))
    if folds < 2:
        raise ValueError("cross-domain validation requires two training workloads")
    train_local, valid_local = next(GroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(indices, groups=selected))
    return indices[train_local], indices[valid_local]


def _fit_transform(features: np.ndarray, anchor_feature: np.ndarray, correction: np.ndarray) -> dict[str, np.ndarray]:
    active = np.flatnonzero(features.std(axis=0) > 1e-8)
    if not len(active):
        raise ValueError("training fold has no varying semantic features")
    x = np.column_stack((features[:, active], anchor_feature))
    return {
        "active": active,
        "feature_mean": x.mean(axis=0, keepdims=True),
        "feature_scale": np.maximum(x.std(axis=0, keepdims=True), 1e-6),
        "target_mean": np.asarray([correction.mean()], dtype=np.float32),
        "target_scale": np.asarray([max(float(correction.std()), 1e-6)], dtype=np.float32),
    }


def _transform_features(features: np.ndarray, anchor_feature: np.ndarray, transform: dict[str, np.ndarray]) -> np.ndarray:
    x = np.column_stack((features[:, transform["active"]], anchor_feature))
    return np.clip((x - transform["feature_mean"]) / transform["feature_scale"], -20.0, 20.0).astype(np.float32)


def _transform_target(correction: np.ndarray, transform: dict[str, np.ndarray]) -> np.ndarray:
    return ((correction - transform["target_mean"][0]) / transform["target_scale"][0]).astype(np.float32)


def _restore_target(values: np.ndarray, transform: dict[str, np.ndarray]) -> np.ndarray:
    return (values * transform["target_scale"][0] + transform["target_mean"][0]).astype(np.float32)


def _fit_ridge(spec: object, train_x: np.ndarray, train_y: np.ndarray, transform: dict[str, np.ndarray], test_anchor: np.ndarray, test_counts: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    model = Ridge(alpha=1.0).fit(train_x, train_y)
    correction = _restore_target(model.predict(test_x), transform)
    return _decode(spec, test_anchor, correction, test_counts), {"model_parameters": int(model.coef_.size + 1)}


def _predict(network: nn.Module, features: np.ndarray, batch_size: int) -> np.ndarray:
    network.eval()
    with torch.inference_mode():
        values = [network(batch[0]).squeeze(1).cpu().numpy() for batch in DataLoader(TensorDataset(torch.from_numpy(features)), batch_size=batch_size)]
    result = np.concatenate(values).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("cross-domain correction produced non-finite values")
    return result


def _validation_loss(network: nn.Module, features: np.ndarray, target: np.ndarray, batch_size: int) -> float:
    network.eval()
    with torch.inference_mode():
        values = [float(nn.functional.smooth_l1_loss(network(batch[0]).squeeze(1), batch[1])) for batch in DataLoader(TensorDataset(torch.from_numpy(features), torch.from_numpy(target)), batch_size=batch_size)]
    network.train()
    return float(np.mean(values)) if values else np.inf


def _fit_mlp(task: object, train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray, valid_y: np.ndarray, transform: dict[str, np.ndarray], spec: object, test_anchor: np.ndarray, test_counts: np.ndarray, test_x: np.ndarray, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
    torch.manual_seed(seed)
    torch.set_num_threads(task.num_threads)
    network = CrossDomainNetwork(train_x.shape[1], task.hidden_dims, task.dropout)
    optimizer = torch.optim.AdamW(network.parameters(), lr=task.learning_rate, weight_decay=task.weight_decay)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=task.batch_size, shuffle=True)
    best_loss, best_state, best_epoch, stale = np.inf, None, 0, 0
    for epoch in range(task.max_epochs):
        network.train()
        for feature_batch, target_batch in loader:
            loss = nn.functional.smooth_l1_loss(network(feature_batch).squeeze(1), target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        current = _validation_loss(network, valid_x, valid_y, task.batch_size)
        if current < best_loss - 1e-6:
            best_loss, best_state, best_epoch, stale = current, copy.deepcopy(network.state_dict()), epoch + 1, 0
        else:
            stale += 1
            if stale >= task.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("cross-domain training produced no model")
    network.load_state_dict(best_state)
    correction = _restore_target(_predict(network, test_x, task.batch_size), transform)
    return _decode(spec, test_anchor, correction, test_counts), {"selected_epoch": best_epoch, "model_parameters": int(sum(parameter.numel() for parameter in network.parameters()))}


def _macro_errors(truth: np.ndarray, prediction: np.ndarray, workloads: np.ndarray) -> dict[str, float]:
    summaries = []
    for workload in np.unique(workloads):
        select = workloads == workload
        actual, estimated = truth[select], prediction[select]
        smape = 200.0 * np.mean(np.abs(estimated - actual) / np.maximum(np.abs(estimated) + np.abs(actual), 1e-8))
        wape = 100.0 * np.abs(estimated - actual).sum() / max(np.abs(actual).sum(), 1e-8)
        summaries.append((smape, wape))
    return {"smape_pct": float(np.mean([item[0] for item in summaries])), "wape_pct": float(np.mean([item[1] for item in summaries]))}


def _is_sparse_truth(truth: np.ndarray) -> bool:
    active = truth > 0.0
    return bool(active.any() and (~active).any())


def _model_summary(truth: np.ndarray, prediction: np.ndarray, workloads: np.ndarray, *, sparse: bool) -> dict[str, float]:
    if not sparse:
        return {"smape_pct": _macro_errors(truth, prediction, workloads)["smape_pct"]}
    active = truth > 0.0
    return {
        "average_precision": float(average_precision_score(active, prediction)),
        "positive_wape_pct": _macro_errors(truth[active], prediction[active], workloads[active])["wape_pct"],
    }


def _is_degenerate_truth(truth: np.ndarray) -> bool:
    return bool(np.std(truth.astype(np.float64, copy=False)) <= 1e-12)


def _best_model(summary: dict[str, object]) -> str:
    models = summary["models"]
    assert isinstance(models, dict)
    if summary["metric_kind"] == "sparse":
        return min(
            models,
            key=lambda name: (
                -float(models[name]["average_precision"]),
                float(models[name]["positive_wape_pct"]),
            ),
        )
    return min(models, key=lambda name: float(models[name]["smape_pct"]))


def _metric_summary(truth: np.ndarray, predictions: dict[str, np.ndarray], workloads: np.ndarray) -> dict[str, object]:
    sparse = _is_sparse_truth(truth)
    output = {
        "metric_kind": "sparse" if sparse else "continuous",
        "truth_degenerate": _is_degenerate_truth(truth),
        "models": {
            variant: _model_summary(truth, prediction, workloads, sparse=sparse)
            for variant, prediction in predictions.items()
        },
    }
    output["best_model"] = _best_model(output)
    output["selected_model"] = _selected_model(output)
    output["selected_regressed_from_anchor"] = _regressed_from_anchor(output, str(output["selected_model"]))
    return output


def _model_improves(summary: dict[str, object], model: str) -> bool:
    models = summary["models"]
    assert isinstance(models, dict)
    anchor, candidate = models["anchor"], models[model]
    if summary["metric_kind"] == "sparse":
        return (
            float(candidate["average_precision"]) >= float(anchor["average_precision"])
            and float(candidate["positive_wape_pct"]) < float(anchor["positive_wape_pct"])
        )
    return float(candidate["smape_pct"]) < float(anchor["smape_pct"])


def _regressed_from_anchor(summary: dict[str, object], model: str, *, tolerance_pct: float = 5.0) -> bool:
    if model == "anchor":
        return False
    models = summary["models"]
    assert isinstance(models, dict)
    anchor, candidate = models["anchor"], models[model]
    if summary["metric_kind"] == "sparse":
        return (
            float(candidate["average_precision"]) + 1e-12 < float(anchor["average_precision"])
            or float(candidate["positive_wape_pct"]) > float(anchor["positive_wape_pct"]) * (1.0 + tolerance_pct / 100.0)
        )
    return float(candidate["smape_pct"]) > float(anchor["smape_pct"]) * (1.0 + tolerance_pct / 100.0)


def _selected_model(summary: dict[str, object]) -> str:
    best = str(summary["best_model"])
    return "anchor" if _regressed_from_anchor(summary, best) else best


def _acceptance(task: object, metrics: dict[str, dict[str, object]], fold_metrics: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    improved: list[str] = []
    selected: dict[str, str] = {}
    regressed: list[str] = []
    degenerate: list[str] = []
    for spec in task.targets:
        summary = metrics[spec.metric]
        chosen = str(summary["selected_model"])
        selected[spec.metric] = chosen
        if bool(summary["truth_degenerate"]):
            degenerate.append(spec.metric)
            continue
        if _regressed_from_anchor(summary, chosen):
            regressed.append(spec.metric)
            continue
        stable = sum(_model_improves(item, "mlp") for item in fold_metrics[spec.metric]) >= 4
        if chosen != "anchor" and (_model_improves(summary, chosen) or (chosen == "mlp" and stable)):
            improved.append(spec.metric)
    evaluated = len(tuple(task.targets)) - len(degenerate)
    return {
        "selected_models": selected,
        "improved_targets": improved,
        "regressed_targets": regressed,
        "degenerate_targets": degenerate,
        "passed": not regressed and evaluated > 0 and bool(improved),
    }


def _derived(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    def ratio(numerator: str, denominator: str) -> np.ndarray:
        return np.divide(values[numerator], values[denominator], out=np.full_like(values[numerator], np.nan), where=values[denominator] > 0.0)
    output: dict[str, np.ndarray] = {}
    if "CPI" in values:
        output["IPC"] = ratio("__one", "CPI")
    if {"UOPS_PER_INST", "CPI"} <= set(values):
        output["UOPS_PER_CYCLE"] = ratio("UOPS_PER_INST", "CPI")
    if {"BRANCH_MPKI", "BRANCHES_PER_KI"} <= set(values):
        output["BRANCH_MISPRED_RATIO"] = ratio("BRANCH_MPKI", "BRANCHES_PER_KI")
    if {"MEM_READ_PER_KI", "MEM_WRITE_PER_KI"} <= set(values):
        total_memory = values["MEM_READ_PER_KI"] + values["MEM_WRITE_PER_KI"]
        output["MEMORY_ACCESSES_PER_KI"] = total_memory
        output["MEMORY_READ_RATIO"] = np.divide(values["MEM_READ_PER_KI"], total_memory, out=np.full_like(total_memory, np.nan), where=total_memory > 0.0)
        output["MEMORY_WRITE_RATIO"] = np.divide(values["MEM_WRITE_PER_KI"], total_memory, out=np.full_like(total_memory, np.nan), where=total_memory > 0.0)
    for level in ("L1D", "L2", "L3"):
        numerator, denominator = f"{level}_REFILL_MPKI", f"{level}_ACCESSES_PER_KI"
        if {numerator, denominator} <= set(values):
            output[f"{level}_REFILL_RATIO"] = ratio(numerator, denominator)
    return output


def evaluate_cross_domain(*, task: object, dataset_path: str | Path, output_dir: str | Path) -> dict[str, str]:
    frame = pq.read_table(dataset_path).to_pandas()
    required = {"workload_id", "interval_index", "instruction_count", "stats_anchors", "stats_features", *(spec.metric for spec in task.targets)}
    if frame.empty or not required <= set(frame) or frame.duplicated(["workload_id", "interval_index"]).any():
        raise ValueError("cross-domain dataset has invalid identities or columns")
    anchors, features = _vector_column(frame["stats_anchors"]), _vector_column(frame["stats_features"])
    truth = frame.loc[:, [spec.metric for spec in task.targets]].to_numpy(dtype=np.float32)
    counts = frame["instruction_count"].to_numpy(dtype=np.float32)
    if anchors.shape != (len(frame), len(task.targets)) or features.shape != (len(frame), task.stats_width) or not np.isfinite(np.concatenate((anchors, features, truth), axis=1)).all():
        raise ValueError("cross-domain dataset contains incompatible or non-finite values")
    workloads = frame["workload_id"].astype(str).to_numpy()
    folds = min(task.evaluation_folds, len(np.unique(workloads)))
    if folds != task.evaluation_folds:
        raise ValueError("cross-domain evaluation requires five workloads")
    predictions = {"anchor": anchors.copy(), "ridge": np.empty_like(truth), "mlp": np.empty_like(truth)}
    reports: list[dict[str, object]] = []
    fold_metrics = {spec.metric: [] for spec in task.targets}
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=task.seed)
    for fold, (train_valid, test) in enumerate(splitter.split(features, groups=workloads)):
        train, valid = _validation_split(train_valid, workloads, seed=task.seed + fold)
        report = {"heldout_workloads": sorted(set(workloads[test])), "validation_workloads": sorted(set(workloads[valid])), "train_rows": int(len(train)), "validation_rows": int(len(valid)), "test_rows": int(len(test)), "targets": {}}
        for index, spec in enumerate(task.targets):
            correction = _correction(spec, anchors[:, index], truth[:, index], counts)
            feature = _anchor_feature(spec, anchors[:, index], counts)
            transform = _fit_transform(features[train], feature[train], correction[train])
            train_x, valid_x, test_x = (_transform_features(features[item], feature[item], transform) for item in (train, valid, test))
            train_y, valid_y = (_transform_target(correction[item], transform) for item in (train, valid))
            ridge, ridge_report = _fit_ridge(spec, train_x, train_y, transform, anchors[test, index], counts[test], test_x)
            mlp, mlp_report = _fit_mlp(task, train_x, train_y, valid_x, valid_y, transform, spec, anchors[test, index], counts[test], test_x, seed=task.seed + fold + index)
            predictions["ridge"][test, index], predictions["mlp"][test, index] = ridge, mlp
            fold_metrics[spec.metric].append(_metric_summary(
                truth[test, index],
                {"anchor": anchors[test, index], "mlp": mlp},
                workloads[test],
            ))
            report["targets"][spec.metric] = {"active_semantic_features": int(len(transform["active"])), "ridge": ridge_report, "mlp": mlp_report}
        reports.append(report)
    return _publish(task, frame, truth, predictions, workloads, reports, fold_metrics, output_dir)


def _publish(task: object, frame: pd.DataFrame, truth: np.ndarray, predictions: dict[str, np.ndarray], workloads: np.ndarray, folds: list[dict[str, object]], fold_metrics: dict[str, list[dict[str, object]]], output_dir: str | Path) -> dict[str, str]:
    destination, partial = Path(output_dir), Path(output_dir).with_name(f".{Path(output_dir).name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    table = pa.Table.from_pandas(frame.loc[:, ["workload_id", "interval_index"]], preserve_index=False)
    report_metrics: dict[str, dict[str, object]] = {}
    values = {variant: {spec.metric: prediction[:, index] for index, spec in enumerate(task.targets)} for variant, prediction in predictions.items()}
    values["truth"] = {spec.metric: truth[:, index] for index, spec in enumerate(task.targets)}
    for index, spec in enumerate(task.targets):
        table = table.append_column(f"truth_{spec.metric}", pa.array(truth[:, index]))
        for variant, prediction in predictions.items():
            table = table.append_column(f"{variant}_{spec.metric}", pa.array(prediction[:, index]))
        report_metrics[spec.metric] = _metric_summary(
            truth[:, index],
            {variant: prediction[:, index] for variant, prediction in predictions.items()},
            workloads,
        )
        selected = str(report_metrics[spec.metric]["selected_model"])
        table = table.append_column(f"selected_{spec.metric}", pa.array(predictions[selected][:, index]))
    diagnostics: dict[str, dict[str, float]] = {}
    for metric in getattr(task, "diagnostic_ids", ()):
        diagnostic_values = frame[metric].to_numpy(dtype=np.float32)
        finite = np.isfinite(diagnostic_values)
        table = table.append_column(f"truth_{metric}", pa.array(diagnostic_values))
        workload_means = [
            diagnostic_values[(workloads == workload) & finite].mean()
            for workload in np.unique(workloads)
            if ((workloads == workload) & finite).any()
        ]
        diagnostics[metric] = {
            "finite_samples": int(finite.sum()),
            "workload_macro_mean": float(np.mean(workload_means)) if workload_means else float("nan"),
        }
    derived_metrics: dict[str, dict[str, object]] = {}
    derived_values = {variant: _derived({**item, "__one": np.ones(len(frame), dtype=np.float32)}) for variant, item in values.items()}
    for name, actual in derived_values["truth"].items():
        if name == "__one":
            continue
        valid = np.isfinite(actual)
        for variant in ("anchor", "ridge", "mlp"):
            valid &= np.isfinite(derived_values[variant][name])
        if not valid.any():
            continue
        derived_metrics[name] = _metric_summary(
            actual[valid],
            {variant: derived_values[variant][name][valid] for variant in ("anchor", "ridge", "mlp")},
            workloads[valid],
        )
        table = table.append_column(f"truth_{name}", pa.array(actual))
        for variant in ("anchor", "ridge", "mlp"):
            table = table.append_column(f"{variant}_{name}", pa.array(derived_values[variant][name]))
    oof = partial / "oof_predictions.parquet"
    pq.write_table(table, oof, compression="zstd")
    report = partial / "evaluation.json"
    report.write_text(json.dumps({"protocol": "workload_kfold", "samples": int(len(frame)), "workloads": len(set(workloads)), "outer_folds": folds, "metrics": report_metrics, "derived_metrics": derived_metrics, "diagnostics": diagnostics, "acceptance": _acceptance(task, report_metrics, fold_metrics)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if destination.exists():
        shutil.rmtree(destination)
    partial.replace(destination)
    return {"evaluation": str(destination / report.name), "oof_predictions": str(destination / oof.name)}
