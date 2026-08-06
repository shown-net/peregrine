"""Canonical TorchMetrics definitions for surrogate error reporting."""

from __future__ import annotations

import torch
from torchmetrics import MetricCollection
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    SymmetricMeanAbsolutePercentageError,
    WeightedMeanAbsolutePercentageError,
)


def supports_mape(metric: str) -> bool:
    return metric in {"CPI", "BRANCH_RATE"}


def regression_metrics(metric: str) -> MetricCollection:
    metrics = {
        "mae": MeanAbsoluteError(),
        "rmse": MeanSquaredError(squared=False),
        "wmape_pct": WeightedMeanAbsolutePercentageError(),
        "smape_pct": SymmetricMeanAbsolutePercentageError(),
    }
    if supports_mape(metric):
        metrics["mape_pct"] = MeanAbsolutePercentageError()
    return MetricCollection(metrics)


def compute_regression_metrics(
    metric: str, prediction: torch.Tensor, truth: torch.Tensor,
) -> dict[str, float]:
    values = regression_metrics(metric)(prediction, truth)
    return {
        name: float(value.item() * 100.0) if name.endswith("_pct") else float(value.item())
        for name, value in values.items()
    }
