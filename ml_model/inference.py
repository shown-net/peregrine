from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch

from .multitask import IDENTITY_TARGET_TRANSFORM
from .multitask import inverse_transform_targets
from .multitask import normalize_target_transforms
from .model import MultiHeadPeregrineModel


IDENTITY_COLUMNS = ("workload_id", "region_id", "config_id")


class CpuMultiHeadPredictor:
    def __init__(self, checkpoint_path: str | Path, *, num_threads: int | None = None) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        required = {
            "state_dict", "feature_columns", "label_columns", "output_metrics",
            "hidden_dims", "num_threads", "feature_mean", "feature_scale", "label_mean", "label_scale",
        }
        if missing := sorted(required - set(checkpoint)):
            raise ValueError(f"Peregrine checkpoint is missing required data: {missing}")
        self.feature_columns = tuple(checkpoint["feature_columns"])
        self.label_columns = tuple(checkpoint["label_columns"])
        self.output_metrics = tuple(checkpoint["output_metrics"])
        if len(self.label_columns) != len(self.output_metrics):
            raise ValueError("checkpoint label columns and output metrics differ")
        saved_threads = int(checkpoint["num_threads"])
        if saved_threads < 1:
            raise ValueError("checkpoint num_threads must be positive")
        if num_threads is not None and num_threads < 1:
            raise ValueError("inference num_threads must be positive")
        torch.set_num_threads(num_threads if num_threads is not None else saved_threads)
        self.feature_mean = _vector(checkpoint, "feature_mean", len(self.feature_columns))
        self.feature_scale = _vector(checkpoint, "feature_scale", len(self.feature_columns))
        self.label_mean = _vector(checkpoint, "label_mean", len(self.label_columns))
        self.label_scale = _vector(checkpoint, "label_scale", len(self.label_columns))
        self.target_transforms = normalize_target_transforms(
            self.label_columns,
            checkpoint.get("target_transforms")
            or (IDENTITY_TARGET_TRANSFORM,) * len(self.label_columns),
        )
        if np.any(self.feature_scale <= 0.0) or np.any(self.label_scale <= 0.0):
            raise ValueError("Peregrine checkpoint scales must be positive")
        self.model = MultiHeadPeregrineModel(len(self.feature_columns), tuple(checkpoint["hidden_dims"]), self.label_columns)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def predict_array(self, features: np.ndarray, *, batch_size: int = 4096) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("inference batch size must be positive")
        matrix = np.asarray(features, dtype=np.float32, order="C")
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_columns) or not np.isfinite(matrix).all():
            raise ValueError("invalid inference feature matrix")
        prediction = np.empty((len(matrix), len(self.output_metrics)), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(matrix), batch_size):
                stop = min(start + batch_size, len(matrix))
                raw = matrix[start:stop]
                normalized = (raw - self.feature_mean) / self.feature_scale
                learned = self.model(torch.from_numpy(np.ascontiguousarray(normalized))).numpy()
                transformed = learned * self.label_scale + self.label_mean
                prediction[start:stop] = inverse_transform_targets(
                    transformed,
                    self.target_transforms,
                )
        if not np.isfinite(prediction).all():
            raise ValueError("Peregrine inference produced non-finite predictions")
        return prediction


class PredictorBundle:
    """A task-level predictor composed of fixed single-head models."""

    def __init__(self, bundle_path: str | Path, *, num_threads: int | None = None) -> None:
        path = Path(bundle_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {"identity_columns", "output_metrics", "selected", "model_paths"}
        if missing := sorted(required - set(payload)):
            raise ValueError(f"predictor bundle is missing required data: {missing}")
        self.identity_columns = tuple(payload["identity_columns"])
        self.output_metrics = tuple(payload["output_metrics"])
        self.selected = dict(payload["selected"])
        if set(self.output_metrics) != set(self.selected):
            raise ValueError("predictor bundle selections differ from output metrics")
        paths = dict(payload["model_paths"])
        self.models = {
            candidate_id: CpuMultiHeadPredictor(
                _resolve_bundle_path(path.parent, model_path), num_threads=num_threads,
            )
            for candidate_id, model_path in paths.items()
        }
        for metric, entry in self.selected.items():
            if entry["predictor_kind"] != "singlehead" or entry["candidate_id"] not in self.models:
                raise ValueError(f"predictor bundle has no model for {metric}")

    @property
    def required_columns(self) -> tuple[str, ...]:
        columns = [*self.identity_columns]
        for entry in self.selected.values():
            columns.extend(self.models[entry["candidate_id"]].feature_columns)
        return tuple(dict.fromkeys(columns))

    def predict_batch(self, batch: pa.RecordBatch, *, batch_size: int) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        cache: dict[str, np.ndarray] = {}
        for metric in self.output_metrics:
            entry = self.selected[metric]
            candidate_id = entry["candidate_id"]
            predictor = self.models[candidate_id]
            if candidate_id not in cache:
                cache[candidate_id] = predictor.predict_array(
                    _feature_matrix(batch, predictor.feature_columns), batch_size=batch_size,
                )
            result[metric] = cache[candidate_id][:, predictor.output_metrics.index(metric)]
        return result


def predict_parquet(*, predictor: CpuMultiHeadPredictor, features_path: str | Path, output_path: str | Path, batch_size: int = 4096, identity_columns: tuple[str, ...] = IDENTITY_COLUMNS) -> dict[str, Any]:
    source = ds.dataset(features_path, format="parquet")
    columns = [*identity_columns, *predictor.feature_columns]
    if missing := sorted(set(columns) - set(source.schema.names)):
        raise ValueError(f"features parquet missing columns: {missing}")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    writer = None
    rows = 0
    try:
        for batch in source.to_batches(batch_size=batch_size, columns=columns):
            prediction = predictor.predict_array(
                _feature_matrix(batch, predictor.feature_columns),
                batch_size=batch_size,
            )
            output = pa.Table.from_batches([batch]).select(list(identity_columns))
            for column, values in zip(predictor.output_metrics, prediction.T, strict=True):
                output = output.append_column(f"prediction_{column}", pa.array(values))
            if writer is None:
                writer = pq.ParquetWriter(partial, output.schema, compression="zstd")
            writer.write_table(output)
            rows += output.num_rows
        if not rows:
            raise ValueError("features parquet contains no rows")
        writer.close()
        writer = None
        partial.replace(destination)
    finally:
        if writer is not None:
            writer.close()
        if partial.exists():
            partial.unlink()
    return {"rows": rows, "output": str(destination)}


def predict_bundle_parquet(*, predictor: PredictorBundle, features_path: str | Path, output_path: str | Path, batch_size: int = 4096) -> dict[str, Any]:
    source = ds.dataset(features_path, format="parquet")
    columns = list(predictor.required_columns)
    if missing := sorted(set(columns) - set(source.schema.names)):
        raise ValueError(f"features parquet missing columns: {missing}")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    writer = None
    rows = 0
    try:
        for batch in source.to_batches(batch_size=batch_size, columns=columns):
            output = pa.Table.from_batches([batch]).select(list(predictor.identity_columns))
            for metric, values in predictor.predict_batch(batch, batch_size=batch_size).items():
                output = output.append_column(f"prediction_{metric}", pa.array(values))
            if writer is None:
                writer = pq.ParquetWriter(partial, output.schema, compression="zstd")
            writer.write_table(output)
            rows += output.num_rows
        if not rows:
            raise ValueError("features parquet contains no rows")
        writer.close()
        writer = None
        partial.replace(destination)
    finally:
        if writer is not None:
            writer.close()
        if partial.exists():
            partial.unlink()
    return {"rows": rows, "output": str(destination)}

def _feature_matrix(batch: pa.RecordBatch, feature_columns: tuple[str, ...]) -> np.ndarray:
    matrix = np.empty((batch.num_rows, len(feature_columns)), dtype=np.float32)
    for index, column in enumerate(feature_columns):
        matrix[:, index] = batch.column(batch.schema.get_field_index(column)).to_numpy(
            zero_copy_only=False
        )
    return matrix


def _vector(checkpoint: dict[str, Any], key: str, size: int) -> np.ndarray:
    value = checkpoint[key]
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    if array.size != size or not np.isfinite(array).all():
        raise ValueError(f"invalid checkpoint vector: {key}")
    return array


def _resolve_bundle_path(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate
