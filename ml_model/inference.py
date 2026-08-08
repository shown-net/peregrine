"""Streaming Arrow inference through the canonical Lightning module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch

from .module import SurrogateModule


IDENTITY_COLUMNS = ("workload_id", "window_index", "config_id")


def predict_parquet(
    *, checkpoint_path: str | Path, features_path: str | Path, output_path: str | Path,
    batch_size: int = 4096, identity_columns: tuple[str, ...] = IDENTITY_COLUMNS,
) -> dict[str, int | str]:
    if batch_size < 1:
        raise ValueError("inference batch size must be positive")
    module = SurrogateModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    module.eval()
    source = ds.dataset(features_path, format="parquet")
    columns = [*identity_columns, *module.feature_columns]
    if missing := sorted(set(columns) - set(source.schema.names)):
        raise ValueError(f"features parquet missing columns: {missing}")
    _validate_channel_schema(source.schema, module)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        with torch.inference_mode():
            for batch in source.to_batches(batch_size=batch_size, columns=columns):
                matrix = _feature_matrix(batch, module)
                prediction = module(torch.from_numpy(matrix)).cpu().numpy()
                output = pa.Table.from_batches([batch]).select(list(identity_columns))
                for metric, values in zip(module.output_metrics, prediction.T, strict=True):
                    output = output.append_column(f"prediction_{metric}", pa.array(values))
                if writer is None:
                    writer = pq.ParquetWriter(partial, output.schema, compression="zstd")
                writer.write_table(output)
                rows += output.num_rows
        if not rows:
            raise ValueError("features parquet contains no rows")
        assert writer is not None
        writer.close()
        writer = None
        partial.replace(destination)
    finally:
        if writer is not None:
            writer.close()
        partial.unlink(missing_ok=True)
    return {"rows": rows, "output": str(destination)}


def _feature_matrix(batch: pa.RecordBatch, module: SurrogateModule) -> np.ndarray:
    if not module.channel_schema:
        matrix = np.empty((batch.num_rows, len(module.feature_columns)), dtype=np.float32)
        for index, column in enumerate(module.feature_columns):
            matrix[:, index] = batch.column(batch.schema.get_field_index(column)).to_numpy(zero_copy_only=False)
        if not np.isfinite(matrix).all():
            raise ValueError("invalid inference feature matrix")
        return matrix
    values = batch.column(batch.schema.get_field_index("stats_values"))
    matrix = np.asarray(values.to_pylist(), dtype=np.float32)
    if matrix.shape != (batch.num_rows, len(module.channel_schema)):
        raise ValueError("cross-domain inference scalar channel width differs from checkpoint")
    if not np.isfinite(matrix).all() or (matrix < 0.0).any():
        raise ValueError("invalid inference feature matrix")
    return matrix[:, module.active_channel_indices]


def _validate_channel_schema(schema: pa.Schema, module: SurrogateModule) -> None:
    """Reject inference data whose scalar-channel schema differs from training."""
    if not module.channel_schema:
        return
    field = schema.field("stats_values")
    expected = json.dumps([item.__dict__ for item in module.channel_schema], sort_keys=True).encode("utf-8")
    if field.type != pa.list_(pa.float64(), len(module.channel_schema)) or (field.metadata or {}).get(b"cross_domain.channel_schema") != expected:
        raise ValueError("cross-domain scalar channel schema differs from checkpoint")
