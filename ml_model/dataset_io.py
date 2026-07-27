from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


def read_dataset_shards(
    dataset_dir: str | Path,
    *,
    columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    root = Path(dataset_dir)
    shards = tuple(sorted(root.glob("*.parquet")))
    if not shards:
        raise ValueError(f"dataset directory contains no workload shards: {root}")
    return ds.dataset(shards, format="parquet").to_table(columns=list(columns)).to_pandas()
