from __future__ import annotations

from pathlib import Path

import yaml

from anamol.python.design_space import load_peregrine_config
from anamol.python.microarchitecture import load_microarchitecture_config


def cpu_microarchitecture_root() -> Path:
    root = Path.cwd().resolve()
    for candidate in (root.parent, root.parent / "cpu_microarchitecture"):
        if (candidate / "configs/metrics.yaml").is_file():
            return candidate
    raise FileNotFoundError("could not locate cpu_microarchitecture config root")


METRICS_CONFIG = cpu_microarchitecture_root() / "configs/metrics.yaml"
MICROARCHITECTURE_CONFIG = cpu_microarchitecture_root() / "configs/microarchitectures/zte_neoverse_n2.yaml"


def default_metric_set_id() -> str:
    config = yaml.safe_load(Path("configs/peregrine.yaml").read_text(encoding="utf-8"))
    return str(config["labels"]["metric_set"])


def load_test_config():
    return load_peregrine_config(
        "configs/peregrine.yaml",
        metrics_config=METRICS_CONFIG,
        microarchitecture=load_microarchitecture_config(MICROARCHITECTURE_CONFIG),
    )
