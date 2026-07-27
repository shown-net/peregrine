from __future__ import annotations

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec
from importlib.util import spec_from_loader
from pathlib import Path

import yaml

from anamol.python.design_space import load_peregrine_config
from anamol.python.microarchitecture import load_microarchitecture_config


METRICS_CONFIG = Path("../cpu_microarchitecture/configs/metrics.yaml")
MICROARCHITECTURE_CONFIG = Path("../cpu_microarchitecture/configs/microarchitectures/zte_neoverse_n2.yaml")


def default_metric_set_id() -> str:
    config = yaml.safe_load(Path("configs/peregrine.yaml").read_text(encoding="utf-8"))
    return str(config["labels"]["metric_set"])


def load_test_config():
    return load_peregrine_config(
        "configs/peregrine.yaml",
        metrics_config=METRICS_CONFIG,
        microarchitecture=load_microarchitecture_config(MICROARCHITECTURE_CONFIG),
    )


def load_cli_module():
    loader = SourceFileLoader("peregrine_cli_script", str(Path("peregrine").resolve()))
    spec = spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError("failed to load Peregrine CLI script")
    module = module_from_spec(spec)
    loader.exec_module(module)
    return module
