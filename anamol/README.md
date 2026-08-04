# Anamol

Anamol is the in-process analytical feature engine used by the Peregrine
Python APIs. The parent `cpu_microarchitecture` repository owns data
collection and workflow orchestration; Anamol no longer exposes a standalone
training or sweep pipeline in this repository.

## Active Role

- `anamol/python/feature_pipeline.py` calls the compiled `_analysis` extension
  to turn CPU-owned Peregrine traces into analytical feature batches.
- `anamol/python/dataset.py` combines those feature batches with CPU-owned
  gem5 statistics to build L1 surrogate datasets.
- `registry.yaml` and the generated headers define the analytical resources
  compiled into the extension.

## Build

Build the Python extension from this directory:

```bash
make PYTHON=/path/to/python GEM5_ROOT=/path/to/gem5 python-extension
```

The extension output `_analysis*.so` and `build/` are local build artifacts and
must remain untracked.

## Maintained Files

- `registry.yaml`: source for analytical resources and generated bindings.
- `src/` and `include/`: C++ parser, analytical models, and extension code.
- `python/feature_pipeline.py`: active Python entrypoint used by L1 dataset
  construction.
- `python/gen_registry.py`: generator used by the Makefile when
  `registry.yaml` changes.

Historical lookup-table, sweep-to-training, and standalone training scripts
were removed when the parent repository became the canonical workflow owner.
