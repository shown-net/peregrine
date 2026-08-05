# Anamol

Anamol is the in-process analytical feature engine used by the Peregrine
Python APIs. The parent `cpu_microarchitecture` repository owns data
collection and workflow orchestration; Anamol no longer exposes a standalone
training or sweep pipeline in this repository.

## Active Role

- `anamol/python/feature_pipeline.py` calls the compiled `_analysis` extension
  to turn one CPU-owned full-ROI trace into causally continuous feature windows.
- `anamol/python/dataset.py` aligns those windows with CPU-owned gem5 statistics
  after the shared warm-up interval.

## Build

Build the Python extension from this directory:

```bash
make PYTHON=/path/to/python GEM5_ROOT=/path/to/gem5 python-extension
```

The extension output `_analysis*.so` and `build/` are local build artifacts and
must remain untracked.

## Maintained Files

- `src/` and `include/`: C++ parser, causal component states, and extension code.
- `python/feature_pipeline.py`: active Python entrypoint used by L1 dataset
  construction.
Historical random-region, registry-generated, and standalone analysis paths
were removed; the full-ROI extension is the only analytical entrypoint.
