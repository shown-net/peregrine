# L1 surrogate

The CPU microarchitecture repository owns the microbench roster, workload and ROI
identity, metric formulas, design domains, constraints, and full-ROI trace/statistics
evidence. Peregrine consumes those sources in place: `peregrine dataset build` creates
derived analytical shards, `peregrine model evaluate` performs configuration-grouped
evaluation, and `peregrine model train` creates the deployment predictor.

Current options and paths are documented by each command's `--help` output.
