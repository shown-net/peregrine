# L1 surrogate and L2 search

The CPU microarchitecture repository owns the microbench roster, workload and ROI
identity, metric formulas, design domains, constraints, and full-ROI trace/statistics
evidence. Peregrine consumes those sources in place: `peregrine dataset build` creates
derived analytical shards, `peregrine model evaluate` performs configuration-grouped
evaluation, and `peregrine model train` creates the deployment predictor.

`peregrine l2 search` reuses the same CPU-owned baseline rich traces, generates legal
configurations from the loaded CPU design space, invokes the L1 predictor directly,
and writes a predicted Pareto candidate queue for real-validation planning. Current
options and paths are documented by each command's `--help` output.
