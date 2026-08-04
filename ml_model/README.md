# L1 surrogate and L2 search

The CPU microarchitecture repository owns the microbench roster, workload and ROI
identity, metric formulas, design domains, constraints, and full-ROI trace/statistics
evidence. The parent repository's `modeling l1` and `modeling l3` commands are the
canonical workflow entrypoints. Peregrine provides the Python APIs those commands
call for dataset construction, training, evaluation, prediction, plotting, and L2
search.

`ml_model.l2.search_l2` reuses the same CPU-owned baseline rich traces, generates
legal configurations from the loaded CPU design space, invokes the L1 predictor
directly, and writes a predicted Pareto candidate queue for real-validation planning.
