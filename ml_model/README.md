# Performance surrogate and cross-domain baseline

The CPU microarchitecture repository owns the microbench roster, workload and ROI
identity, metric formulas, design domains, constraints, and full-ROI trace/statistics
evidence. The parent repository owns the canonical workflow entrypoints.
Peregrine provides the Python APIs for surrogate training and inference.

The parent repository owns the self-contained cross-domain window dataset.
Cross-domain training, evaluation, and inference use the same Lightning module,
TorchMetrics reports, and Arrow streaming inference path as the performance
surrogate.
