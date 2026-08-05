# Performance surrogate and cross-domain calibrator

The CPU microarchitecture repository owns the microbench roster, workload and ROI
identity, metric formulas, design domains, constraints, and full-ROI trace/statistics
evidence. The parent repository owns the canonical workflow entrypoints.
Peregrine provides the Python APIs for surrogate training/inference and the
N2-anchored calibrator.

The calibrator consumes the parent repository's explicitly configured gem5 proxy
windows and predicts independent fixed-N2 PMU event rates. Repeated PMU runs provide
the target mean and covariance of the mean. Peregrine owns model fitting and held-out
prediction; the parent repository converts those event rates to canonical
performance proxies and evaluates them in their natural units.
