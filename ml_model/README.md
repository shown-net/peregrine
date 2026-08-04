# Performance surrogate and cross-domain calibrator

The CPU microarchitecture repository owns the microbench roster, workload and ROI
identity, metric formulas, design domains, constraints, and full-ROI trace/statistics
evidence. The parent repository owns the canonical workflow entrypoints.
Peregrine provides the Python APIs for surrogate training/inference and the
N2-anchored calibrator. The calibrator consumes canonical config vectors plus
frozen surrogate metric predictions aggregated to the PMU observation window.
It learns a pointwise residual map with paired baseline supervision and
balanced Sinkhorn distribution alignment; it does not consume trace, raw stats,
or analytical feature summaries.
