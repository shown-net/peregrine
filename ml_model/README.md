# Performance surrogate and cross-domain calibrator

The CPU microarchitecture repository owns the microbench roster, workload and ROI
identity, metric formulas, design domains, constraints, and full-ROI trace/statistics
evidence. The parent repository owns the canonical workflow entrypoints.
Peregrine provides the Python APIs for surrogate training/inference and the
N2-anchored calibrator. The calibrator consumes canonical config vectors plus
simulator metric windows and maps them directly to fixed-N2 performance intervals.
Repeated PMU samples provide target means and sample covariance; reference windows
use paired Gaussian moment supervision while other configs provide balanced OT
regularization. Inference does not consume PMU targets or OT couplings.
