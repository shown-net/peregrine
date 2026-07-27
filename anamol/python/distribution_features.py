from __future__ import annotations


PERCENTILE_POINTS = tuple(range(1, 100, 2))
FEATURES_PER_DISTRIBUTION = 2 * len(PERCENTILE_POINTS) + 1


def distribution_feature_columns(prefix: str) -> tuple[str, ...]:
    return (
        *(f"{prefix}_raw_p{int(point)}" for point in PERCENTILE_POINTS),
        *(f"{prefix}_weighted_p{int(point)}" for point in PERCENTILE_POINTS),
        f"{prefix}_mean",
    )
