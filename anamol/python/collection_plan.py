from __future__ import annotations

import hashlib
import random
from .run_config import RunConfig


def sample_run_configs(
    *,
    config,
    count: int,
    seed: int,
    exclude_baseline: bool = False,
) -> tuple[RunConfig, ...]:
    if count < 1:
        raise ValueError("config sample count must be positive")
    domains = _parameter_domains(config)
    total = 1
    for values in domains.values():
        total *= len(values)
    if count > total:
        raise ValueError(f"config sample count exceeds design space size: {count} > {total}")

    rng = random.Random(seed)
    samples: list[RunConfig] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(1000, count * 50)
    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        values = _random_legal_values(config, domains, rng)
        if not is_legal(values, config=config):
            continue
        candidate = _run_config(values, config=config)
        if exclude_baseline and candidate.parameter_values == config.microarchitecture.baseline_values:
            continue
        if candidate.config_id in seen:
            continue
        seen.add(candidate.config_id)
        samples.append(candidate)
    if len(samples) != count:
        raise ValueError(
            f"unable to sample {count} legal configs from design space with seed={seed}"
        )
    return tuple(samples)


def sample_region_run_configs(
    *,
    config,
    workload_id: str,
    region_id: str,
) -> tuple[RunConfig, ...]:
    seed = _stable_seed(config.collection_sampling.seed, workload_id, region_id)
    count = config.collection_sampling.configs_per_region
    baseline = _run_config({}, config=config)
    if count == 1:
        return (baseline,)
    sampled = sample_run_configs(config=config, count=count - 1, seed=seed, exclude_baseline=True)
    return (baseline, *sampled)


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256(
        ("\0".join((str(seed), *parts))).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _random_legal_values(config, domains: dict[str, tuple[int | float | str, ...]], rng: random.Random) -> dict[str, int | float | str]:
    values = {
        name: rng.choice(domains[name])
        for name in config.design_parameter_names
    }
    if {"fetch_width", "decode_width", "rename_width", "commit_width"} <= set(values):
        fetch = int(rng.choice(domains["fetch_width"]))
        decode = int(rng.choice(tuple(
            value
            for value in domains["decode_width"]
            if int(value) <= fetch
            and any(int(rename) <= int(value) for rename in domains["rename_width"])
        )))
        rename = int(rng.choice(tuple(
            value
            for value in domains["rename_width"]
            if int(value) <= decode
        )))
        commit = int(rng.choice(tuple(value for value in domains["commit_width"] if int(value) <= rename)))
        values.update(
            {
                "fetch_width": fetch,
                "decode_width": decode,
                "rename_width": rename,
                "commit_width": commit,
            }
        )

    lsu_width = int(values.get("read_port_issue_width", 0)) + int(values.get("rdwr_port_issue_width", 0))
    for queue_name in ("lq_entries", "sq_entries"):
        if queue_name in values:
            legal = tuple(value for value in domains[queue_name] if int(value) >= lsu_width)
            values[queue_name] = rng.choice(legal)
    return values


def is_legal(values: dict[str, int | str], *, config) -> bool:
    try:
        config.microarchitecture.parameter_values(values)
    except ValueError:
        return False
    return True


def _run_config(values: dict[str, int | str], *, config) -> RunConfig:
    resolved = config.microarchitecture.parameter_values(values)
    config_id = config.microarchitecture.config_id(resolved)
    return RunConfig(config_id, resolved, config.microarchitecture.gem5_args(values))


def _parameter_domains(config) -> dict[str, tuple[int | float | str, ...]]:
    return {name: config.microarchitecture.parameters_by_name[name].values for name in config.design_parameter_names}
