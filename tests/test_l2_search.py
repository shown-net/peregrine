from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anamol.python.design_space import load_peregrine_config
from anamol.python.microarchitecture import load_microarchitecture_config
from ml_model.l2 import consume_candidate_queue
from ml_model.l2 import generate_legal_configs
from ml_model.l2 import pareto_mask
from ml_model.l2 import prioritized_pareto_indices
from tests.helpers import METRICS_CONFIG
from tests.helpers import MICROARCHITECTURE_CONFIG


def _config():
    return load_peregrine_config(
        "configs/peregrine.yaml",
        metrics_config=METRICS_CONFIG,
        microarchitecture=load_microarchitecture_config(MICROARCHITECTURE_CONFIG),
    )


def test_direct_candidates_are_distinct_canonical_legal_configurations() -> None:
    config = _config()
    candidates = generate_legal_configs(config=config, count=12, seed=19)

    assert len({candidate.config_id for candidate in candidates}) == len(candidates)
    for candidate in candidates:
        assert candidate.config_id == config.microarchitecture.config_id(candidate.parameter_values)
        config.microarchitecture.validate_values(candidate.parameter_values)


def test_pareto_queue_is_deterministic_and_contains_only_nondominated_candidates() -> None:
    objectives = np.asarray([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [3.0, 3.0]])
    ids = ("a", "b", "c", "d")

    first = prioritized_pareto_indices(objectives, ids, 3)
    second = prioritized_pareto_indices(objectives, ids, 3)

    assert first == second
    assert set(first) <= set(np.flatnonzero(pareto_mask(objectives)))
    assert 3 not in first


def test_validation_handoff_consumes_queue_priority_batches_and_prediction_evidence(tmp_path: Path) -> None:
    config = _config()
    candidate = generate_legal_configs(config=config, count=1, seed=23)[0]
    objectives = tuple(label.metric_id for label in config.labels.labels if label.role == "minimize")
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({
        "source_bundle": "/bundle/predictor_bundle.json",
        "objectives": objectives,
        "candidates": [{
            "priority": 1, "batch": 0, "config_id": candidate.config_id,
            "parameter_values": dict(candidate.parameter_values),
            "predicted_objectives": {name: float(index + 1) for index, name in enumerate(objectives)},
            "evidence": "l1_prediction",
        }],
    }))

    handoff = consume_candidate_queue(queue, config=config)
    json.dumps(handoff)

    assert handoff["source_bundle"] == "/bundle/predictor_bundle.json"
    assert handoff["objectives"] == objectives
    planned = handoff["batches"][0]["candidates"][0]
    assert planned["priority"] == 1
    assert planned["config_id"] == candidate.config_id
    assert planned["parameter_values"] == candidate.parameter_values
    assert planned["gem5_args"] == candidate.gem5_args
    assert planned["evidence"] == "l1_prediction"
    json.dumps(handoff)


def test_validation_handoff_rejects_noncanonical_objectives(tmp_path: Path) -> None:
    config = _config()
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({
        "source_bundle": "/bundle/predictor_bundle.json",
        "objectives": ["invented_weighted_score"],
        "candidates": [],
    }))

    with pytest.raises(ValueError, match="CPU metric configuration"):
        consume_candidate_queue(queue, config=config)
