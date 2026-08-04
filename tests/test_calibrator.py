from __future__ import annotations

import numpy as np
import torch

from ml_model import calibrator
from ml_model.calibrator import _sinkhorn_divergence, build_calibrator_dataset, predict_calibrator, train_calibrator


def _records() -> list[dict[str, object]]:
    rows = []
    for workload, workload_offset in (("a", 0.0), ("b", 0.2)):
        for interval in range(2):
            label = np.asarray([1.2 + workload_offset + interval], np.float32)
            for config_id, config, proxy_offset in (("baseline", 0.0, 0.1), ("near_n2_a", 1.0, -0.2), ("near_n2_b", 2.0, 0.05)):
                rows.append({"workload_id": workload, "interval_index": interval, "config_id": config_id,
                             "config": np.asarray([config], np.float32),
                             "proxy_metrics": label + proxy_offset, "label": label})
    return rows


def test_calibrator_is_pointwise_and_uses_only_config_and_proxy_metrics(tmp_path) -> None:
    data = build_calibrator_dataset(_records(), config_names=("config",), proxy_names=("CPI",), output_names=("CPI",), reference_config_id="baseline", output_path=tmp_path / "samples.parquet")
    fitted = train_calibrator(data, epochs=4, output_path=tmp_path / "model.pt")
    direct = predict_calibrator(fitted, data)
    reloaded = predict_calibrator(tmp_path / "model.pt", tmp_path / "samples.parquet")
    assert len(direct) == len(data.workload_ids)
    assert [item.config_id for item in direct] == [item.config_id for item in reloaded]
    assert np.allclose(np.stack([item.prediction for item in direct]), np.stack([item.prediction for item in reloaded]))


def test_sinkhorn_divergence_is_independent_of_config_metadata() -> None:
    source = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
    target = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
    loss = _sinkhorn_divergence(source, target, 0.1)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(source.grad).all()


def test_calibrator_evaluates_workload_and_config_ood(monkeypatch) -> None:
    data = build_calibrator_dataset(_records(), config_names=("config",), proxy_names=("CPI",), output_names=("CPI",), reference_config_id="baseline")
    original = calibrator.train_calibrator
    monkeypatch.setattr(calibrator, "train_calibrator", lambda dataset, **_kwargs: original(dataset, epochs=2))

    report = calibrator.evaluate_calibrator(data)

    assert report["workload_ood"]["folds"] == 2
    assert report["config_ood"]["folds"] == 2
    assert set(report["workload_ood"]) == {"folds", "proxy", "calibrated"}
