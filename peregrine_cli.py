#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from anamol.python.design_space import load_peregrine_config
from anamol.python.microarchitecture import load_microarchitecture_config
from anamol.python.dataset import build_dataset_shards
from ml_model.inference import CpuMultiHeadPredictor
from ml_model.inference import predict_parquet
from ml_model.train import train_surrogate


def _print_json(payload: dict) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _dataset_build(args: argparse.Namespace) -> int:
    config = load_peregrine_config(args.config, metrics_config=args.metrics_config, microarchitecture=load_microarchitecture_config(args.microarchitecture_config))
    report = build_dataset_shards(
        config=config,
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        workload_ids=tuple(args.workload_id or ()) or None,
        manifest_path=args.manifest,
        workers=args.workers,
    )
    return _print_json(report)


def _model_train(args: argparse.Namespace) -> int:
    config = load_peregrine_config(args.config, metrics_config=args.metrics_config, microarchitecture=load_microarchitecture_config(args.microarchitecture_config))
    return _print_json(_train_from_args(args, config=config))


def _model_predict(args: argparse.Namespace) -> int:
    return _print_json(_predict_from_args(args))


def _selected_workloads(dataset_dir: str | Path, workload_ids: list[str]) -> tuple[str, ...]:
    return tuple(workload_ids or ()) or tuple(
        path.stem for path in sorted(Path(dataset_dir).glob("*.parquet"))
    )


def _train_from_args(
    args: argparse.Namespace,
    *,
    config,
    feature_columns: tuple[str, ...] | None = None,
    label_columns: tuple[str, ...] | None = None,
    output_metrics: tuple[str, ...] | None = None,
) -> dict:
    report = train_surrogate(
        config=config,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        workload_ids=tuple(args.workload_id or ()) or None,
        feature_columns=feature_columns,
        label_columns=label_columns,
        output_metrics=output_metrics,
    )
    return report


def _predict_from_args(args: argparse.Namespace) -> dict:
    workload_ids = _selected_workloads(args.dataset_dir, args.workload_id)
    predictions_root = Path(args.predictions_dir)
    predictions_root.mkdir(parents=True, exist_ok=True)
    predictor = CpuMultiHeadPredictor(
        Path(args.model_dir) / "checkpoint.pt",
        num_threads=args.num_threads,
    )
    reports = tuple({
        "workload_id": workload_id,
        **predict_parquet(
            predictor=predictor,
            features_path=Path(args.dataset_dir) / f"{workload_id}.parquet",
            output_path=predictions_root / f"{workload_id}.parquet",
            batch_size=args.batch_size,
        ),
    } for workload_id in workload_ids)
    return {"predictions_dir": str(predictions_root), "workloads": reports}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="peregrine",
        description="Peregrine trace dataset workflows",
    )
    sub = parser.add_subparsers(dest="domain", required=True)

    dataset = sub.add_parser("dataset")
    dataset_sub = dataset.add_subparsers(dest="action", required=True)
    build = dataset_sub.add_parser("build")
    build.add_argument("--config", default="configs/peregrine.yaml")
    build.add_argument("--metrics-config", required=True)
    build.add_argument("--microarchitecture-config", required=True)
    build.add_argument("--raw-root", required=True)
    build.add_argument("--manifest", type=Path, default=None)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--workload-id", action="append", default=[])
    build.add_argument("--workers", type=int, default=24)
    build.set_defaults(func=_dataset_build)

    model = sub.add_parser("model")
    model_sub = model.add_subparsers(dest="action", required=True)
    train = model_sub.add_parser("train")
    train.add_argument("--config", default="configs/peregrine.yaml")
    train.add_argument("--metrics-config", required=True)
    train.add_argument("--microarchitecture-config", required=True)
    train.add_argument("--dataset-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--workload-id", action="append", default=[])
    train.set_defaults(func=_model_train)

    predict = model_sub.add_parser("predict")
    predict.add_argument("--dataset-dir", required=True)
    predict.add_argument("--model-dir", required=True)
    predict.add_argument("--predictions-dir", required=True)
    predict.add_argument("--workload-id", action="append", default=[])
    predict.set_defaults(func=_model_predict)

    predict.add_argument("--batch-size", type=int, default=4096)
    predict.add_argument("--num-threads", type=int)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
