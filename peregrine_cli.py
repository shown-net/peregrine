#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from anamol.python.design_space import load_peregrine_config
from anamol.python.microarchitecture import load_microarchitecture_config
from anamol.python.dataset import build_full_roi_window_dataset_shards
from ml_model.inference import PredictorBundle
from ml_model.inference import predict_bundle_parquet
from ml_model.l2 import search_l2
from ml_model.plots import plot_l1_summary
from ml_model.prediction import train_prediction_task
from ml_model.prediction import evaluate_prediction_task
from ml_model.prediction import evaluate_random_roi_split_prediction_task
from ml_model.real_anchor import build_real_anchor_dataset
from ml_model.tasks import L1_TASK_ID
from ml_model.tasks import L3_TASK_ID
from ml_model.tasks import l1_task
from ml_model.tasks import l3_task


DEFAULT_OUTPUT_ROOT = Path("output/no_cache_hnf_gem5_surrogate")
DEFAULT_DATASET_DIR = DEFAULT_OUTPUT_ROOT / "dataset"
DEFAULT_MODEL_DIR = DEFAULT_OUTPUT_ROOT / "model"
DEFAULT_PREDICTIONS_DIR = DEFAULT_OUTPUT_ROOT / "predictions"
DEFAULT_EVALUATION_DIR = DEFAULT_OUTPUT_ROOT / "evaluation"


def _print_json(payload: dict) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _dataset_build(args: argparse.Namespace) -> int:
    if args.task == L3_TASK_ID:
        return _print_json(build_real_anchor_dataset(
            raw_root=args.raw_root, metrics_config=args.metrics_config,
            output_dir=args.output_dir, workload_ids=tuple(args.workload_id or ()) or None,
        ))
    if args.manifest is not None:
        raise ValueError("l1-surrogate full-ROI dataset build does not accept --manifest")
    config = load_peregrine_config(args.config, metrics_config=args.metrics_config, microarchitecture=load_microarchitecture_config(args.microarchitecture_config))
    report = build_full_roi_window_dataset_shards(
        config=config,
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        workload_ids=tuple(args.workload_id or ()) or None,
        workers=args.workers,
    )
    return _print_json(report)


def _model_train(args: argparse.Namespace) -> int:
    return _print_json(train_prediction_task(
        task=_task_from_args(args), dataset_dir=args.dataset_dir, output_dir=args.output_dir,
        workload_ids=tuple(args.workload_id or ()) or None,
    ))


def _model_predict(args: argparse.Namespace) -> int:
    return _print_json(_predict_from_args(args))


def _model_evaluate(args: argparse.Namespace) -> int:
    task = _task_from_args(args)
    valid_protocol = (
        (task.task_id == L1_TASK_ID and args.protocol == "config-generalization")
        or (task.task_id == L3_TASK_ID and args.protocol == "workload-ood")
    )
    if args.protocol == "random-roi" and task.task_id == L1_TASK_ID:
        evaluate = evaluate_random_roi_split_prediction_task
    elif valid_protocol:
        evaluate = evaluate_prediction_task
    else:
        raise ValueError(f"protocol {args.protocol} does not apply to task {task.task_id}")
    return _print_json(evaluate(
        task=task, dataset_dir=args.dataset_dir, output_dir=args.output_dir,
        workload_ids=tuple(args.workload_id or ()) or None,
    ))


def _plot_l1_summary(args: argparse.Namespace) -> int:
    return _print_json(plot_l1_summary(
        dataset_dir=args.dataset_dir,
        config_generalization_dir=args.config_generalization_dir,
        random_roi_dir=args.random_roi_dir,
        output_dir=args.output_dir,
    ))


def _l2_search(args: argparse.Namespace) -> int:
    config = load_peregrine_config(
        args.config, metrics_config=args.metrics_config,
        microarchitecture=load_microarchitecture_config(args.microarchitecture_config),
    )
    return _print_json(search_l2(
        config=config, bundle_path=args.bundle, raw_root=args.raw_root,
        dataset_dir=args.dataset_dir, output_dir=args.output_dir,
        exploration_count=args.exploration_count, queue_size=args.queue_size,
        queue_batch_size=args.queue_batch_size,
        analysis_batch_size=args.analysis_batch_size,
        inference_batch_size=args.inference_batch_size, seed=args.seed,
    ))


def _selected_workloads(dataset_dir: str | Path, workload_ids: list[str]) -> tuple[str, ...]:
    return tuple(workload_ids or ()) or tuple(
        path.stem for path in sorted(Path(dataset_dir).glob("*.parquet"))
    )


def _task_from_args(args: argparse.Namespace):
    if args.task == L3_TASK_ID:
        return l3_task()
    if args.task == L1_TASK_ID:
        if not args.microarchitecture_config:
            raise ValueError("l1-surrogate requires --microarchitecture-config")
        config = load_peregrine_config(
            args.config, metrics_config=args.metrics_config,
            microarchitecture=load_microarchitecture_config(args.microarchitecture_config),
        )
        return l1_task(config)
    raise ValueError(f"unknown prediction task: {args.task}")


def _predict_from_args(args: argparse.Namespace) -> dict:
    workload_ids = _selected_workloads(args.dataset_dir, args.workload_id)
    predictions_root = Path(args.predictions_dir)
    predictions_root.mkdir(parents=True, exist_ok=True)
    predictor = PredictorBundle(Path(args.model_dir) / "predictor_bundle.json", num_threads=args.num_threads)
    reports = tuple({
        "workload_id": workload_id,
        **predict_bundle_parquet(
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
    build.add_argument("--task", choices=(L1_TASK_ID, L3_TASK_ID), required=True)
    build.add_argument("--config", default="configs/peregrine.yaml")
    build.add_argument("--metrics-config", required=True)
    build.add_argument("--microarchitecture-config")
    build.add_argument("--raw-root", required=True)
    build.add_argument("--manifest", type=Path, default=None)
    build.add_argument("--output-dir", default=str(DEFAULT_DATASET_DIR))
    build.add_argument("--workload-id", action="append", default=[])
    build.add_argument("--workers", type=int, default=24)
    build.set_defaults(func=_dataset_build)

    model = sub.add_parser("model")
    model_sub = model.add_subparsers(dest="action", required=True)
    train = model_sub.add_parser("train")
    train.add_argument("--task", choices=(L1_TASK_ID, L3_TASK_ID), required=True)
    train.add_argument("--config", default="configs/peregrine.yaml")
    train.add_argument("--metrics-config", required=True)
    train.add_argument("--microarchitecture-config")
    train.add_argument("--dataset-dir", required=True)
    train.add_argument("--output-dir", default=str(DEFAULT_MODEL_DIR))
    train.add_argument("--workload-id", action="append", default=[])
    train.set_defaults(func=_model_train)

    predict = model_sub.add_parser("predict")
    predict.add_argument("--dataset-dir", required=True)
    predict.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    predict.add_argument("--predictions-dir", default=str(DEFAULT_PREDICTIONS_DIR))
    predict.add_argument("--workload-id", action="append", default=[])
    predict.set_defaults(func=_model_predict)

    predict.add_argument("--batch-size", type=int, default=4096)
    predict.add_argument("--num-threads", type=int)

    evaluate = model_sub.add_parser("evaluate")
    evaluate.add_argument("--task", choices=(L1_TASK_ID, L3_TASK_ID), required=True)
    evaluate.add_argument("--config", default="configs/peregrine.yaml")
    evaluate.add_argument("--metrics-config", required=True)
    evaluate.add_argument("--microarchitecture-config")
    evaluate.add_argument("--dataset-dir", required=True)
    evaluate.add_argument("--output-dir", default=str(DEFAULT_EVALUATION_DIR))
    evaluate.add_argument("--workload-id", action="append", default=[])
    evaluate.add_argument(
        "--protocol",
        choices=("config-generalization", "workload-ood", "random-roi"),
        required=True,
    )
    evaluate.set_defaults(func=_model_evaluate)

    plot = sub.add_parser("plot")
    plot_sub = plot.add_subparsers(dest="action", required=True)
    l1_summary = plot_sub.add_parser("l1-summary")
    l1_summary.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    l1_summary.add_argument("--config-generalization-dir", default=str(DEFAULT_EVALUATION_DIR))
    l1_summary.add_argument("--random-roi-dir", default=str(DEFAULT_EVALUATION_DIR))
    l1_summary.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "plots_l1"))
    l1_summary.set_defaults(func=_plot_l1_summary)

    l2 = sub.add_parser("l2")
    l2_sub = l2.add_subparsers(dest="action", required=True)
    search = l2_sub.add_parser("search")
    search.add_argument("--config", default="configs/peregrine.yaml")
    search.add_argument("--metrics-config", required=True)
    search.add_argument("--microarchitecture-config", required=True)
    search.add_argument("--bundle", required=True)
    search.add_argument("--raw-root", required=True)
    search.add_argument("--dataset-dir", required=True)
    search.add_argument("--output-dir", required=True)
    search.add_argument("--exploration-count", type=int, required=True)
    search.add_argument("--queue-size", type=int, required=True)
    search.add_argument("--queue-batch-size", type=int, required=True)
    search.add_argument("--analysis-batch-size", type=int, default=16)
    search.add_argument("--inference-batch-size", type=int, default=4096)
    search.add_argument("--seed", type=int, required=True)
    search.set_defaults(func=_l2_search)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
