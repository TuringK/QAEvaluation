#!/usr/bin/env python3
"""
CLI entrypoint for QA evaluation framework.

Usage:
    python run_eval.py --config configs/example.json
    python run_eval.py --config configs/example.json --output_dir ./results --run_name test_run
    python run_eval.py --config configs/example.json --max_examples 100
"""

import argparse
import sys
from pathlib import Path

from qa_eval.config import load_eval_config
from qa_eval.evaluator import run_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QA evaluation with abstention analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Override run name",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Limit number of examples per dataset (DEBUG)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size for generation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name or path",
    )

    return parser.parse_args()


def apply_overrides(eval_cfg, args: argparse.Namespace):
    if args.output_dir is not None:
        eval_cfg.output_dir = args.output_dir

    if args.run_name is not None:
        eval_cfg.run_name = args.run_name

    if args.batch_size is not None:
        eval_cfg.model.batch_size = args.batch_size

    if args.model is not None:
        eval_cfg.model.model_name_or_path = args.model

    if args.max_examples is not None:
        for ds_cfg in eval_cfg.datasets:
            ds_cfg.max_examples = args.max_examples

    return eval_cfg


def main() -> int:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1

    print(f"Loading config from: {args.config}")
    try:
        eval_cfg = load_eval_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    eval_cfg = apply_overrides(eval_cfg, args)

    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"Model: {eval_cfg.model.model_name_or_path}")
    print(f"Datasets: {[ds.name for ds in eval_cfg.datasets]}")
    print(f"Output: {eval_cfg.output_dir}/{eval_cfg.run_name}")
    print(f"Batch size: {eval_cfg.model.batch_size}")
    print("=" * 60 + "\n")

    try:
        results = run_eval(eval_cfg)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        return 130
    except Exception as e:
        print(f"\nError during evaluation: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)

    for dataset_name, metrics in results.items():
        if "error" in metrics:
            print(f"{dataset_name}: ERROR - {metrics['error']}")
        else:
            acc = metrics.get("accuracy", 0)
            n = metrics.get("n", 0)
            n_correct = metrics.get("n_correct", 0)
            print(f"{dataset_name}: {acc:.2%} ({n_correct}/{n})")

    print("=" * 60)

    valid_results = [m for m in results.values() if "accuracy" in m]
    if len(valid_results) > 1:
        avg_acc = sum(m["accuracy"] for m in valid_results) / len(valid_results)
        print(f"Average accuracy: {avg_acc:.2%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
