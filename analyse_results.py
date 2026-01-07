#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from qa_eval.analysis import generate_analysis_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse QA evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory containing results (e.g., outputs/Qwen2_5_1_5B_all)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for analysis results (default: <experiment_dir>/analysis)",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots (only create CSV files)",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    experiment_dir = Path(args.experiment_dir)

    if not experiment_dir.exists():
        print(
            f"Error: Experiment directory not found: {experiment_dir}", file=sys.stderr
        )
        return 1

    if not experiment_dir.is_dir():
        print(f"Error: {experiment_dir} is not a directory", file=sys.stderr)
        return 1

    # Check if directory contains results
    metrics_files = list(experiment_dir.glob("*/metrics.json"))
    if not metrics_files:
        print(f"Error: No results found in {experiment_dir}", file=sys.stderr)
        print(
            "Expected directory structure: <experiment_dir>/<dataset>_<split>/metrics.json",
            file=sys.stderr,
        )
        return 1

    print("=" * 70)
    print("QA Evaluation Results Analysis")
    print("=" * 70)
    print(f"Experiment: {experiment_dir.name}")
    print(f"Found {len(metrics_files)} dataset results")
    print("=" * 70 + "\n")

    try:
        output_dir = Path(args.output_dir) if args.output_dir else None
        results = generate_analysis_report(experiment_dir, output_dir=output_dir)

        print("\n" + "=" * 70)
        print("Analysis Summary")
        print("=" * 70)

        df_agg = results["aggregated"]
        print("\nAggregated Results:")
        print(df_agg.to_string(index=False))

        df_detailed = results["detailed"]

        df_subsets = df_detailed[df_detailed["subset"] != "N/A"]
        if not df_subsets.empty:
            print("\n\nSubset Results (sample):")
            print(df_subsets.head(10).to_string(index=False))
            if len(df_subsets) > 10:
                print(f"\n... and {len(df_subsets) - 10} more subsets")

        print("\n" + "=" * 70)
        print("Analysis complete")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        return 130
    except Exception as e:
        print(f"\nError during analysis: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
