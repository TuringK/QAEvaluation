import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import numpy as np


def load_predictions_file(predictions_path: Path) -> List[dict]:
    predictions = []
    with open(predictions_path, "r") as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def compute_abstention_metrics(predictions: List[dict]) -> Dict[str, float]:
    """
    - n_total: Total samples
    - n_answered: Questions attempted (may be < n_total with abstention)
    - n_abstained: Questions skipped
    - n_correct: Correct predictions
    - n_wrong: Wrong predictions
    - coverage: % of questions answered (n_answered / n_total)
    - accuracy_all: Accuracy on all samples (standard metric)
    - accuracy_answered: Accuracy only on answered questions (selective accuracy)
    - selective_risk: % wrong among answered (error rate on attempted questions)
    """
    n_total = len(predictions)

    # Just checking for no answer or failed retrieval for now
    # TODO: Implement same abstention retrieval as in AbstentionBench?
    n_abstained = 0
    n_correct = 0
    n_answered = 0

    for p in predictions:
        # A prediction is treated as abstained if any of the following holds:
        # - The pipeline explicitly flags it as abstained.
        # - The answer extraction step failed.
        # - No prediction was extracted and the match type indicates no match.
        is_abstained = (
            p.get("abstained", False)
            or p.get("extraction_method") == "failed"
            or (p.get("predicted_idx") is None and p.get("match_type") == "none")
        )

        if is_abstained:
            n_abstained += 1
        else:
            n_answered += 1
            if p.get("is_correct", p.get("correct", False)):
                n_correct += 1

    n_wrong = n_answered - n_correct

    coverage = n_answered / n_total if n_total > 0 else 0.0
    accuracy_all = n_correct / n_total if n_total > 0 else 0.0
    accuracy_answered = n_correct / n_answered if n_answered > 0 else 0.0
    selective_risk = n_wrong / n_answered if n_answered > 0 else 0.0

    return {
        "n_total": n_total,
        "n_answered": n_answered,
        "n_abstained": n_abstained,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "coverage": coverage,
        "accuracy_all": accuracy_all,
        "accuracy_answered": accuracy_answered,
        "selective_risk": selective_risk,
    }


def load_experiment_results(experiment_dir: Path) -> Dict[str, dict]:
    experiment_dir = Path(experiment_dir)
    results = {}

    for metrics_path in experiment_dir.glob("*/metrics.json"):
        dataset_dir = metrics_path.parent
        dataset_split_name = dataset_dir.name

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        predictions_path = dataset_dir / "predictions.jsonl"
        predictions = []
        if predictions_path.exists():
            predictions = load_predictions_file(predictions_path)

        if predictions:
            abstention_metrics = compute_abstention_metrics(predictions)
            metrics.update(abstention_metrics)

        results[dataset_split_name] = {"metrics": metrics, "predictions": predictions}

    return results


def parse_dataset_name(dataset_split_name: str) -> Tuple[str, Optional[str], str]:
    parts = dataset_split_name.split("_")

    known_splits = {"test", "validation", "train", "dev"}

    split = None
    split_idx = len(parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in known_splits:
            split = parts[i]
            split_idx = i
            break

    if split is None:
        split = parts[-1]
        split_idx = len(parts) - 1

    remaining = parts[:split_idx]

    if not remaining:
        return (dataset_split_name, None, split)

    # Check for multi-word datasets
    if remaining[0] in ["mmlu", "mmlu_redux"] and len(remaining) > 1:
        if remaining[0] == "mmlu" and len(remaining) > 1 and remaining[1] == "redux":
            dataset = "mmlu_redux"
            subset = "_".join(remaining[2:]) if len(remaining) > 2 else None
        else:
            dataset = remaining[0]
            subset = "_".join(remaining[1:]) if len(remaining) > 1 else None
    elif remaining[0] == "musr" and len(remaining) > 1:
        dataset = "musr"
        subset = "_".join(remaining[1:])
    else:
        dataset = remaining[0]
        subset = "_".join(remaining[1:]) if len(remaining) > 1 else None

    return (dataset, subset, split)


def create_detailed_dataframe(results: Dict[str, dict]) -> pd.DataFrame:
    """
    - dataset, subset, split: Identifiers
    - n_total: Total samples
    - n_answered, n_abstained: Coverage breakdown
    - coverage: % answered
    - accuracy_all: Standard accuracy
    - accuracy_answered: Accuracy on answered questions only
    - n_correct, n_wrong: Response breakdown
    - selective_risk: Error rate on answered
    """
    rows = []

    for dataset_split_name, data in results.items():
        metrics = data["metrics"]
        dataset, subset, split = parse_dataset_name(dataset_split_name)

        row = {
            "dataset": dataset,
            "subset": subset if subset else "N/A",
            "split": split,
            "n_total": metrics.get(
                "n_total", metrics.get("n", metrics.get("n_samples", 0))
            ),
            "n_answered": metrics.get("n_answered", metrics.get("n", 0)),
            "n_abstained": metrics.get("n_abstained", 0),
            "coverage": metrics.get("coverage", 1.0),
            "accuracy_all": metrics.get("accuracy_all", metrics.get("accuracy", 0.0)),
            "accuracy_answered": metrics.get(
                "accuracy_answered", metrics.get("accuracy", 0.0)
            ),
            "n_correct": metrics.get("n_correct", 0),
            "n_wrong": metrics.get("n_wrong", 0),
            "selective_risk": metrics.get("selective_risk", 0.0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    df = df.sort_values(["dataset", "subset"]).reset_index(drop=True)

    return df


def create_aggregated_dataframe(results: Dict[str, dict]) -> pd.DataFrame:
    grouped = {}

    for dataset_split_name, data in results.items():
        metrics = data["metrics"]
        dataset, subset, split = parse_dataset_name(dataset_split_name)

        key = (dataset, split)

        if key not in grouped:
            grouped[key] = {"predictions": [], "n_samples": 0, "n_correct": 0}

        grouped[key]["predictions"].extend(data["predictions"])
        grouped[key]["n_samples"] += metrics.get("n", metrics.get("n_samples", 0))
        grouped[key]["n_correct"] += metrics.get("n_correct", 0)

    rows = []

    for (dataset, split), data in grouped.items():
        if data["predictions"]:
            abstention_metrics = compute_abstention_metrics(data["predictions"])
        else:
            # Fallback for missing predictions
            n_samples = data["n_samples"]
            n_correct = data["n_correct"]
            accuracy = n_correct / n_samples if n_samples > 0 else 0.0
            abstention_metrics = {
                "n_total": n_samples,
                "n_answered": n_samples,
                "n_abstained": 0,
                "n_correct": n_correct,
                "n_wrong": n_samples - n_correct,
                "coverage": 1.0,
                "accuracy_all": accuracy,
                "accuracy_answered": accuracy,
                "selective_risk": 1.0 - accuracy,
            }

        row = {
            "dataset": dataset,
            "subset": "aggregated",
            "split": split,
            "n_total": abstention_metrics["n_total"],
            "n_answered": abstention_metrics["n_answered"],
            "n_abstained": abstention_metrics["n_abstained"],
            "coverage": abstention_metrics["coverage"],
            "accuracy_all": abstention_metrics["accuracy_all"],
            "accuracy_answered": abstention_metrics["accuracy_answered"],
            "n_correct": abstention_metrics["n_correct"],
            "n_wrong": abstention_metrics["n_wrong"],
            "selective_risk": abstention_metrics["selective_risk"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["dataset", "split"]).reset_index(drop=True)

    return df


def plot_aggregated_performance(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10

    metrics_to_plot = [
        "coverage",
        "accuracy_all",
        "accuracy_answered",
        "selective_risk",
    ]

    df["label"] = df["dataset"] + "_" + df["split"]

    plot_data = df.melt(
        id_vars=["label"],
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score",
    )

    metric_names = {
        "coverage": "Coverage",
        "accuracy_all": "Accuracy (All)",
        "accuracy_answered": "Accuracy (Answered)",
        "selective_risk": "Selective Risk",
    }
    plot_data["Metric"] = plot_data["Metric"].map(metric_names)

    fig, ax = plt.subplots(figsize=figsize)

    x_labels = df["label"].unique()
    x = np.arange(len(x_labels))
    width = 0.2

    colors = ["#3498db", "#f39c12", "#2ecc71", "#e74c3c"]

    for i, (metric, color) in enumerate(zip(metric_names.values(), colors)):
        metric_data = plot_data[plot_data["Metric"] == metric]
        values = [
            (
                metric_data[metric_data["label"] == label]["Score"].values[0]
                if len(metric_data[metric_data["label"] == label]) > 0
                else 0
            )
            for label in x_labels
        ]

        ax.bar(x + i * width, values, width, label=metric, color=color, alpha=0.8)

    ax.set_xlabel("Dataset", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Performance Comparison Across Datasets", fontweight="bold", pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_subset_performance(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    dataset_filter: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    max_subsets: int = 50,
) -> plt.Figure:
    df_subsets = df[df["subset"] != "N/A"].copy()

    if dataset_filter:
        df_subsets = df_subsets[df_subsets["dataset"] == dataset_filter].copy()
        if df_subsets.empty:
            print(f"No subset data available for dataset: {dataset_filter}")
            return None

    if df_subsets.empty:
        print("No subset data available for plotting.")
        return None

    if len(df_subsets) > max_subsets:
        print(
            f"Warning: {len(df_subsets)} subsets found. Showing top {max_subsets} by accuracy_answered."
        )
        df_subsets = df_subsets.nlargest(max_subsets, "accuracy_answered")

    n_subsets = len(df_subsets)

    # Determine grid layout based on number of subsets
    # This can be played around with for better readability
    if n_subsets <= 6:
        nrows, ncols = 1, n_subsets
        if figsize is None:
            figsize = (4 * n_subsets, 6)
    elif n_subsets <= 10:
        nrows, ncols = 2, (n_subsets + 1) // 2
        if figsize is None:
            figsize = (4 * ncols, 10)
    else:
        nrows = 3
        ncols = (n_subsets + nrows - 1) // nrows
        if figsize is None:
            figsize = (4 * ncols, 12)

    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9

    metrics_to_plot = [
        "coverage",
        "accuracy_all",
        "accuracy_answered",
        "selective_risk",
    ]

    if dataset_filter:
        df_subsets["label"] = df_subsets["subset"]
        main_title = f"Performance Across {dataset_filter.upper()} Subsets"
    else:
        df_subsets["label"] = df_subsets["dataset"] + ": " + df_subsets["subset"]
        main_title = "Performance Across Dataset Subsets"

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    colors = ["#3498db", "#f39c12", "#2ecc71", "#e74c3c"]
    metric_names = {
        "coverage": "Coverage",
        "accuracy_all": "Accuracy (All)",
        "accuracy_answered": "Accuracy (Ans)",
        "selective_risk": "Sel. Risk",
    }

    subsets_list = df_subsets["label"].tolist()

    for idx, subset_label in enumerate(subsets_list):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        subset_data = df_subsets[df_subsets["label"] == subset_label].iloc[0]

        values = [
            subset_data["coverage"],
            subset_data["accuracy_all"],
            subset_data["accuracy_answered"],
            subset_data["selective_risk"],
        ]

        x = np.arange(len(metrics_to_plot))
        bars = ax.bar(x, values, color=colors, alpha=0.8, width=0.6)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ax.set_title(subset_label, fontweight="bold", fontsize=10, pad=8)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [metric_names[m] for m in metrics_to_plot],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax.grid(axis="y", alpha=0.3)

        if col == 0:
            ax.set_ylabel("Score", fontweight="bold", fontsize=9)

    for idx in range(n_subsets, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis("off")

    fig.suptitle(main_title, fontweight="bold", fontsize=14, y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_coverage_accuracy_tradeoff(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14

    fig, ax = plt.subplots(figsize=figsize)

    df["label"] = df["dataset"] + "_" + df["split"]

    for idx, row in df.iterrows():
        ax.scatter(
            row["coverage"],
            row["accuracy_answered"],
            s=200,
            alpha=0.7,
            edgecolors="black",
            linewidths=1.5,
        )

        ax.annotate(
            row["label"],
            xy=(row["coverage"], row["accuracy_answered"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Coverage (% Questions Answered)", fontweight="bold", fontsize=13)
    ax.set_ylabel("Accuracy on Answered Questions", fontweight="bold", fontsize=13)
    ax.set_title(
        "Coverage vs Accuracy Trade-off\n(Comparing Selective Prediction Performance)",
        fontweight="bold",
        pad=20,
        fontsize=14,
    )

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="Random baseline")

    ax.grid(True, alpha=0.3)

    legend_elements = [
        Rectangle((0, 0), 1, 1, fc="blue", alpha=0.7, label="Dataset performance"),
        Rectangle(
            (0, 0), 1, 1, fc="none", ec="black", linestyle="--", label="Random baseline"
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def generate_analysis_report(
    experiment_dir: Path, output_dir: Optional[Path] = None
) -> Dict[str, pd.DataFrame]:
    experiment_dir = Path(experiment_dir)

    if output_dir is None:
        output_dir = experiment_dir / "analysis"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {experiment_dir}")
    results = load_experiment_results(experiment_dir)
    print(f"Found {len(results)} dataset results")

    df_aggregated = create_aggregated_dataframe(results)
    df_detailed = create_detailed_dataframe(results)
    aggregated_csv = output_dir / "aggregated_results.csv"
    detailed_csv = output_dir / "detailed_results.csv"

    df_aggregated.to_csv(aggregated_csv, index=False)
    print(f"Saved aggregated results to: {aggregated_csv}")

    df_detailed.to_csv(detailed_csv, index=False)
    print(f"Saved detailed results to: {detailed_csv}")

    agg_plot_path = output_dir / "performance_aggregated.png"
    plot_aggregated_performance(df_aggregated, output_path=agg_plot_path)
    print(f"Saved aggregated performance plot to: {agg_plot_path}")
    plt.close()

    tradeoff_plot_path = output_dir / "coverage_accuracy_tradeoff.png"
    plot_coverage_accuracy_tradeoff(df_aggregated, output_path=tradeoff_plot_path)
    print(f"Saved coverage-accuracy trade-off plot to: {tradeoff_plot_path}")
    plt.close()

    df_with_subsets = df_detailed[df_detailed["subset"] != "N/A"]
    if not df_with_subsets.empty:
        # Only create plots for MMLU and MUSR
        target_datasets = ["mmlu_redux", "musr"]
        available_datasets = df_with_subsets["dataset"].unique()

        for dataset in target_datasets:
            if dataset in available_datasets:
                plot_filename = f"performance_subsets_{dataset}.png"
                subset_plot_path = output_dir / plot_filename

                fig = plot_subset_performance(
                    df_detailed, output_path=subset_plot_path, dataset_filter=dataset
                )

                if fig is not None:
                    print(
                        f"Saved {dataset} subset performance plot to: {subset_plot_path}"
                    )
                    plt.close()

    print(f"\nAnalysis complete. All results saved to: {output_dir}")

    return {"aggregated": df_aggregated, "detailed": df_detailed}
