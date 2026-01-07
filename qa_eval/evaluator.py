import json
import logging
from pathlib import Path
from typing import Any, Optional

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from qa_eval.config import DatasetConfig, EvalConfig
from qa_eval.datasets import (
    expand_dataset_configs,
    get_dataset_type,
    load_dataset_by_config,
)
from qa_eval.metrics import score_mc_examples, score_open_examples
from qa_eval.model_utils import (
    generate_batches,
    get_model_info,
    load_model_and_tokeniser,
)
from qa_eval.prompts import build_prompt, index_to_letter


logger = logging.getLogger(__name__)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _get_dataset_output_dir(eval_cfg: EvalConfig, ds_cfg: DatasetConfig) -> Path:
    base = Path(eval_cfg.output_dir) / eval_cfg.run_name

    # Dataset identifier
    ds_id = ds_cfg.name
    if ds_cfg.subset:
        ds_id = f"{ds_id}_{ds_cfg.subset}"
    ds_id = f"{ds_id}_{ds_cfg.split}"

    return base / ds_id


def _write_predictions(
    output_dir: Path,
    examples: list[dict],
    predictions: list[str],
    results: dict,
    dataset_type: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"

    per_example = results.get("per_example", [])

    with open(predictions_path, "w") as f:
        for i, (ex, pred, res) in enumerate(zip(examples, predictions, per_example)):
            record = {
                "id": ex.get("id", i),
                "question": ex["question"],
                "prediction_raw": pred,
                "is_correct": res.get("correct", False),
            }

            if dataset_type == "mc":
                record["options"] = ex.get("options", [])
                record["gold_idx"] = ex.get("correct_option_idx")
                record["gold_letter"] = index_to_letter(ex["correct_option_idx"])
                record["predicted_letter"] = res.get("extracted_letter")
                record["predicted_idx"] = res.get("predicted_idx")
                record["extraction_method"] = res.get("extraction_method")
            else:
                record["gold"] = ex.get("answer")
                record["match_type"] = res.get("match_type")

            f.write(json.dumps(record) + "\n")

    logger.info(f"Wrote predictions to {predictions_path}")


def _write_metrics(output_dir: Path, metrics: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_to_save = {k: v for k, v in metrics.items() if k != "per_example"}

    with open(metrics_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    logger.info(f"Wrote metrics to {metrics_path}")


def _aggregate_subset_results(
    results: dict[str, dict],
    dataset_name: str,
) -> dict[str, Any]:
    subset_results = [
        (key, metrics)
        for key, metrics in results.items()
        if key.startswith(f"{dataset_name}_") and "error" not in metrics
    ]

    if not subset_results:
        return {}

    # Aggregate metrics
    total_correct = sum(metrics["n_correct"] for _, metrics in subset_results)
    total_examples = sum(metrics["n"] for _, metrics in subset_results)

    aggregated = {
        "dataset": dataset_name,
        "subset": "all",
        "n_subsets": len(subset_results),
        "subsets": [key.replace(f"{dataset_name}_", "") for key, _ in subset_results],
        "n_examples": total_examples,
        "n_correct": total_correct,
        "accuracy": total_correct / total_examples if total_examples > 0 else 0.0,
        "per_subset_accuracy": {
            key.replace(f"{dataset_name}_", ""): metrics["accuracy"]
            for key, metrics in subset_results
        },
    }

    return aggregated


def run_single_dataset_eval(
    eval_cfg: EvalConfig,
    ds_cfg: DatasetConfig,
    model: Optional[PreTrainedModel] = None,
    tokeniser: Optional[PreTrainedTokenizerBase] = None,
) -> dict[str, Any]:
    output_dir = _get_dataset_output_dir(eval_cfg, ds_cfg)
    dataset_type = get_dataset_type(ds_cfg)

    logger.info(
        f"Starting evaluation: {ds_cfg.name} ({ds_cfg.subset or 'default'}) - {ds_cfg.split}"
    )

    if model is None or tokeniser is None:
        logger.info(f"Loading model: {eval_cfg.model.model_name_or_path}")
        model, tokeniser = load_model_and_tokeniser(eval_cfg.model)
        logger.info("Model loaded successfully")

    logger.info("Loading dataset...")
    dataset = load_dataset_by_config(ds_cfg)
    examples = list(dataset)
    logger.info(f"Loaded {len(examples)} examples")

    logger.info("Building prompts...")
    prompts = [build_prompt(ex, eval_cfg.prompt, ds_cfg.name) for ex in examples]

    logger.info(f"Generating predictions (batch_size={eval_cfg.model.batch_size})...")
    predictions = generate_batches(
        model=model,
        tokeniser=tokeniser,
        prompts=prompts,
        model_cfg=eval_cfg.model,
        show_progress=True,
    )

    logger.info("Scoring predictions...")
    if dataset_type == "mc":
        results = score_mc_examples(examples, predictions)
    else:
        results = score_open_examples(examples, predictions)

    logger.info(
        f"Results: accuracy={results['accuracy']:.4f} "
        f"({results['n_correct']}/{results['n']})"
    )

    _write_predictions(output_dir, examples, predictions, results, dataset_type)

    metrics = {
        "dataset": ds_cfg.name,
        "subset": ds_cfg.subset,
        "split": ds_cfg.split,
        "n_examples": len(examples),
        "model": eval_cfg.model.model_name_or_path,
        **{k: v for k, v in results.items() if k != "per_example"},
    }
    _write_metrics(output_dir, metrics)

    return metrics


def run_eval(eval_cfg: EvalConfig) -> dict[str, Any]:
    _setup_logging(level="INFO")

    # Expand multi subset datasets
    original_datasets = eval_cfg.datasets
    expanded_datasets = expand_dataset_configs(eval_cfg.datasets)

    # Identify which datasets were expanded for later aggregation
    has_mmlu_expansion = any(
        ds.name == "mmlu_redux" and ds.subset is None for ds in original_datasets
    )
    has_musr_expansion = any(
        ds.name == "musr" and ds.subset is None for ds in original_datasets
    )

    logger.info("=" * 60)
    logger.info(f"Starting evaluation run: {eval_cfg.run_name}")
    logger.info(f"Output directory: {eval_cfg.output_dir}")
    logger.info(f"Original dataset configs: {len(original_datasets)}")
    logger.info(f"Expanded to {len(expanded_datasets)} evaluations")
    if has_mmlu_expansion:
        logger.info("  - MMLU-Redux: evaluating all subsets")
    if has_musr_expansion:
        logger.info("  - MuSR: evaluating all subsets")
    logger.info("=" * 60)

    logger.info(f"Loading model: {eval_cfg.model.model_name_or_path}")
    model, tokeniser = load_model_and_tokeniser(eval_cfg.model)

    model_info = get_model_info(model, tokeniser)
    logger.info(f"Model loaded: {model_info['model_class']}")
    logger.info(f"  Device: {model_info['device']}, Dtype: {model_info['dtype']}")
    logger.info(f"  Parameters: {model_info['total_parameters']:,}")

    all_results = {}

    for i, ds_cfg in enumerate(expanded_datasets, 1):
        logger.info("-" * 60)
        logger.info(f"Dataset {i}/{len(expanded_datasets)}: {ds_cfg.name}")
        if ds_cfg.subset:
            logger.info(f"  Subset: {ds_cfg.subset}")

        try:
            metrics = run_single_dataset_eval(
                eval_cfg=eval_cfg,
                ds_cfg=ds_cfg,
                model=model,
                tokeniser=tokeniser,
            )

            # Build result key
            result_key = ds_cfg.name
            if ds_cfg.subset:
                result_key = f"{result_key}_{ds_cfg.subset}"

            all_results[result_key] = metrics

        except Exception as e:
            logger.error(f"Failed to evaluate {ds_cfg.name}: {e}")
            result_key = ds_cfg.name
            if ds_cfg.subset:
                result_key = f"{result_key}_{ds_cfg.subset}"
            all_results[result_key] = {"error": str(e)}

    # Aggregate multi subset results if they were expanded
    if has_mmlu_expansion:
        logger.info("-" * 60)
        logger.info("Aggregating MMLU-Redux results across all subsets...")
        aggregated = _aggregate_subset_results(all_results, "mmlu_redux")
        if aggregated:
            all_results["mmlu_redux_combined"] = aggregated
            logger.info(
                f"  MMLU-Redux combined: {aggregated['accuracy']:.4f} "
                f"({aggregated['n_correct']}/{aggregated['n_examples']}) "
                f"across {aggregated['n_subsets']} subsets"
            )

    if has_musr_expansion:
        logger.info("-" * 60)
        logger.info("Aggregating MuSR results across all subsets...")
        aggregated = _aggregate_subset_results(all_results, "musr")
        if aggregated:
            all_results["musr_combined"] = aggregated
            logger.info(
                f"  MuSR combined: {aggregated['accuracy']:.4f} "
                f"({aggregated['n_correct']}/{aggregated['n_examples']}) "
                f"across {aggregated['n_subsets']} subsets"
            )

    logger.info("=" * 60)
    logger.info("Evaluation complete. Summary:")

    for name, metrics in all_results.items():
        if "error" in metrics:
            logger.info(f"  {name}: ERROR - {metrics['error']}")
        else:
            logger.info(f"  {name}: accuracy={metrics['accuracy']:.4f}")

    summary_path = Path(eval_cfg.output_dir) / eval_cfg.run_name / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    return all_results
