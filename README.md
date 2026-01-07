> [!WARNING]
> TODOs:
> 1. Improve prediction answer detection (after analysing some outputs manually, the current system doesn't seem to handle complex answers well)
> 2. Implement AbstentionBench's keyword abstention detection to compare results accurately


A QA evaluation framework for testing language models on multiple question-answering datasets.

## Overview

Currently seven benchmarks are included:
- **CSQA2** - Commonsense QA 2
- **StrategyQA** - Strategy-based reasoning
- **MMLU Redux** - Massive Multitask Language Understanding
- **GPQA** - Graduate-level Professional QA
- **GSM8K** - Grade School Math
- **AGIEval** - AGI evaluation benchmarks
- **MuSR** - Multi-step reasoning

Supports both multiple-choice and open-ended question formats with configurable prompting strategies.

> [!TIP]
> You should be able to use the same venv as AbstentionBench, it worked for me with no problem


### Basic Evaluation

Run evaluation using a configuration file:

```bash
python run_eval.py --config configs/qwen2_5_1_5B_all.json
```

### Command-Line Options

CLI arguments will override config arguments

```bash
python run_eval.py \
  --config configs/example.json \
  --output_dir ./results \
  --run_name my_experiment \
  --max_examples 100 \
  --batch_size 8 \
  --model "Qwen/Qwen2.5-1.5B-Instruct"
```

**Available Arguments:**
- `--config`: Path to JSON configuration file (required)
- `--output_dir`: Override output directory (default: `./outputs`)
- `--run_name`: Name for this evaluation run (default: `eval_run`)
- `--max_examples`: Limit number of examples per dataset (for debugging)
- `--batch_size`: Override batch size for generation
- `--model`: Override model name or path

## Configuration

Create a JSON configuration file (see `configs/qwen2_5_1_5B_all.json` for example):

```json
{
  "model": {
    "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
    "dtype": "auto",
    "device": "auto",
    "max_new_tokens": 256,
    "temperature": 0.6,
    "top_p": 0.95,
    "do_sample": true,
    "batch_size": 8
  },
  "datasets": [
    {
      "name": "csqa2",
      "split": "validation",
      "max_examples": null,
      "shuffle_seed": 42
    },
    {
      "name": "mmlu_redux",
      "split": "test",
      "subset": "college_computer_science",
      "max_examples": 100
    }
  ],
  "output_dir": "./outputs",
  "run_name": "my_eval"
}
```

### Model Configuration

- `model_name_or_path`: HuggingFace model ID or local path
- `dtype`: `"auto"`, `"float16"`, `"bfloat16"`, or `"float32"`
- `device`: `"auto"`, `"cuda"`, `"cuda:0"`, or `"cpu"`
- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature (0.0 for greedy)
- `top_p`, `top_k`: Nucleus and top-k sampling parameters
- `do_sample`: Enable/disable sampling
- `batch_size`: Batch size for generation

### Dataset Configuration

- `name`: Dataset identifier (see supported datasets above)
- `split`: Dataset split (`"train"`, `"validation"`, `"test"`)
- `subset`: Dataset subset/config (e.g., MMLU subject) (**NOTE**: For MMLU and MUSR if the subset is set to `null` then all dataset subsets will be evaluated)
- `max_examples`: Limit number of examples (null = all)
- `shuffle_seed`: Seed for shuffling before truncation

## Output

Results are saved to `{output_dir}/{run_name}/`:

```
outputs/
└── my_eval/
    ├── csqa2_validation/
    │   ├── predictions.jsonl
    │   └── metrics.json
    ├── mmlu_redux_college_computer_science_test/
    │   ├── predictions.jsonl
    │   └── metrics.json
    └── ...
```

### Metrics

Each dataset evaluation produces:
- **accuracy**: Overall accuracy
- **n**: Total number of examples
- **n_correct**: Number of correct predictions
- Per-example results with extraction details

### Predictions Format

`predictions.jsonl` contains one JSON object per line:

```json
{
  "id": 0,
  "question": "What is...?",
  "prediction_raw": "The answer is B",
  "is_correct": true,
  "options": ["A", "B", "C", "D"],
  "gold_letter": "B",
  "predicted_letter": "B",
  "extraction_method": "regex"
}
```

## Example Workflow

1. **Create a configuration file:**
   ```bash
   cp configs/qwen2_5_1_5B_all.json configs/my_config.json
   # Edit my_config.json with your settings
   ```

2. **Run evaluation:**
   ```bash
   python run_eval.py --config configs/my_config.json
   ```

3. **View results:**
   ```bash
   cat outputs/eval_run/*/metrics.json
   ```

## Analysis & Reporting

### Automatic Analysis

Analysis runs automatically after evaluation completes. To disable:

```bash
python run_eval.py --config configs/my_config.json --no-analysis
```

### Standalone Analysis Script

You can reanalyse existing results without re-running evaluation:

```bash
python analyse_results.py outputs/my_eval
```

This generates analysis reports in `outputs/my_eval/analysis/`.

### Metrics Tracked

The analysis extracts and computes the following metrics for each dataset:

**Coverage Metrics:**
- `n_total`: Total number of samples
- `n_answered`: Questions attempted
- `n_abstained`: Questions skipped (Currently only a basic abstention detector is implemented, would be nice to implement the keyword one from AbstentionBench)
- `coverage`: Percentage of questions answered (n_answered / n_total)

**Accuracy Metrics:**
- `accuracy_all`: Standard accuracy on all samples (n_correct / n_total)
- `accuracy_answered`: Selective accuracy on answered questions only (n_correct / n_answered)
- `selective_risk`: Error rate on answered questions (n_wrong / n_answered)
- `n_correct`: Number of correct predictions
- `n_wrong`: Number of incorrect predictions

### Results Produced

#### CSV Tables
- `aggregated_results.csv`: Combined metrics across dataset subsets (e.g. all MMLU subjects in one row)
- `detailed_results.csv`: Individual subset metrics (e.g. each MMLU subject as separate row)

#### Visualisations

- `performance_aggregated.png`: Bar chart comparing 4 key metrics (coverage, accuracy_all, accuracy_answered, selective_risk) across all datasets
  
- `performance_subsets_<dataset>.png`: Grid layouts showing per-subset performance for MMLU and MuSR datasets

- `coverage_accuracy_tradeoff.png`: Scatter plot showing the coverage-accuracy trade-off across all datasets:
  - X-axis: Coverage (% questions answered)
  - Y-axis: Accuracy on answered questions

## Advanced Usage

### Custom Datasets

To add a new dataset, modify [qa_eval/datasets.py](qa_eval/datasets.py) with:
1. Dataset loader function
2. Normalisation function
3. Register in the dataset type mapping

### Custom Prompts

Modify [qa_eval/prompts.py](qa_eval/prompts.py) to customise prompt templates and formatting.
