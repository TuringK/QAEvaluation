from typing import Callable
import datasets as hf_datasets
from qa_eval.config import DatasetConfig
import json
import urllib.request
import random

# ==============================
# = Dataset subset definitions =
# ==============================

MMUL_REDUX_SUBSETS = [
    "anatomy",
    "business_ethics",
    "clinical_knowledge",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "econometrics",
    "electrical_engineering",
    "formal_logic",
    "global_facts",
    "high_school_chemistry",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "human_aging",
    "logical_fallacies",
    "machine_learning",
    "miscellaneous",
    "philosophy",
    "professional_accounting",
    "public_relations",
    "virology",
    "conceptual_physics",
    "high_school_us_history",
    "astronomy",
    "high_school_geography",
    "high_school_macroeconomics",
    "professional_law",
]

MUSR_SUBSETS = ["murder_mystery", "object_placements", "team_allocation"]

# ============================================
# = Dataset specific loaders and normalisers =
# ============================================


def _load_csqa2(cfg: DatasetConfig) -> hf_datasets.Dataset:
    """
    CommonsenseQA 2.0 - Yes/No/I don't know questions
    Official: https://github.com/allenai/csqa2
    Original fields: id, question, answer ("yes"/"no")
    """
    base_url = "https://raw.githubusercontent.com/allenai/csqa2/master/dataset"

    split_map = {
        "train": f"{base_url}/CSQA2_train.json.gz",
        "dev": f"{base_url}/CSQA2_dev.json.gz",
        "validation": f"{base_url}/CSQA2_dev.json.gz",
        "test": f"{base_url}/CSQA2_test_no_answers.json.gz",  # No answers available
    }

    if cfg.split not in split_map:
        raise ValueError(
            f"csqa2 split '{cfg.split}' not available. "
            f"Available: {list(split_map.keys())}"
        )

    if cfg.split == "test":
        raise ValueError(
            "csqa2 test split does not contain answers. Use 'dev' or 'train' instead."
        )

    ds = hf_datasets.load_dataset(
        "json", data_files=split_map[cfg.split], split="train"
    )

    def normalise(ex, index):
        # Binary QA with possible abstention
        # answer field contains list like ["yes", "yes"] from multiple annotators
        # Take majority vote
        answers = ex["answer"]
        if isinstance(answers, list):
            yes_count = sum(1 for a in answers if a.lower() == "yes")
            answer = "yes" if yes_count > len(answers) / 2 else "no"
        else:
            answer = answers.lower().strip()

        options = ["yes", "no"]
        correct_idx = 0 if answer == "yes" else 1
        return {
            "id": ex.get("id", f"csqa2_{index}"),
            "question": ex["question"],
            "options": options,
            "correct_option_idx": correct_idx,
            "answer": answer,  # Keep original for reference
        }

    return ds.map(normalise, with_indices=True, remove_columns=ds.column_names)


def _load_strategyqa(cfg: DatasetConfig) -> hf_datasets.Dataset:
    """
    StrategyQA - Yes/No reasoning questions
    Official: https://github.com/eladsegal/strategyqa
    Original fields: qid, term, description, question, answer (bool), facts, decomposition
    """
    base_url = (
        "https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa"
    )

    split_map = {
        "train": f"{base_url}/train.json",
        "dev": f"{base_url}/dev.json",
        "validation": f"{base_url}/dev.json",
    }

    if cfg.split not in split_map:
        raise ValueError(
            f"StrategyQA split '{cfg.split}' not available. "
            f"Available: {list(split_map.keys())}. Note: No official test split exists."
        )

    url = split_map[cfg.split]
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode("utf-8"))

    # Extract only the fields we need
    clean_data = []
    for item in data:
        clean_data.append(
            {
                "qid": item.get("qid", ""),
                "question": item["question"],
                "answer": item["answer"],
                "facts": item.get("facts", []),
            }
        )

    ds = hf_datasets.Dataset.from_list(clean_data)

    def normalise(ex, index):
        # Answer is boolean
        answer_bool = ex["answer"]
        options = ["yes", "no"]
        correct_idx = 0 if answer_bool else 1
        return {
            "id": ex.get("qid", f"strategyqa_{index}"),
            "question": ex["question"],
            "options": options,
            "correct_option_idx": correct_idx,
            "answer": "yes" if answer_bool else "no",
        }

    return ds.map(normalise, with_indices=True, remove_columns=ds.column_names)


def _load_mmlu_redux(cfg: DatasetConfig) -> hf_datasets.Dataset:
    """
    MMLU-Redux - Cleaned MMLU with error corrections.
    Official: https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux
    Original fields: question, choices (list), answer (int 0-3), subject

    Note: If cfg.subset is None, this should not be called directly.
    Use expand_dataset_configs() to expand into individual subsets.
    """
    # Load specific subset (subset must be provided)
    subset = cfg.subset
    if subset is None:
        raise ValueError(
            "MMLU-Redux requires a subset to be specified. "
            f"Available subsets: {MMUL_REDUX_SUBSETS}"
        )

    ds = hf_datasets.load_dataset(
        "edinburgh-dawg/mmlu-redux",
        name=subset,
        split=cfg.split,
    )

    def normalise(ex, index):
        return {
            "id": f"mmlu_redux_{subset}_{index}" if subset else f"mmlu_redux_{index}",
            "question": ex["question"],
            "options": ex["choices"],
            "correct_option_idx": ex["answer"],
        }

    return ds.map(normalise, with_indices=True, remove_columns=ds.column_names)


def _load_gpqa(cfg: DatasetConfig) -> hf_datasets.Dataset:
    """
    GPQA - Graduate-level science questions.
    Official: https://huggingface.co/datasets/Idavidrein/gpqa
    Subsets: "gpqa_main", "gpqa_diamond", "gpqa_extended"
    Original fields: Question, Correct Answer, Incorrect Answer 1/2/3
    """
    subset = cfg.subset or "gpqa_diamond"
    ds = hf_datasets.load_dataset("Idavidrein/gpqa", name=subset, split=cfg.split)

    def normalise(ex, index):
        # Correct answer is always first, then shuffle
        correct = ex["Correct Answer"]
        incorrect = [
            ex["Incorrect Answer 1"],
            ex["Incorrect Answer 2"],
            ex["Incorrect Answer 3"],
        ]

        # Deterministic shuffle based on index to randomise option order
        rng = random.Random(index)
        options = [correct] + incorrect
        rng.shuffle(options)
        correct_idx = options.index(correct)

        return {
            "id": f"gpqa_{subset}_{index}",
            "question": ex["Question"],
            "options": options,
            "correct_option_idx": correct_idx,
        }

    return ds.map(normalise, with_indices=True, remove_columns=ds.column_names)


def _load_gsm8k(cfg: DatasetConfig) -> hf_datasets.Dataset:
    """
    GSM8K - Grade school math word problems (open-ended).
    Official: https://huggingface.co/datasets/openai/gsm8k
    Original fields: question, answer (contains reasoning + "#### <final>")
    """
    ds = hf_datasets.load_dataset("openai/gsm8k", "main", split=cfg.split)

    def normalise(ex, index):
        # Extract final numerical answer after ####
        full_answer = ex["answer"]
        if "####" in full_answer:
            final_answer = full_answer.split("####")[-1].strip()
        else:
            final_answer = full_answer.strip()

        return {
            "id": f"gsm8k_{index}",
            "question": ex["question"],
            "answer": final_answer,
            "full_solution": full_answer,
        }

    return ds.map(normalise, with_indices=True, remove_columns=ds.column_names)


def _load_agieval(cfg: DatasetConfig) -> hf_datasets.Dataset:
    """
    AGIEval - English MC subsets (SAT, LSAT, etc.).
    Official: https://github.com/ruixiangcui/AGIEval
    Subsets: "sat-en", "sat-math", "lsat-ar", "lsat-lr", "lsat-rc", "logiqa-en", "aqua-rat"
    Original fields: passage (optional), question, options (list), label (str letter)
    """
    subset = cfg.subset or "sat-en"

    # Map subset names to GitHub data files
    english_subsets = {
        "sat-en",
        "sat-math",
        "lsat-ar",
        "lsat-lr",
        "lsat-rc",
        "logiqa-en",
        "aqua-rat",
    }

    if subset not in english_subsets:
        raise ValueError(
            f"AGIEval subset '{subset}' not available. "
            f"Available English subsets: {sorted(english_subsets)}"
        )

    base_url = "https://raw.githubusercontent.com/ruixiangcui/AGIEval/main/data/v1_1"
    data_url = f"{base_url}/{subset}.jsonl"

    try:
        ds = hf_datasets.load_dataset(
            "json",
            data_files=data_url,
            split="train",
        )
    except Exception as e:
        raise ValueError(
            f"Failed to load AGIEval subset '{subset}' from GitHub. Error: {e}"
        )

    def normalise(ex, index):
        question = ex["question"]

        if "passage" in ex and ex["passage"]:
            question = f"{ex['passage']}\n\nQuestion: {question}"

        options = ex["options"]
        label = ex["label"]

        # Label is always a letter A-E
        correct_idx = ord(label.upper()) - ord("A")

        return {
            "id": f"agieval_{subset}_{index}",
            "question": question,
            "options": options,
            "correct_option_idx": correct_idx,
        }

    return ds.map(normalise, with_indices=True, remove_columns=ds.column_names)


def _load_musr(cfg: DatasetConfig) -> hf_datasets.Dataset:
    """
    MuSR - Multi-step Soft Reasoning.
    Official: https://github.com/Zayne-sprague/MuSR
    Subsets: "murder_mystery", "object_placements", "team_allocation"
    Original fields: narrative, question, choices, answer, answer_index

    Note: If cfg.subset is None, this should not be called directly.
    Use expand_dataset_configs() to expand into individual subsets.
    """
    subset = cfg.subset
    if subset is None:
        raise ValueError(
            "MuSR requires a subset to be specified. "
            f"Available subsets: {MUSR_SUBSETS}"
        )

    available_subsets = {
        "murder_mystery",
        "object_placements",
        "team_allocation",
    }

    if subset not in available_subsets:
        raise ValueError(
            f"MuSR subset '{subset}' not available. "
            f"Available: {sorted(available_subsets)}"
        )

    base_url = "https://raw.githubusercontent.com/Zayne-sprague/MuSR/main/datasets"
    data_url = f"{base_url}/{subset}.json"

    try:
        ds = hf_datasets.load_dataset(
            "json",
            data_files=data_url,
            split="train",
        )
    except Exception as e:
        raise ValueError(
            f"Failed to load MuSR subset '{subset}' from GitHub. Error: {e}"
        )

    def normalise(ex, index):
        context = ex.get("context", "")
        questions_list = ex.get("questions", [])

        if not questions_list:
            raise ValueError(f"No questions found in example {index}")

        q = questions_list[0]
        question = q["question"]
        full_question = f"{context}\n\nQuestion: {question}" if context else question

        return {
            "id": f"musr_{subset}_{index}",
            "question": full_question,
            "options": q["choices"],
            "correct_option_idx": q["answer"],
        }

    return ds.map(normalise, with_indices=True, remove_columns=ds.column_names)


# ==============
# = Dispatcher =
# ==============

DATASET_LOADERS: dict[str, Callable[[DatasetConfig], hf_datasets.Dataset]] = {
    "csqa2": _load_csqa2,
    "strategyqa": _load_strategyqa,
    "mmlu_redux": _load_mmlu_redux,
    "gpqa": _load_gpqa,
    "gsm8k": _load_gsm8k,
    "agieval": _load_agieval,
    "musr": _load_musr,
}


def load_dataset_by_config(ds_cfg: DatasetConfig) -> hf_datasets.Dataset:
    if ds_cfg.name not in DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset: {ds_cfg.name} "
            f"Available: {list(DATASET_LOADERS.keys())}"
        )

    loader = DATASET_LOADERS[ds_cfg.name]
    ds = loader(ds_cfg)

    if ds_cfg.shuffle_seed is not None:
        ds = ds.shuffle(seed=ds_cfg.shuffle_seed)

    if ds_cfg.max_examples is not None and ds_cfg.max_examples < len(ds):
        ds = ds.select(range(ds_cfg.max_examples))

    return ds


def get_dataset_type(ds_cfg: DatasetConfig) -> str:
    open_ended = {"gsm8k"}
    return "open" if ds_cfg.name in open_ended else "mc"


def expand_dataset_configs(ds_cfgs: list[DatasetConfig]) -> list[DatasetConfig]:
    expanded = []

    for cfg in ds_cfgs:
        if cfg.name == "mmlu_redux" and cfg.subset is None:
            # Expand to all MMLU-Redux subsets
            for subset in MMUL_REDUX_SUBSETS:
                expanded.append(
                    DatasetConfig(
                        name=cfg.name,
                        split=cfg.split,
                        subset=subset,
                        max_examples=cfg.max_examples,
                        shuffle_seed=cfg.shuffle_seed,
                    )
                )
        elif cfg.name == "musr" and cfg.subset is None:
            # Expand to all MuSR subsets
            for subset in MUSR_SUBSETS:
                expanded.append(
                    DatasetConfig(
                        name=cfg.name,
                        split=cfg.split,
                        subset=subset,
                        max_examples=cfg.max_examples,
                        shuffle_seed=cfg.shuffle_seed,
                    )
                )
        else:
            expanded.append(cfg)

    return expanded
