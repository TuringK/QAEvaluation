import re
import string
from typing import Optional

from qa_eval.prompts import OPTION_LETTERS, letter_to_index


# ================================
# = Text normalisation utilities =
# ================================


def normalise_answer(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def extract_number(text: str) -> Optional[float]:
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    numbers = re.findall(r"-?\d+\.?\d*", text)

    if not numbers:
        return None

    try:
        return float(numbers[-1])
    except ValueError:
        return None


def numbers_equal(a: float, b: float, tolerance: float = 1e-6) -> bool:
    return abs(a - b) < tolerance


# =====================================
# = Multiple-choice answer extraction =
# =====================================


def extract_mc_letter(text: str, num_options: int = 4) -> Optional[str]:
    if not text:
        return None

    text = text.strip()
    valid_letters = set(OPTION_LETTERS[:num_options])

    # Look for explicit answer patterns first
    patterns = [
        r"(?:answer|choice|option)[\s:]*\(?([A-Z])\)?",
        r"^\(?([A-Z])\)?[\.\):\s]",
        r"\(?([A-Z])\)?[\.\)]?\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in valid_letters:
                return letter

    # Find first valid capital letter
    for char in text:
        if char.upper() in valid_letters:
            return char.upper()

    return None


def match_option_by_text(
    text: str,
    options: list[str],
    threshold: float = 0.8,
) -> Optional[int]:
    text_norm = normalise_answer(text)

    if not text_norm:
        return None

    best_idx = None
    best_score = 0.0

    for idx, option in enumerate(options):
        option_norm = normalise_answer(option)

        if option_norm in text_norm:
            score = len(option_norm) / len(text_norm) if text_norm else 0
            if score > best_score:
                best_score = score
                best_idx = idx
        elif text_norm in option_norm:
            score = len(text_norm) / len(option_norm) if option_norm else 0
            if score > best_score:
                best_score = score
                best_idx = idx

    return best_idx if best_score >= threshold else None


# =====================
# = Scoring functions =
# =====================


def score_mc_example(
    example: dict,
    prediction: str,
    use_text_fallback: bool = True,
) -> dict:
    options = example["options"]
    correct_idx = example["correct_option_idx"]
    num_options = len(options)

    extracted_letter = extract_mc_letter(prediction, num_options)

    if extracted_letter is not None:
        predicted_idx = letter_to_index(extracted_letter)
        extraction_method = "letter"
    elif use_text_fallback:
        predicted_idx = match_option_by_text(prediction, options)
        extraction_method = "text_match" if predicted_idx is not None else "failed"
    else:
        predicted_idx = None
        extraction_method = "failed"

    is_correct = predicted_idx == correct_idx if predicted_idx is not None else False

    return {
        "id": example.get("id"),
        "correct": is_correct,
        "predicted_idx": predicted_idx,
        "correct_idx": correct_idx,
        "extracted_letter": extracted_letter,
        "extraction_method": extraction_method,
        "raw_prediction": prediction,
    }


def score_mc_examples(
    examples: list[dict],
    predictions: list[str],
    use_text_fallback: bool = True,
) -> dict:
    assert len(examples) == len(predictions), "Examples and predictions must match"

    results = [
        score_mc_example(ex, pred, use_text_fallback)
        for ex, pred in zip(examples, predictions)
    ]

    n_correct = sum(r["correct"] for r in results)
    n_total = len(results)
    n_extracted = sum(r["extraction_method"] != "failed" for r in results)

    return {
        "accuracy": n_correct / n_total if n_total > 0 else 0.0,
        "n": n_total,
        "n_correct": n_correct,
        "n_extracted": n_extracted,
        "extraction_rate": n_extracted / n_total if n_total > 0 else 0.0,
        "per_example": results,
    }


def score_open_example(
    example: dict,
    prediction: str,
    try_numeric: bool = True,
) -> dict:
    gold_answers = example["answer"]
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]

    pred_norm = normalise_answer(prediction)
    is_correct = False
    match_type = "none"

    # Try numeric comparison first (for GSM8K-style)
    if try_numeric:
        pred_num = extract_number(prediction)
        if pred_num is not None:
            for gold in gold_answers:
                gold_num = extract_number(gold)
                if gold_num is not None and numbers_equal(pred_num, gold_num):
                    is_correct = True
                    match_type = "numeric"
                    break

    # Normalised string comparison
    if not is_correct:
        gold_norms = [normalise_answer(g) for g in gold_answers]
        if pred_norm in gold_norms:
            is_correct = True
            match_type = "exact"
        else:
            for gold_norm in gold_norms:
                if gold_norm and gold_norm in pred_norm:
                    is_correct = True
                    match_type = "contains"
                    break

    return {
        "id": example.get("id"),
        "correct": is_correct,
        "match_type": match_type,
        "gold_answers": gold_answers,
        "raw_prediction": prediction,
        "normalized_prediction": pred_norm,
    }


def score_open_examples(
    examples: list[dict],
    predictions: list[str],
    try_numeric: bool = True,
) -> dict:
    assert len(examples) == len(predictions), "Examples and predictions must match"

    results = [
        score_open_example(ex, pred, try_numeric)
        for ex, pred in zip(examples, predictions)
    ]

    n_correct = sum(r["correct"] for r in results)
    n_total = len(results)

    match_types = {}
    for r in results:
        mt = r["match_type"]
        match_types[mt] = match_types.get(mt, 0) + 1

    return {
        "accuracy": n_correct / n_total if n_total > 0 else 0.0,
        "n": n_total,
        "n_correct": n_correct,
        "match_type_counts": match_types,
        "per_example": results,
    }


def score_examples(
    examples: list[dict],
    predictions: list[str],
    dataset_type: str = "auto",
) -> dict:
    if dataset_type == "auto":
        if examples and "options" in examples[0]:
            dataset_type = "mc"
        else:
            dataset_type = "open"

    if dataset_type == "mc":
        return score_mc_examples(examples, predictions)
    else:
        return score_open_examples(examples, predictions)
