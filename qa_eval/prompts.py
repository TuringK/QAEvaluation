import string
from typing import Optional

from qa_eval.config import PromptConfig


# Letter labels for MC options
OPTION_LETTERS = list(string.ascii_uppercase)


def _format_mc_options(options: list[str], as_letters: bool = True) -> str:
    if as_letters:
        lines = [f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
    else:
        lines = [f"- {opt}" for opt in options]
    return "\n".join(lines)


def _build_mc_prompt(
    question: str,
    options: list[str],
    prompt_cfg: PromptConfig,
    system_prompt: Optional[str] = None,
) -> str:
    parts = []

    # System prompt
    if system_prompt:
        parts.append(f"System: {system_prompt}")
        parts.append("")

    # Question
    parts.append(f"Question: {question}")
    parts.append("")

    # Options
    options_str = _format_mc_options(
        options, as_letters=prompt_cfg.include_options_as_letters
    )
    parts.append("Options:")
    parts.append(options_str)
    parts.append("")

    # Answer instruction
    if prompt_cfg.force_answer_letter and prompt_cfg.include_options_as_letters:
        parts.append("Answer with only the letter of the correct option.")
    else:
        parts.append("Answer:")

    return "\n".join(parts)


def _build_open_qa_prompt(
    question: str,
    prompt_cfg: PromptConfig,
    system_prompt: Optional[str] = None,
) -> str:
    parts = []

    # System prompt
    if system_prompt:
        parts.append(f"System: {system_prompt}")
        parts.append("")

    # Question
    parts.append(f"Question: {question}")
    parts.append("")

    # Instruction
    if prompt_cfg.style == "cot":
        parts.append(
            "Think step-by-step to solve this problem. "
            "Show your reasoning, then provide your final answer after 'Final answer:'."
        )
    elif prompt_cfg.style == "minimal":
        parts.append("Answer:")
    else:
        parts.append("Give a short answer.")
        parts.append("")
        parts.append("Answer:")

    return "\n".join(parts)


def _build_abstention_mc_prompt(
    question: str,
    options: list[str],
    prompt_cfg: PromptConfig,
    system_prompt: Optional[str] = None,
) -> str:
    parts = []

    # System prompt with abstention instruction
    abstention_system = (
        system_prompt
        or "You are a helpful assistant. If you are uncertain about the answer, "
        "it is better to say you don't know than to guess."
    )
    parts.append(f"System: {abstention_system}")
    parts.append("")

    # Question
    parts.append(f"Question: {question}")
    parts.append("")

    # Options with abstention as last option
    options_with_abstention = options + ["I don't know"]
    options_str = _format_mc_options(
        options_with_abstention, as_letters=prompt_cfg.include_options_as_letters
    )
    parts.append("Options:")
    parts.append(options_str)
    parts.append("")

    # Instruction
    if prompt_cfg.force_answer_letter and prompt_cfg.include_options_as_letters:
        parts.append(
            "Answer with only the letter of the correct option. "
            f"Choose {OPTION_LETTERS[len(options)]} if you are not sure."
        )
    else:
        parts.append("Answer:")

    return "\n".join(parts)


def _build_abstention_open_prompt(
    question: str,
    prompt_cfg: PromptConfig,
    system_prompt: Optional[str] = None,
) -> str:
    parts = []

    abstention_system = (
        system_prompt
        or "You are a helpful assistant. If you are uncertain about the answer, "
        "respond with 'I don't know' rather than guessing."
    )
    parts.append(f"System: {abstention_system}")
    parts.append("")

    parts.append(f"Question: {question}")
    parts.append("")

    if prompt_cfg.style == "cot":
        parts.append(
            "Think step-by-step. If you can solve it, show your reasoning and "
            "provide your final answer after 'Final answer:'. "
            "If you are unsure, respond with 'I don't know'."
        )
    else:
        parts.append(
            "Give a short answer. If you are unsure, respond with 'I don't know'."
        )
        parts.append("")
        parts.append("Answer:")

    return "\n".join(parts)


def build_prompt(
    example: dict,
    prompt_cfg: PromptConfig,
    dataset_name: str,
) -> str:
    question = example["question"]
    options = example.get("options")
    system_prompt = prompt_cfg.system_prompt

    is_mc = options is not None and len(options) > 0

    if prompt_cfg.style == "abstention":
        if is_mc:
            return _build_abstention_mc_prompt(
                question, options, prompt_cfg, system_prompt
            )
        else:
            return _build_abstention_open_prompt(question, prompt_cfg, system_prompt)
    else:
        if is_mc:
            return _build_mc_prompt(question, options, prompt_cfg, system_prompt)
        else:
            return _build_open_qa_prompt(question, prompt_cfg, system_prompt)


def get_answer_prefix(prompt_cfg: PromptConfig, is_mc: bool) -> str:
    if is_mc and prompt_cfg.force_answer_letter:
        return ""  # Expect just a letter
    if prompt_cfg.style == "cot":
        return "Final answer:"
    return ""


def letter_to_index(letter: str) -> int:
    return ord(letter.upper()) - ord("A")


def index_to_letter(index: int) -> str:
    return OPTION_LETTERS[index]
