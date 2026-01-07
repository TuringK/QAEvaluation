from typing import Any

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from qa_eval.config import ModelConfig

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "auto": "auto",
}


def load_model_and_tokeniser(
    model_cfg: ModelConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    torch_dtype = DTYPE_MAP.get(model_cfg.dtype, torch.bfloat16)

    if model_cfg.device == "auto":
        device_map = "auto"
    elif model_cfg.device.startswith("cuda"):
        device_map = model_cfg.device
    else:
        device_map = None

    tokeniser = AutoTokenizer.from_pretrained(
        model_cfg.model_name_or_path,
        revision=model_cfg.revision,
        trust_remote_code=True,
    )

    # Use eos_token if not set
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token
        tokeniser.pad_token_id = tokeniser.eos_token_id

    model_kwargs: dict[str, Any] = {
        "revision": model_cfg.revision,
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }

    if device_map is not None:
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        **model_kwargs,
    )

    if device_map is None and model_cfg.device != "cpu":
        model = model.to(model_cfg.device)
    elif device_map is None:
        model = model.to("cpu")

    model.eval()

    return model, tokeniser


def get_model_device(model: PreTrainedModel) -> torch.device:
    return next(model.parameters()).device


def generate_batches(
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizerBase,
    prompts: list[str],
    model_cfg: ModelConfig,
    show_progress: bool = True,
) -> list[str]:
    device = get_model_device(model)
    batch_size = model_cfg.batch_size
    generation_config = model_cfg.get_generation_config()

    all_completions: list[str] = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    iterator = range(0, len(prompts), batch_size)
    if show_progress:
        iterator = tqdm(
            iterator,
            total=num_batches,
            desc="Generating",
            unit="batch",
        )

    with torch.no_grad():
        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            tokeniser.padding_side = "left"
            inputs = tokeniser(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(device)

            prompt_lengths = inputs["attention_mask"].sum(dim=1)

            outputs = model.generate(
                **inputs,
                **generation_config,
                pad_token_id=tokeniser.pad_token_id,
                eos_token_id=tokeniser.eos_token_id,
            )

            # Exclude prompt
            for i, (output, prompt_len) in enumerate(zip(outputs, prompt_lengths)):
                generated_tokens = output[prompt_len:]
                completion = tokeniser.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                all_completions.append(completion.strip())

    return all_completions


def generate_single(
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizerBase,
    prompt: str,
    model_cfg: ModelConfig,
) -> str:
    completions = generate_batches(
        model, tokeniser, [prompt], model_cfg, show_progress=False
    )
    return completions[0]


def count_parameters(model: PreTrainedModel) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def get_model_info(
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    param_counts = count_parameters(model)
    return {
        "model_class": model.__class__.__name__,
        "device": str(get_model_device(model)),
        "dtype": str(next(model.parameters()).dtype),
        "total_parameters": param_counts["total"],
        "trainable_parameters": param_counts["trainable"],
        "vocab_size": tokeniser.vocab_size,
        "pad_token": tokeniser.pad_token,
        "eos_token": tokeniser.eos_token,
    }
