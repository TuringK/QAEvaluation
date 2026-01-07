from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    model_name_or_path: str = Field(
        ..., description="HuggingFace model ID or local path"
    )
    revision: Optional[str] = Field(default=None, description="Model revision/branch")
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = Field(
        default="bfloat16", description="Model dtype for inference"
    )
    device: str = Field(
        default="auto", description="Device or device_map for model loading"
    )

    # Generation parameters
    max_new_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    do_sample: bool = Field(default=False)
    batch_size: int = Field(default=1, ge=1)

    # Additional kwargs passed directly to model.generate()
    generation_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("temperature")
    @classmethod
    def validate_temperature_sampling(cls, v: float, info) -> float:
        # Warn if temperature > 0 but do_sample might be False
        return v

    def get_generation_config(self) -> dict[str, Any]:
        base = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
        }
        return {**base, **self.generation_kwargs}


class DatasetConfig(BaseModel):
    name: Literal[
        "csqa2", "strategyqa", "mmlu_redux", "gpqa", "gsm8k", "agieval", "musr"
    ] = Field(..., description="Dataset identifier")
    split: str = Field(default="test", description="Dataset split to evaluate")
    subset: Optional[str] = Field(
        default=None, description="Dataset subset/config (e.g. MMLU subject)"
    )
    max_examples: Optional[int] = Field(
        default=None, ge=1, description="Limit number of examples (None = all)"
    )
    shuffle_seed: Optional[int] = Field(
        default=None, description="Seed for shuffling before truncation"
    )


class PromptConfig(BaseModel):
    style: Literal["default", "cot", "abstention", "minimal"] = Field(
        default="default", description="Prompt template style"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt for chat models"
    )
    few_shot_examples_path: Optional[str] = Field(
        default=None, description="Path to JSON file with few-shot examples"
    )
    include_options_as_letters: bool = Field(
        default=True, description="Format MC options as A/B/C/D"
    )
    force_answer_letter: bool = Field(
        default=True, description="Instruct model to respond with just the letter"
    )


class EvalConfig(BaseModel):
    model: ModelConfig
    datasets: list[DatasetConfig] = Field(
        ..., min_length=1, description="List of datasets to evaluate"
    )
    prompt: PromptConfig = Field(default_factory=PromptConfig)

    # Output settings
    output_dir: str = Field(default="./outputs")
    run_name: str = Field(default="eval_run")

    # Runtime settings
    num_workers: int = Field(default=0, ge=0)
    log_every: int = Field(default=50, ge=1)

    def get_output_path(self) -> Path:
        return Path(self.output_dir) / self.run_name


def load_eval_config(path: str) -> EvalConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    return EvalConfig.model_validate_json(config_path.read_text())


# =============================================================================
# Example JSON config (for reference):
# =============================================================================
#
# {
#   "model": {
#     "model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
#     "revision": null,
#     "dtype": "bfloat16",
#     "device": "auto",
#     "max_new_tokens": 512,
#     "temperature": 0.6,
#     "top_p": 0.95,
#     "top_k": 20,
#     "do_sample": true,
#     "batch_size": 8,
#     "generation_kwargs": {}
#   },
#   "datasets": [
#     {
#       "name": "mmlu_redux",
#       "split": "test",
#       "subset": "abstract_algebra",
#       "max_examples": null,
#       "shuffle_seed": null
#     }
#   ],
#   "prompt": {
#     "style": "default",
#     "system_prompt": "You are a helpful assistant. Answer the following multiple choice question by responding with just the letter of the correct answer.",
#     "few_shot_examples_path": null,
#     "include_options_as_letters": true,
#     "force_answer_letter": true
#   },
#   "output_dir": "./outputs",
#   "run_name": "llama3_8b_mmlu_redux",
#   "num_workers": 0,
#   "log_every": 50
# }
#
