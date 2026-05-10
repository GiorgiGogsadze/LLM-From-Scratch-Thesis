from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_DATA_DIR = PROJECT_ROOT / "data"
LOCAL_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOCAL_CACHE_DIR = PROJECT_ROOT / "hf_cache"


@dataclass(slots=True)
class LocalPaths:
    """Local file-system paths used before data is uploaded to Modal."""

    prepared_dir: Path = LOCAL_DATA_DIR / "prepared"
    interim_dir: Path = LOCAL_DATA_DIR / "interim"
    outputs_dir: Path = LOCAL_OUTPUTS_DIR
    cache_dir: Path = LOCAL_CACHE_DIR


@dataclass(slots=True)
class SourcePaths:
    """Local paths for datasets that are not trivially available on Hugging Face."""

    grammar_json: Path | None = None
    gyafc_dir: Path | None = None
    parasci_dir: Path | None = None


@dataclass(slots=True)
class DatasetBuildConfig:
    """Configuration for building the mixed instruction dataset."""

    seed: int = 123
    max_examples_per_source: int | None = None
    include_grammar: bool = True
    include_gyafc: bool = True
    include_parasci: bool = True
    include_human_ai: bool = True
    include_jfleg_train_support: bool = False
    human_ai_max_chars: int = 1_500
    human_ai_min_chars: int = 80
    human_ai_min_words: int = 30
    human_ai_max_words: int = 450
    human_ai_min_word_overlap: float = 0.15
    human_ai_min_char_similarity: float = 0.0
    human_ai_min_length_ratio: float = 0.55
    human_ai_max_length_ratio: float = 2.50
    pilot_examples: int = 8_000
    train_ratio: float = 0.85
    val_ratio: float = 0.05
    instruction_templates: dict[str, list[str]] = field(
        default_factory=lambda: {
            "grammar": [
                "Correct the grammar of the text.",
                "Correct grammar mistakes and improve fluency.",
            ],
            "academic": [
                "Rewrite the text in polished academic English.",
                "Rewrite the text in a more professional and formal style.",
                "Improve clarity and conciseness while preserving meaning.",
            ],
            "natural": [
                "Rewrite the text to sound more natural and fluent while preserving meaning.",
                "Rewrite the text in a more human and less robotic style.",
            ],
        }
    )


@dataclass(slots=True)
class TrainConfig:
    """Training configuration for the Modal run."""

    model_name: str = "gpt2-medium"
    block_size: int = 768
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    num_train_epochs: float = 2.0
    warmup_ratio: float = 0.03
    logging_steps: int = 25
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2
    max_grad_norm: float = 1.0
    seed: int = 123
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass(slots=True)
class ModalConfig:
    """Modal deployment configuration for training."""

    app_name: str = "chapter7-academic-writing"
    image_python: str = "3.11"
    gpu: str = "L40S"
    timeout_seconds: int = 60 * 60 * 12
    data_volume_name: str = "chapter7-academic-writing-data"
    outputs_volume_name: str = "chapter7-academic-writing-outputs"
    cache_volume_name: str = "chapter7-academic-writing-cache"
    remote_root: str = "/root/project"
    remote_data_dir: str = "/vol/data"
    remote_outputs_dir: str = "/vol/outputs"
    remote_cache_dir: str = "/vol/cache"
    prepared_subdir: str = "prepared"
