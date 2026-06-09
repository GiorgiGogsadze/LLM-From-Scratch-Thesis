"""Evaluation for from-scratch GPT-2 (no Hugging Face).

Measures loss/perplexity on held-out examples, and compares generated
outputs against references using token-F1 and character similarity.
Both base GPT-2 and finetuned checkpoints are loaded via the custom
GPTModel + gpt_download.py (no transformers library).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from gpt_model import (
    GPTModel,
    generate,
    get_model_config,
    load_weights_into_gpt,
    text_to_token_ids,
    token_ids_to_text,
)
from modal_training_config import TrainConfig

PAD_TOKEN_ID = 50256
IGNORE_INDEX = -100
GPT2_SIZE = "355M"


@dataclass(slots=True)
class EvalConfig:
    split: str = "test"
    loss_examples: int | None = None
    generation_examples: int = 50
    qualitative_examples: int = 8
    batch_size: int = 4
    max_new_tokens: int = 128
    seed: int = 123
    temperature: float = 0.75
    top_k: int = 50
    top_p: float = 0.90
    repetition_penalty: float = 1.05


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def choose_eval_rows(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if limit is None or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    sampled = rows[:]
    rng.shuffle(sampled)
    return sampled[:limit]


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).lower().split()
    ref_tokens = normalize_text(reference).lower().split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = 0
    for t, c in pred_counts.items():
        overlap += min(c, ref_counts.get(t, 0))
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def char_similarity(prediction: str, reference: str) -> float:
    return SequenceMatcher(None, normalize_text(prediction), normalize_text(reference)).ratio()


def format_prompt(instruction: str, input_text: str) -> str:
    prefix = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    prompt = f"{prefix}\n\n### Instruction:\n{instruction}"
    if input_text.strip():
        prompt += f"\n\n### Input:\n{input_text.strip()}"
    prompt += "\n\n### Response:\n"
    return prompt


def resolve_model_checkpoint(model_ref: Path) -> Path:
    candidates = [
        model_ref / "model.pth",
        model_ref / "final_model" / "model.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No scratch model.pth checkpoint found under {model_ref}. "
        "Expected either <ref>/model.pth or <ref>/final_model/model.pth."
    )


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> GPTModel:
    """Load a finetuned model saved by train_scratch.py (model.pth)."""
    cfg = get_model_config("gpt2-medium")
    model = GPTModel(cfg)
    state = torch.load(resolve_model_checkpoint(checkpoint_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_base_model(cache_dir: Path, device: torch.device) -> GPTModel:
    """Load pretrained GPT-2-medium via gpt_download (no HF)."""
    cfg = get_model_config("gpt2-medium")
    from gpt_download import download_and_load_gpt2

    _settings, params = download_and_load_gpt2(model_size=GPT2_SIZE, models_dir=str(cache_dir / "gpt2"))
    model = GPTModel(cfg)
    load_weights_into_gpt(model, params)
    model.to(device)
    model.eval()
    return model


def generate_response(
    model: GPTModel,
    tokenizer: Any,
    instruction: str,
    input_text: str,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> str:
    prompt = format_prompt(instruction, input_text)
    context_size = model.pos_emb.weight.shape[0]
    enc = text_to_token_ids(prompt, tokenizer).to(device)
    with torch.no_grad():
        ids = generate(model, enc, max_new_tokens=max_new_tokens,
                       context_size=context_size, temperature=temperature,
                       top_k=top_k, top_p=top_p,
                       repetition_penalty=repetition_penalty,
                       eos_id=PAD_TOKEN_ID)
    response = token_ids_to_text(ids, tokenizer)
    return normalize_text(response[len(prompt):])


def compute_loss(
    model: GPTModel,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    cfg: TrainConfig,
    eval_cfg: EvalConfig,
    device: torch.device,
) -> dict[str, float]:
    """Compute loss/perplexity on a set of examples using the custom model."""
    if not rows:
        return {"loss": math.nan, "perplexity": math.nan}

    class _EvalDataset(Dataset):
        def __init__(self, rows_inner):
            self.data = []
            for row in rows_inner:
                prompt = format_prompt(row["instruction"], row["input"])
                response = row["output"]
                full_text = prompt + response
                tokens = tokenizer.encode(full_text, allowed_special={"<|endoftext|>"})
                self.data.append(tokens)

        def __getitem__(self, idx):
            return self.data[idx]

        def __len__(self):
            return len(self.data)

    def _collate(batch):
        """Simple collate with prompt masking for eval."""
        batch_max = max(len(item) + 1 for item in batch)
        inp_ids, labs = [], []
        for item in batch:
            new_item = item + [PAD_TOKEN_ID]
            padded = new_item + [PAD_TOKEN_ID] * (batch_max - len(new_item))
            inputs = torch.tensor(padded[:-1])
            targets = torch.tensor(padded[1:])
            mask = targets == PAD_TOKEN_ID
            idxs = torch.nonzero(mask).squeeze()
            if idxs.numel() > 1:
                targets[idxs[1:]] = IGNORE_INDEX
            if cfg.block_size is not None:
                inputs = inputs[:cfg.block_size]
                targets = targets[:cfg.block_size]
            inp_ids.append(inputs)
            labs.append(targets)
        return torch.stack(inp_ids).to(device), torch.stack(labs).to(device)

    loader = DataLoader(_EvalDataset(rows), batch_size=eval_cfg.batch_size,
                        collate_fn=_collate, shuffle=False)
    total_loss = 0.0
    total = 0
    model.eval()
    with torch.no_grad():
        for input_batch, target_batch in loader:
            logits = model(input_batch)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
            bs = input_batch.size(0)
            total_loss += float(loss.item()) * bs
            total += bs
    mean_loss = total_loss / total
    return {"loss": mean_loss, "perplexity": float(math.exp(mean_loss))}


def _seed_torch_for_generation(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_generations(
    base_model: GPTModel,
    tuned_model: GPTModel,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    eval_cfg: EvalConfig,
    device: torch.device,
) -> dict[str, Any]:
    _seed_torch_for_generation(eval_cfg.seed)
    records: list[dict[str, Any]] = []
    for row in rows:
        base_output = generate_response(base_model, tokenizer, row["instruction"],
                                        row["input"], eval_cfg.max_new_tokens, device,
                                        temperature=eval_cfg.temperature, top_k=eval_cfg.top_k,
                                        top_p=eval_cfg.top_p, repetition_penalty=eval_cfg.repetition_penalty)
        tuned_output = generate_response(tuned_model, tokenizer, row["instruction"],
                                         row["input"], eval_cfg.max_new_tokens, device,
                                         temperature=eval_cfg.temperature, top_k=eval_cfg.top_k,
                                         top_p=eval_cfg.top_p, repetition_penalty=eval_cfg.repetition_penalty)
        reference = normalize_text(row["output"])
        input_text = normalize_text(row["input"])
        records.append({
            "source_row_id": row["source_row_id"],
            "instruction": row["instruction"],
            "input": input_text,
            "reference": reference,
            "task_mode": row.get("task_mode"),
            "base_output": base_output,
            "finetuned_output": tuned_output,
            "base_token_f1": token_f1(base_output, reference),
            "finetuned_token_f1": token_f1(tuned_output, reference),
            "base_char_similarity": char_similarity(base_output, reference),
            "finetuned_char_similarity": char_similarity(tuned_output, reference),
            "base_changed_input": base_output != input_text,
            "finetuned_changed_input": tuned_output != input_text,
        })

    def avg(key: str) -> float:
        return sum(float(r[key]) for r in records) / len(records) if records else math.nan

    return {
        "count": len(records),
        "base_avg_token_f1": avg("base_token_f1"),
        "finetuned_avg_token_f1": avg("finetuned_token_f1"),
        "base_avg_char_similarity": avg("base_char_similarity"),
        "finetuned_avg_char_similarity": avg("finetuned_char_similarity"),
        "base_changed_rate": avg("base_changed_input"),
        "finetuned_changed_rate": avg("finetuned_changed_input"),
        "finetuned_better_token_f1": sum(1 for r in records if r["finetuned_token_f1"] > r["base_token_f1"]),
        "finetuned_better_char_similarity": sum(1 for r in records if r["finetuned_char_similarity"] > r["base_char_similarity"]),
        "qualitative_examples": records[: eval_cfg.qualitative_examples],
    }


def evaluate_finetuned_generations(
    tuned_model: GPTModel,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    eval_cfg: EvalConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Generate/evaluate finetuned outputs only.

    Used by the Modal smoke path to avoid loading the base GPT-2 model.
    """
    _seed_torch_for_generation(eval_cfg.seed)
    records: list[dict[str, Any]] = []
    for row in rows:
        tuned_output = generate_response(
            tuned_model,
            tokenizer,
            row["instruction"],
            row["input"],
            eval_cfg.max_new_tokens,
            device,
            temperature=eval_cfg.temperature,
            top_k=eval_cfg.top_k,
            top_p=eval_cfg.top_p,
            repetition_penalty=eval_cfg.repetition_penalty,
        )
        reference = normalize_text(row["output"])
        input_text = normalize_text(row["input"])
        records.append(
            {
                "source_row_id": row["source_row_id"],
                "instruction": row["instruction"],
                "input": input_text,
                "reference": reference,
                "task_mode": row.get("task_mode"),
                "base_output": None,
                "finetuned_output": tuned_output,
                "base_token_f1": None,
                "finetuned_token_f1": token_f1(tuned_output, reference),
                "base_char_similarity": None,
                "finetuned_char_similarity": char_similarity(tuned_output, reference),
                "base_changed_input": None,
                "finetuned_changed_input": tuned_output != input_text,
            }
        )

    def avg(key: str) -> float:
        return sum(float(r[key]) for r in records) / len(records) if records else math.nan

    return {
        "base_skipped": True,
        "count": len(records),
        "base_avg_token_f1": None,
        "finetuned_avg_token_f1": avg("finetuned_token_f1"),
        "base_avg_char_similarity": None,
        "finetuned_avg_char_similarity": avg("finetuned_char_similarity"),
        "base_changed_rate": None,
        "finetuned_changed_rate": avg("finetuned_changed_input"),
        "finetuned_better_token_f1": None,
        "finetuned_better_char_similarity": None,
        "qualitative_examples": records[: eval_cfg.qualitative_examples],
    }


def run_comparison(
    *,
    prepared_dir: Path,
    finetuned_model_dir: Path,
    cache_dir: Path,
    output_path: Path,
    train_cfg: TrainConfig,
    eval_cfg: EvalConfig,
    include_base: bool = True,
) -> dict[str, Any]:
    rows = load_jsonl(prepared_dir / f"{eval_cfg.split}.jsonl")

    # Determinism notes:
    # - Example selection is seeded: loss uses eval_cfg.seed; generation uses (seed + 17) so the
    #   two subsets are stable but not identical.
    # - Generation seeds Python random + torch (CPU/CUDA) via _seed_torch_for_generation(seed).
    # - Exact token outputs can still differ across CUDA kernels/hardware/torch versions.
    loss_rows = choose_eval_rows(rows, eval_cfg.loss_examples, eval_cfg.seed)
    generation_rows = choose_eval_rows(rows, eval_cfg.generation_examples, eval_cfg.seed + 17)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    base_model = None
    base_loss: dict[str, float | None] = {"loss": None, "perplexity": None}
    if include_base:
        base_model = load_base_model(cache_dir, device)
        base_loss = compute_loss(base_model, tokenizer, loss_rows, train_cfg, eval_cfg, device)

    tuned_model = load_model_from_checkpoint(finetuned_model_dir, device)
    tuned_loss = compute_loss(tuned_model, tokenizer, loss_rows, train_cfg, eval_cfg, device)

    if include_base and base_model is not None:
        generation_summary = evaluate_generations(base_model, tuned_model, tokenizer, generation_rows, eval_cfg, device)
    else:
        generation_summary = evaluate_finetuned_generations(tuned_model, tokenizer, generation_rows, eval_cfg, device)

    report = {
        "eval_config": asdict(eval_cfg),
        "train_config": asdict(train_cfg),
        "device": str(device),
        "base_included": include_base,
        "split_size": len(rows),
        "loss_subset_size": len(loss_rows),
        "generation_subset_size": len(generation_rows),
        "base_loss": base_loss,
        "finetuned_loss": tuned_loss,
        "generation_summary": generation_summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
