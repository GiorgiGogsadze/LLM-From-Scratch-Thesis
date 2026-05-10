"""From-scratch instruction finetuning for GPT-2 (no Hugging Face Trainer).

Replicates the reference approach (Chapter7 (1).py):
- Custom GPTModel with tiktoken
- Manual training loop with cosine LR, gradient clipping, mixed precision
- LoRA adapters (only ~0.3% params trainable)
- Best-val-loss checkpointing
- Multi-stage continuation via torch.load/torch.save
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from gpt_model import (
    GPTModel,
    apply_lora_to_model,
    get_model_config,
    load_weights_into_gpt,
    merge_lora_into_base,
    remove_lora_wrappers,
    generate,
    text_to_token_ids,
    token_ids_to_text,
)
from modal_training_config import TrainConfig

PAD_TOKEN_ID = 50256
IGNORE_INDEX = -100
GPT2_SIZE = "355M"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_prompt(instruction: str, input_text: str) -> str:
    prefix = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    prompt = f"{prefix}\n\n### Instruction:\n{instruction}"
    if input_text.strip():
        prompt += f"\n\n### Input:\n{input_text.strip()}"
    prompt += "\n\n### Response:\n"
    return prompt


class InstructionDataset(Dataset):
    def __init__(self, data: list[dict[str, Any]], tokenizer: Any) -> None:
        self.encoded_texts: list[list[int]] = []
        for entry in data:
            full_text = format_prompt(entry["instruction"], entry["input"]) + entry["output"]
            self.encoded_texts.append(
                tokenizer.encode(full_text, allowed_special={"<|endoftext|>"})
            )

    def __getitem__(self, index: int) -> list[int]:
        return self.encoded_texts[index]

    def __len__(self) -> int:
        return len(self.encoded_texts)


def custom_collate_fn(
    batch: list[list[int]],
    pad_token_id: int = PAD_TOKEN_ID,
    ignore_index: int = IGNORE_INDEX,
    allowed_max_length: int | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic padding with prompt masking via ignore_index on extra EOS tokens."""
    if device is None:
        device = torch.device("cpu")
    batch_max = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item + [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        idxs = torch.nonzero(mask).squeeze()
        if idxs.numel() > 1:
            targets[idxs[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    return torch.stack(inputs_lst).to(device), torch.stack(targets_lst).to(device)


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    logits = model(input_batch.to(device))
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.to(device).flatten()
    )


def calc_loss_loader(
    loader: DataLoader, model: GPTModel, device: torch.device, num_batches: int | None = None
) -> float:
    if len(loader) == 0:
        return float("nan")
    num = num_batches if num_batches is not None else len(loader)
    num = min(num, len(loader))
    total = 0.0
    for i, (inp, tgt) in enumerate(loader):
        if i >= num:
            break
        total += calc_loss_batch(inp, tgt, model, device).item()
    return total / num


def evaluate_model(
    model: GPTModel, train_loader: DataLoader, val_loader: DataLoader,
    device: torch.device, eval_iter: int,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        tl = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        vl = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return tl, vl


def generate_and_print_sample(
    model: GPTModel, tokenizer: Any, device: torch.device, start_context: str,
) -> None:
    model.eval()
    ctx = model.pos_emb.weight.shape[0]
    enc = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        ids = generate(model, enc, max_new_tokens=50, context_size=ctx,
                       top_k=50, temperature=0.5, eos_id=PAD_TOKEN_ID)
    print(token_ids_to_text(ids, tokenizer).replace("\n", " "))
    model.train()


def resolve_model_checkpoint(model_init_ref: str | Path) -> Path:
    ref = Path(model_init_ref)
    candidates = [
        ref / "model.pth",
        ref / "final_model" / "model.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No scratch model.pth checkpoint found under {ref}. "
        "Expected either <ref>/model.pth or <ref>/final_model/model.pth. "
        "Existing Hugging Face Trainer checkpoints must be converted or retrained."
    )


def train_model(
    *,
    prepared_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    cfg: TrainConfig,
    model_init_ref: str | Path | None = None,
    outputs_volume: Any | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mixed precision: bf16={cfg.bf16}, LoRA rank={cfg.lora_rank}")

    # 1. Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 2. Load data
    train_data = load_jsonl(prepared_dir / "train.jsonl")
    val_data = load_jsonl(prepared_dir / "val.jsonl")
    test_data = load_jsonl(prepared_dir / "test.jsonl")
    if cfg.max_train_samples and cfg.max_train_samples > 0:
        train_data = train_data[: cfg.max_train_samples]
    if cfg.max_eval_samples and cfg.max_eval_samples > 0:
        val_data = val_data[: cfg.max_eval_samples]
        test_data = test_data[: cfg.max_eval_samples]
    print(f"Data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    start_context = format_prompt(val_data[0]["instruction"], val_data[0]["input"])

    # 3. Dataloaders
    collate_fn = partial(custom_collate_fn, allowed_max_length=cfg.block_size, device=device)
    train_loader = DataLoader(InstructionDataset(train_data, tokenizer), batch_size=cfg.train_batch_size,
                              collate_fn=collate_fn, shuffle=True, drop_last=True)
    val_loader = DataLoader(InstructionDataset(val_data, tokenizer), batch_size=cfg.train_batch_size,
                            collate_fn=collate_fn, shuffle=False, drop_last=False)
    test_loader = DataLoader(InstructionDataset(test_data, tokenizer), batch_size=cfg.train_batch_size,
                             collate_fn=collate_fn, shuffle=False, drop_last=False)

    # 4. Model
    model_cfg = get_model_config("gpt2-medium")

    if model_init_ref is not None:
        checkpoint_path = resolve_model_checkpoint(model_init_ref)
        state = torch.load(checkpoint_path, map_location="cpu")
        model = GPTModel(model_cfg)
        model.load_state_dict(state)
        print(f"Resumed from checkpoint: {checkpoint_path}")
    else:
        print("Downloading pretrained GPT-2 weights...")
        from gpt_download import download_and_load_gpt2
        _settings, params = download_and_load_gpt2(model_size=GPT2_SIZE, models_dir=str(cache_dir / "gpt2"))
        model = GPTModel(model_cfg)
        load_weights_into_gpt(model, params)
        print("Pretrained weights loaded")

    model.to(device)

    # 5. LoRA
    if cfg.lora_rank > 0:
        model = apply_lora_to_model(model, rank=cfg.lora_rank, alpha=cfg.lora_alpha)
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    lora = sum(p.numel() for p in trainable)
    print(f"Total params: {total:,} | Trainable LoRA: {lora:,} ({lora/total*100:.2f}%)")

    # 6. Effective batch size: per_step * grad_accum
    effective_batch = cfg.train_batch_size * cfg.gradient_accumulation_steps
    print(f"Per-step batch: {cfg.train_batch_size}, grad_accum: {cfg.gradient_accumulation_steps}, effective batch: {effective_batch}")

    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * int(cfg.num_train_epochs)
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    print(f"{int(cfg.num_train_epochs)} epochs, {total_steps} total steps, {warmup_steps} warmup")

    # 7. Training loop (follows reference: GradScaler, cosine LR, best-val checkpoint)
    scaler = torch.amp.GradScaler(device) if cfg.bf16 else None
    best_val_loss = float("inf")
    global_step = 0
    tokens_seen = 0
    train_losses, val_losses, track_seen = [], [], []
    peak_lr = cfg.learning_rate
    initial_lr = peak_lr * 0.1
    min_lr = peak_lr * 0.02
    lr_inc = (peak_lr - initial_lr) / max(warmup_steps, 1)

    start_time = time.time()
    for epoch in range(int(cfg.num_train_epochs)):
        model.train()
        optimizer.zero_grad()

        for input_batch, target_batch in train_loader:
            global_step += 1

            # LR schedule
            if global_step < warmup_steps:
                lr = initial_lr + global_step * lr_inc
            else:
                progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward + backward with optional mixed precision
            if scaler is not None:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                scaler.scale(loss).backward()
            else:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()

            # Gradient accumulation: only step after accum_steps
            if global_step % cfg.gradient_accumulation_steps == 0:
                if scaler is not None:
                    if global_step > warmup_steps:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if global_step > warmup_steps:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()

            tokens_seen += input_batch.numel()

            # Eval
            if global_step % cfg.eval_steps == 0:
                tl, vl = evaluate_model(model, train_loader, val_loader, device, eval_iter=min(25, len(train_loader)))
                train_losses.append(tl)
                val_losses.append(vl)
                track_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): train {tl:.4f} val {vl:.4f}")

                if vl < best_val_loss:
                    best_val_loss = vl
                    if output_dir is not None:
                        output_dir.mkdir(parents=True, exist_ok=True)
                        lora_state = {k: v for k, v in model.state_dict().items() if "lora" in k}
                        torch.save(lora_state, output_dir / "best_checkpoint_lora.pth")
                        if outputs_volume is not None:
                            outputs_volume.commit()
                        print(f"  -> Best val loss {vl:.4f}, saved LoRA checkpoint")

        generate_and_print_sample(model, tokenizer, device, start_context)

    runtime = time.time() - start_time
    print(f"Training done in {runtime/60:.2f} min")

    # 8. Merge LoRA → base → save final
    merge_lora_into_base(model)
    remove_lora_wrappers(model.trf_blocks)
    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_dir / "model.pth")
    print(f"Final model: {final_dir / 'model.pth'}")

    # 9. Test loss
    model.eval()
    with torch.no_grad():
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=None)
    test_ppl = math.exp(test_loss) if not math.isnan(test_loss) else float("nan")

    summary = {
        "train_metrics": {"train_runtime": runtime, "train_loss": train_losses[-1] if train_losses else None, "epoch": cfg.num_train_epochs},
        "test_metrics": {"eval_loss": test_loss, "eval_perplexity": test_ppl},
        "config": asdict(cfg),
        "model_init_ref": str(model_init_ref) if model_init_ref else "gpt2-medium",
        "prepared_dir": str(prepared_dir),
        "output_dir": str(output_dir),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if outputs_volume is not None:
        outputs_volume.commit()
    return summary
