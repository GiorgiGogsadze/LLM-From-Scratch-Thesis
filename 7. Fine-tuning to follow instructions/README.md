# Scratch Sequential Training Bundle

This folder contains the minimal Python code needed to reproduce the canonical
scratch GPT-2-medium sequential training lineage on Modal:

1. Stage 1 Grammar
2. Stage 2 Naturalness / Humanizer with grammar replay
3. Stage 3 Academic

It intentionally excludes thesis rendering scripts, plots, generated artifacts,
tests, legacy Hugging Face Trainer code, and joint multi-task files.

## Included Python Files

| File | Purpose |
| --- | --- |
| `modal_app.py` | Modal app, remote data-prep entrypoints, remote training entrypoint, and optional scratch evaluation entrypoints. |
| `modal_training_config.py` | Modal volume names, local/remote paths, dataset config, and scratch training hyperparameters. |
| `prepare_datasets.py` | Builds the instruction JSONL datasets for grammar, naturalness, and academic rewriting. |
| `train_scratch.py` | Manual from-scratch GPT-2-medium instruction fine-tuning loop with LoRA and checkpoint continuation. |
| `gpt_model.py` | Raw PyTorch GPT-2 architecture, OpenAI weight loading, generation, and LoRA merge helpers. |
| `evaluate_gpt2.py` | Optional post-training evaluation helper used by `modal_app.py::evaluate-scratch`. |
| `download_scratch_eval_reports.py` | Optional helper for downloading Modal evaluation JSON reports. |

## Expected Modal Volumes

The default volume names are defined in `modal_training_config.py`:

- `chapter7-academic-writing-data`
- `chapter7-academic-writing-outputs`
- `chapter7-academic-writing-cache`

The training commands below assume the source datasets already exist in the
data volume where needed, especially:

- `/vol/data/grammar-data.json`
- GYAFC / ParaSCI local directories if not using the Hugging Face fallbacks

## Stage 1: Grammar

Prepare the grammar dataset:

```bash
modal run modal_app.py::prepare \
  --prepared-subdir prepared \
  --include-grammar \
  --grammar-json-subpath grammar-data.json \
  --max-examples-per-source 12000 \
  --pilot-examples 8000
```

Train Stage 1 from base GPT-2-medium:

```bash
modal run modal_app.py \
  --prepared-subdir prepared \
  --run-name scratch_stage1_grammar
```

Output checkpoint:

```text
/vol/outputs/scratch_stage1_grammar/final_model/model.pth
```

## Stage 2: Naturalness / Humanizer

Prepare the Stage 2 dataset. This mixes filtered
`dmitva/human_ai_generated_text` naturalness pairs with grammar replay:

```bash
modal run modal_app.py::prepare-humanizer-repair \
  --prepared-subdir prepared_humanizer_repaired \
  --grammar-json-subpath grammar-data.json \
  --max-examples-per-source 6000 \
  --pilot-examples 6000 \
  --human-ai-min-words 30 \
  --human-ai-max-words 450 \
  --human-ai-min-word-overlap 0.15 \
  --human-ai-min-char-similarity 0.0 \
  --human-ai-min-length-ratio 0.55 \
  --human-ai-max-length-ratio 2.50
```

Train Stage 2 from Stage 1:

```bash
modal run modal_app.py::train-humanizer-repair \
  --prepared-subdir prepared_humanizer_repaired \
  --run-name scratch_stage2_humanizer_repaired \
  --model-init-subdir scratch_stage1_grammar/final_model
```

Output checkpoint:

```text
/vol/outputs/scratch_stage2_humanizer_repaired/final_model/model.pth
```

## Stage 3: Academic

Prepare the academic dataset from GYAFC and ParaSCI:

```bash
modal run modal_app.py::prepare \
  --prepared-subdir prepared_stage3_remaining_academic \
  --include-gyafc \
  --include-parasci \
  --max-examples-per-source 12000 \
  --pilot-examples 12000
```

Train Stage 3 from Stage 2:

```bash
modal run modal_app.py \
  --prepared-subdir prepared_stage3_remaining_academic \
  --run-name scratch_stage3_academic \
  --model-init-subdir scratch_stage2_humanizer_repaired/final_model
```

Output checkpoint:

```text
/vol/outputs/scratch_stage3_academic/final_model/model.pth
```

## Optional Evaluation

After all three stages finish:

```bash
modal run modal_app.py::evaluate-scratch
python download_scratch_eval_reports.py
```

This evaluates the 3 models against the 3 held-out datasets:

- Grammar held-out set: `prepared`
- Naturalness held-out set: `prepared_humanizer_repaired`
- Academic held-out set: `prepared_stage3_remaining_academic`

## Important Lineage Rule

Do not initialize Stage 2 or Stage 3 from base GPT-2. The thesis lineage is
sequential:

```text
base GPT-2-medium -> scratch_stage1_grammar -> scratch_stage2_humanizer_repaired -> scratch_stage3_academic
```
