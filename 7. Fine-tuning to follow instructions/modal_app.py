from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR / "project"
if PROJECT_DIR.exists():
    sys.path.insert(0, str(PROJECT_DIR))
else:
    sys.path.insert(0, str(THIS_DIR))

from modal_training_config import DatasetBuildConfig, LocalPaths, ModalConfig, SourcePaths, TrainConfig


MODAL_CFG = ModalConfig()
TRAIN_CFG = TrainConfig()

app = modal.App(name=MODAL_CFG.app_name)

image = (
    modal.Image.debian_slim(python_version=MODAL_CFG.image_python)
    .apt_install("git", "curl")
    .uv_pip_install(
        "modal==1.4.0",
        "torch==2.7.1",
        "tiktoken==0.9.0",
        "numpy",
        "requests",
        "tqdm",
        "tensorflow-cpu==2.19.0",
        "datasets==4.4.1",
        "huggingface_hub==1.1.6",
    )
    .run_commands(
        "curl -sSL https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch05/01_main-chapter-code/gpt_download.py -o /usr/local/lib/python3.11/site-packages/gpt_download.py"
    )
    .add_local_dir(
        str(Path(__file__).resolve().parent),
        remote_path=MODAL_CFG.remote_root,
        ignore=[
            "__pycache__",
            ".pytest_cache",
            "artifacts",
            "outputs",
            "hf_cache",
            ".venv",
        ],
    )
)

data_volume = modal.Volume.from_name(MODAL_CFG.data_volume_name, create_if_missing=True)
outputs_volume = modal.Volume.from_name(MODAL_CFG.outputs_volume_name, create_if_missing=True)
cache_volume = modal.Volume.from_name(MODAL_CFG.cache_volume_name, create_if_missing=True)


@app.function(
    image=image,
    timeout=60 * 60 * 3,
    volumes={
        MODAL_CFG.remote_data_dir: data_volume,
        MODAL_CFG.remote_cache_dir: cache_volume,
    },
)
def prepare_remote(
    prepared_subdir: str,
    include_grammar: bool = False,
    include_gyafc: bool = False,
    include_parasci: bool = False,
    include_human_ai: bool = True,
    include_jfleg_train_support: bool = False,
    max_examples_per_source: int | None = 12000,
    pilot_examples: int = 8000,
    human_ai_min_words: int = DatasetBuildConfig().human_ai_min_words,
    human_ai_max_words: int = DatasetBuildConfig().human_ai_max_words,
    human_ai_min_word_overlap: float = DatasetBuildConfig().human_ai_min_word_overlap,
    human_ai_min_char_similarity: float = DatasetBuildConfig().human_ai_min_char_similarity,
    human_ai_min_length_ratio: float = DatasetBuildConfig().human_ai_min_length_ratio,
    human_ai_max_length_ratio: float = DatasetBuildConfig().human_ai_max_length_ratio,
    grammar_json_subpath: str | None = None,
    gyafc_subdir: str | None = None,
    parasci_subdir: str | None = None,
) -> dict[str, object]:
    from prepare_datasets import build_dataset, set_seed

    prepared_dir = Path(MODAL_CFG.remote_data_dir) / prepared_subdir
    cfg = DatasetBuildConfig(
        include_grammar=include_grammar,
        include_gyafc=include_gyafc,
        include_parasci=include_parasci,
        include_human_ai=include_human_ai,
        include_jfleg_train_support=include_jfleg_train_support,
        max_examples_per_source=max_examples_per_source,
        pilot_examples=pilot_examples,
        human_ai_min_words=human_ai_min_words,
        human_ai_max_words=human_ai_max_words,
        human_ai_min_word_overlap=human_ai_min_word_overlap,
        human_ai_min_char_similarity=human_ai_min_char_similarity,
        human_ai_min_length_ratio=human_ai_min_length_ratio,
        human_ai_max_length_ratio=human_ai_max_length_ratio,
    )
    paths = LocalPaths(prepared_dir=prepared_dir)
    sources = SourcePaths(
        grammar_json=(Path(MODAL_CFG.remote_data_dir) / grammar_json_subpath) if grammar_json_subpath else None,
        gyafc_dir=(Path(MODAL_CFG.remote_data_dir) / gyafc_subdir) if gyafc_subdir else None,
        parasci_dir=(Path(MODAL_CFG.remote_data_dir) / parasci_subdir) if parasci_subdir else None,
    )
    set_seed(cfg.seed)
    summary = build_dataset(cfg, paths, sources)
    data_volume.commit()
    return summary


@app.function(
    image=image,
    gpu=MODAL_CFG.gpu,
    timeout=MODAL_CFG.timeout_seconds,
    volumes={
        MODAL_CFG.remote_data_dir: data_volume,
        MODAL_CFG.remote_outputs_dir: outputs_volume,
        MODAL_CFG.remote_cache_dir: cache_volume,
    },
)
def train_remote(
    prepared_subdir: str = MODAL_CFG.prepared_subdir,
    run_name: str = "full-ft",
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    model_init_subdir: str | None = None,
) -> dict[str, object]:
    from train_scratch import train_model

    prepared_dir = Path(MODAL_CFG.remote_data_dir) / prepared_subdir
    output_dir = Path(MODAL_CFG.remote_outputs_dir) / run_name
    cache_dir = Path(MODAL_CFG.remote_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_init_ref = None
    if model_init_subdir:
        model_init_ref = Path(MODAL_CFG.remote_outputs_dir) / model_init_subdir

    cfg = TrainConfig(
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )
    summary = train_model(
        prepared_dir=prepared_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
        cfg=cfg,
        model_init_ref=model_init_ref,
        outputs_volume=outputs_volume,
    )
    return summary


@app.function(image=image, gpu=MODAL_CFG.gpu, timeout=60 * 10)
def check_gpu() -> str:
    return subprocess.check_output(["nvidia-smi"], text=True)


@app.function(
    image=image,
    gpu=MODAL_CFG.gpu,
    timeout=60 * 60 * 3,
    volumes={
        MODAL_CFG.remote_data_dir: data_volume,
        MODAL_CFG.remote_outputs_dir: outputs_volume,
        MODAL_CFG.remote_cache_dir: cache_volume,
    },
)
def evaluate_remote(
    prepared_subdir: str = MODAL_CFG.prepared_subdir,
    run_name: str = "pilot",
    split: str = "test",
    loss_examples: int = 300,
    generation_examples: int = 50,
    qualitative_examples: int = 8,
    model_ref_subdir: str | None = None,
    max_new_tokens: int = 128,
    seed: int = 123,
    temperature: float = 0.75,
    top_k: int = 50,
    top_p: float = 0.90,
    repetition_penalty: float = 1.05,
) -> dict[str, object]:
    from evaluate_gpt2 import EvalConfig, run_comparison

    prepared_dir = Path(MODAL_CFG.remote_data_dir) / prepared_subdir
    model_dir_name = model_ref_subdir if model_ref_subdir else run_name
    finetuned_model_dir = Path(MODAL_CFG.remote_outputs_dir) / model_dir_name / "final_model"
    output_path = Path(MODAL_CFG.remote_outputs_dir) / run_name / "evaluation_report.json"
    cache_dir = Path(MODAL_CFG.remote_cache_dir)

    eval_cfg = EvalConfig(
        split=split,
        loss_examples=loss_examples,
        generation_examples=generation_examples,
        qualitative_examples=qualitative_examples,
        max_new_tokens=max_new_tokens,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    report = run_comparison(
        prepared_dir=prepared_dir,
        finetuned_model_dir=finetuned_model_dir,
        cache_dir=cache_dir,
        output_path=output_path,
        train_cfg=TRAIN_CFG,
        eval_cfg=eval_cfg,
    )
    outputs_volume.commit()
    return report


@app.function(
    image=image,
    gpu=MODAL_CFG.gpu,
    timeout=60 * 60 * 3,
    volumes={
        MODAL_CFG.remote_data_dir: data_volume,
        MODAL_CFG.remote_outputs_dir: outputs_volume,
        MODAL_CFG.remote_cache_dir: cache_volume,
    },
)
def evaluate_suite_remote(
    suite_name: str = "full_model_comparison",
    loss_examples: int = 300,
    generation_examples: int = 32,
    qualitative_examples: int = 8,
    humanizer_prepared_subdir: str = "prepared_human_ai_stage2",
    include_repaired_stage2: bool = False,
) -> dict[str, object]:
    from evaluate_gpt2 import EvalConfig
    from evaluate_suite import DatasetSpec, ModelSpec, run_evaluation_suite

    cache_dir = Path(MODAL_CFG.remote_cache_dir)
    output_path = Path(MODAL_CFG.remote_outputs_dir) / "evaluation_suites" / f"{suite_name}.json"
    model_specs = [
        ModelSpec(key="base_gpt2", label="Base GPT-2", model_ref=TRAIN_CFG.model_name),
        ModelSpec(
            key="stage1_grammar",
            label="Stage 1 Grammar",
            model_ref=str(Path(MODAL_CFG.remote_outputs_dir) / "pilot" / "final_model"),
        ),
        ModelSpec(
            key="stage2_grammar_humanizer",
            label="Stage 2 Grammar + Humanizer",
            model_ref=str(Path(MODAL_CFG.remote_outputs_dir) / "pilot_stage2_human_ai" / "final_model"),
        ),
        ModelSpec(
            key="stage3_full",
            label="Stage 3 Grammar + Humanizer + Academic",
            model_ref=str(Path(MODAL_CFG.remote_outputs_dir) / "pilot_stage3_academic_from_repaired" / "final_model"),
        ),
    ]
    if include_repaired_stage2:
        model_specs.insert(
            3,
            ModelSpec(
                key="stage2_humanizer_repaired",
                label="Stage 2 Humanizer Repaired",
                model_ref=str(Path(MODAL_CFG.remote_outputs_dir) / "pilot_stage2_human_ai_repaired" / "final_model"),
            ),
        )
    dataset_specs = [
        DatasetSpec(
            key="grammar",
            label="Grammar Held-out",
            prepared_dir=str(Path(MODAL_CFG.remote_data_dir) / "prepared"),
        ),
        DatasetSpec(
            key="human_ai",
            label="Humanizer Held-out",
            prepared_dir=str(Path(MODAL_CFG.remote_data_dir) / humanizer_prepared_subdir),
        ),
        DatasetSpec(
            key="academic",
            label="Academic Held-out",
            prepared_dir=str(Path(MODAL_CFG.remote_data_dir) / "prepared_stage3_remaining_academic"),
        ),
    ]
    eval_cfg = EvalConfig(
        split="test",
        loss_examples=loss_examples,
        generation_examples=generation_examples,
        qualitative_examples=qualitative_examples,
    )
    report = run_evaluation_suite(
        suite_name=suite_name,
        model_specs=model_specs,
        dataset_specs=dataset_specs,
        cache_dir=cache_dir,
        output_path=output_path,
        train_cfg=TRAIN_CFG,
        eval_cfg=eval_cfg,
    )
    outputs_volume.commit()
    return report


@app.local_entrypoint()
def main(
    prepared_subdir: str = MODAL_CFG.prepared_subdir,
    run_name: str = "full-ft",
    max_train_samples: int = 0,
    max_eval_samples: int = 0,
    model_init_subdir: str = "",
) -> None:
    train_samples = max_train_samples or None
    eval_samples = max_eval_samples or None
    init_subdir = model_init_subdir or None
    summary = train_remote.remote(
        prepared_subdir=prepared_subdir,
        run_name=run_name,
        max_train_samples=train_samples,
        max_eval_samples=eval_samples,
        model_init_subdir=init_subdir,
    )
    print(json.dumps(summary, indent=2))


@app.local_entrypoint(name="prepare")
def prepare_main(
    prepared_subdir: str,
    include_grammar: bool = False,
    include_gyafc: bool = False,
    include_parasci: bool = False,
    include_human_ai: bool = True,
    include_jfleg_train_support: bool = False,
    max_examples_per_source: int = 12000,
    pilot_examples: int = 8000,
    human_ai_min_words: int = DatasetBuildConfig().human_ai_min_words,
    human_ai_max_words: int = DatasetBuildConfig().human_ai_max_words,
    human_ai_min_word_overlap: float = DatasetBuildConfig().human_ai_min_word_overlap,
    human_ai_min_char_similarity: float = DatasetBuildConfig().human_ai_min_char_similarity,
    human_ai_min_length_ratio: float = DatasetBuildConfig().human_ai_min_length_ratio,
    human_ai_max_length_ratio: float = DatasetBuildConfig().human_ai_max_length_ratio,
    grammar_json_subpath: str = "",
    gyafc_subdir: str = "",
    parasci_subdir: str = "",
) -> None:
    summary = prepare_remote.remote(
        prepared_subdir=prepared_subdir,
        include_grammar=include_grammar,
        include_gyafc=include_gyafc,
        include_parasci=include_parasci,
        include_human_ai=include_human_ai,
        include_jfleg_train_support=include_jfleg_train_support,
        max_examples_per_source=max_examples_per_source or None,
        pilot_examples=pilot_examples,
        human_ai_min_words=human_ai_min_words,
        human_ai_max_words=human_ai_max_words,
        human_ai_min_word_overlap=human_ai_min_word_overlap,
        human_ai_min_char_similarity=human_ai_min_char_similarity,
        human_ai_min_length_ratio=human_ai_min_length_ratio,
        human_ai_max_length_ratio=human_ai_max_length_ratio,
        grammar_json_subpath=grammar_json_subpath or None,
        gyafc_subdir=gyafc_subdir or None,
        parasci_subdir=parasci_subdir or None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


@app.local_entrypoint(name="prepare-humanizer-repair")
def prepare_humanizer_repair_main(
    prepared_subdir: str = "prepared_humanizer_repaired",
    grammar_json_subpath: str = "grammar-data.json",
    max_examples_per_source: int = 6000,
    pilot_examples: int = 6000,
    human_ai_min_words: int = DatasetBuildConfig().human_ai_min_words,
    human_ai_max_words: int = DatasetBuildConfig().human_ai_max_words,
    human_ai_min_word_overlap: float = DatasetBuildConfig().human_ai_min_word_overlap,
    human_ai_min_char_similarity: float = DatasetBuildConfig().human_ai_min_char_similarity,
    human_ai_min_length_ratio: float = DatasetBuildConfig().human_ai_min_length_ratio,
    human_ai_max_length_ratio: float = DatasetBuildConfig().human_ai_max_length_ratio,
) -> None:
    summary = prepare_remote.remote(
        prepared_subdir=prepared_subdir,
        include_grammar=True,
        include_gyafc=False,
        include_parasci=False,
        include_human_ai=True,
        include_jfleg_train_support=False,
        max_examples_per_source=max_examples_per_source or None,
        pilot_examples=pilot_examples,
        human_ai_min_words=human_ai_min_words,
        human_ai_max_words=human_ai_max_words,
        human_ai_min_word_overlap=human_ai_min_word_overlap,
        human_ai_min_char_similarity=human_ai_min_char_similarity,
        human_ai_min_length_ratio=human_ai_min_length_ratio,
        human_ai_max_length_ratio=human_ai_max_length_ratio,
        grammar_json_subpath=grammar_json_subpath or None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


@app.local_entrypoint(name="train-humanizer-repair")
def train_humanizer_repair_main(
    prepared_subdir: str = "prepared_humanizer_repaired",
    run_name: str = "pilot_stage2_human_ai_repaired",
    max_train_samples: int = 0,
    max_eval_samples: int = 0,
    model_init_subdir: str = "pilot/final_model",
) -> None:
    summary = train_remote.remote(
        prepared_subdir=prepared_subdir,
        run_name=run_name,
        max_train_samples=max_train_samples or None,
        max_eval_samples=max_eval_samples or None,
        model_init_subdir=model_init_subdir or "pilot/final_model",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


@app.local_entrypoint(name="evaluate")
def evaluate_main(
    prepared_subdir: str = MODAL_CFG.prepared_subdir,
    run_name: str = "pilot",
    split: str = "test",
    loss_examples: int = 300,
    generation_examples: int = 50,
    qualitative_examples: int = 8,
    model_ref_subdir: str = "",
    max_new_tokens: int = 128,
    seed: int = 123,
    temperature: float = 0.75,
    top_k: int = 50,
    top_p: float = 0.90,
    repetition_penalty: float = 1.05,
) -> None:
    ref = model_ref_subdir or None
    report = evaluate_remote.remote(
        prepared_subdir=prepared_subdir,
        run_name=run_name,
        split=split,
        loss_examples=loss_examples,
        generation_examples=generation_examples,
        qualitative_examples=qualitative_examples,
        model_ref_subdir=ref,
        max_new_tokens=max_new_tokens,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


@app.local_entrypoint(name="evaluate-suite")
def evaluate_suite_main(
    suite_name: str = "full_model_comparison",
    loss_examples: int = 300,
    generation_examples: int = 32,
    qualitative_examples: int = 8,
    humanizer_prepared_subdir: str = "prepared_human_ai_stage2",
    include_repaired_stage2: bool = False,
) -> None:
    report = evaluate_suite_remote.remote(
        suite_name=suite_name,
        loss_examples=loss_examples,
        generation_examples=generation_examples,
        qualitative_examples=qualitative_examples,
        humanizer_prepared_subdir=humanizer_prepared_subdir,
        include_repaired_stage2=include_repaired_stage2,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


# ── Scratch cross-evaluation: 3 models × 3 datasets ──────────────────────
SCRATCH_CROSS_EVALS = [
    # (run_name,          prepared_subdir,                model_ref_subdir)
    ("scratch_stage1_on_grammar",   "prepared",                     "scratch_stage1_grammar"),
    ("scratch_stage1_on_humanizer", "prepared_humanizer_repaired",  "scratch_stage1_grammar"),
    ("scratch_stage1_on_academic",  "prepared_stage3_remaining_academic", "scratch_stage1_grammar"),
    ("scratch_stage2_on_grammar",   "prepared",                     "scratch_stage2_humanizer_repaired"),
    ("scratch_stage2_on_humanizer", "prepared_humanizer_repaired",  "scratch_stage2_humanizer_repaired"),
    ("scratch_stage2_on_academic",  "prepared_stage3_remaining_academic", "scratch_stage2_humanizer_repaired"),
    ("scratch_stage3_on_grammar",   "prepared",                     "scratch_stage3_academic"),
    ("scratch_stage3_on_humanizer", "prepared_humanizer_repaired",  "scratch_stage3_academic"),
    ("scratch_stage3_on_academic",  "prepared_stage3_remaining_academic", "scratch_stage3_academic"),
]


@app.function(
    image=image,
    gpu=MODAL_CFG.gpu,
    timeout=60 * 60 * 4,
    volumes={
        MODAL_CFG.remote_data_dir: data_volume,
        MODAL_CFG.remote_outputs_dir: outputs_volume,
        MODAL_CFG.remote_cache_dir: cache_volume,
    },
)
def evaluate_scratch_one(
    run_name: str,
    prepared_subdir: str,
    model_ref_subdir: str,
    loss_examples: int = 300,
    generation_examples: int = 50,
    qualitative_examples: int = 8,
    max_new_tokens: int = 128,
    seed: int = 123,
    temperature: float = 0.75,
    top_k: int = 50,
    top_p: float = 0.90,
    repetition_penalty: float = 1.05,
    include_base: bool = True,
) -> dict[str, object]:
    """Run a single scratch-model evaluation with configurable sampling."""
    from evaluate_gpt2 import EvalConfig, run_comparison

    prepared_dir = Path(MODAL_CFG.remote_data_dir) / prepared_subdir
    finetuned_model_dir = Path(MODAL_CFG.remote_outputs_dir) / model_ref_subdir / "final_model"
    output_path = Path(MODAL_CFG.remote_outputs_dir) / run_name / "evaluation_report.json"
    cache_dir = Path(MODAL_CFG.remote_cache_dir)

    eval_cfg = EvalConfig(
        split="test",
        loss_examples=loss_examples,
        generation_examples=generation_examples,
        qualitative_examples=qualitative_examples,
        max_new_tokens=max_new_tokens,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    report = run_comparison(
        prepared_dir=prepared_dir,
        finetuned_model_dir=finetuned_model_dir,
        cache_dir=cache_dir,
        output_path=output_path,
        train_cfg=TRAIN_CFG,
        eval_cfg=eval_cfg,
        include_base=include_base,
    )
    outputs_volume.commit()
    return report


@app.local_entrypoint(name="evaluate-scratch")
def evaluate_scratch_main(
    loss_examples: int = 300,
    generation_examples: int = 50,
    qualitative_examples: int = 50,
    temperature: float = 0.75,
    top_k: int = 50,
    top_p: float = 0.90,
    repetition_penalty: float = 1.05,
    max_new_tokens: int = 128,
    seed: int = 123,
) -> None:
    """Run 9 scratch cross-evaluations sequentially."""
    import time

    def _fmt(val: object) -> str:
        return f"{val:.4f}" if isinstance(val, (int, float)) else "?"

    results = {}
    for run_name, prepared_subdir, model_ref_subdir in SCRATCH_CROSS_EVALS:
        print(f"\n{'='*60}")
        print(f"[{run_name}] dataset={prepared_subdir} model={model_ref_subdir}")
        print(f"{'='*60}")
        sys.stdout.flush()
        try:
            report = evaluate_scratch_one.remote(
                run_name=run_name,
                prepared_subdir=prepared_subdir,
                model_ref_subdir=model_ref_subdir,
                loss_examples=loss_examples,
                generation_examples=generation_examples,
                qualitative_examples=qualitative_examples,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
            finetuned_loss_val = (report.get("finetuned_loss") or {}).get("loss")
            finetuned_f1_val = report.get("generation_summary", {}).get("finetuned_avg_token_f1")

            results[run_name] = {
                "status": "ok",
                "base_loss": report.get("base_loss"),
                "finetuned_loss": report.get("finetuned_loss"),
                "gen_count": report.get("generation_summary", {}).get("count"),
                "finetuned_avg_token_f1": finetuned_f1_val,
            }
            print(f"  ✓ loss={_fmt(finetuned_loss_val)}  token_f1={_fmt(finetuned_f1_val)}")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results[run_name] = {"status": "failed", "error": str(e)}
        time.sleep(5)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for run_name, r in results.items():
        status = r["status"]
        if status == "ok":
            loss_val = (r.get("finetuned_loss") or {}).get("loss")
            f1_val = r.get("finetuned_avg_token_f1")
            print(f"  ✓ {run_name:40s} loss={_fmt(loss_val)}  f1={_fmt(f1_val)}")
        else:
            print(f"  ✗ {run_name:40s} FAILED: {r.get('error','?')}")

    print("\nDone.")


@app.local_entrypoint(name="smoke-evaluate-scratch")
def smoke_evaluate_scratch_main(
    run_name: str = "scratch_smoke_stage1_on_grammar",
    prepared_subdir: str = "prepared",
    model_ref_subdir: str = "scratch_stage1_grammar",
    loss_examples: int = 8,
    generation_examples: int = 2,
    qualitative_examples: int = 1,
    max_new_tokens: int = 32,
    seed: int = 123,
    temperature: float = 0.75,
    top_k: int = 50,
    top_p: float = 0.90,
    repetition_penalty: float = 1.05,
    include_base: bool = False,
) -> None:
    """Cheap end-to-end smoke check (1 model × 1 dataset) on Modal.

    Writes `/vol/outputs/<run_name>/evaluation_report.json` to the outputs Volume.
    """
    report = evaluate_scratch_one.remote(
        run_name=run_name,
        prepared_subdir=prepared_subdir,
        model_ref_subdir=model_ref_subdir,
        loss_examples=loss_examples,
        generation_examples=generation_examples,
        qualitative_examples=qualitative_examples,
        max_new_tokens=max_new_tokens,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        include_base=include_base,
    )
    out_path = f"{MODAL_CFG.remote_outputs_dir}/{run_name}/evaluation_report.json"
    print(json.dumps({"run_name": run_name, "saved_to": out_path}, indent=2))
    print(
        json.dumps(
            {
                "finetuned_loss": (report.get("finetuned_loss") or {}).get("loss"),
                "finetuned_ppl": (report.get("finetuned_loss") or {}).get("perplexity"),
                "finetuned_token_f1": report.get("generation_summary", {}).get("finetuned_avg_token_f1"),
                "count": report.get("generation_summary", {}).get("count"),
            },
            indent=2,
        )
    )
