from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import tarfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from modal_training_config import DatasetBuildConfig, LocalPaths, SourcePaths


def set_seed(seed: int) -> None:
    random.seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_instruction(
    task_mode: str,
    cfg: DatasetBuildConfig,
    source_name: str,
    source_row_id: str,
) -> str:
    templates = cfg.instruction_templates[task_mode]
    stable_rng = random.Random(f"{cfg.seed}:{source_name}:{source_row_id}:{task_mode}")
    return stable_rng.choice(templates)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def lexical_overlap_ratio(source_text: str, target_text: str) -> float:
    source_words = set(word_tokens(source_text))
    target_words = set(word_tokens(target_text))
    if not source_words or not target_words:
        return 0.0
    return len(source_words & target_words) / len(source_words | target_words)


def char_similarity_ratio(source_text: str, target_text: str) -> float:
    return difflib.SequenceMatcher(None, source_text, target_text).ratio()


def passes_human_ai_filters(ai_text: str, human_text: str, cfg: DatasetBuildConfig) -> tuple[bool, str]:
    if not (cfg.human_ai_min_chars <= len(ai_text) <= cfg.human_ai_max_chars):
        return False, "char_length"
    if any(token in ai_text.lower() for token in ("### instruction", "prompt:", "undetectable ai")):
        return False, "blacklist"

    ai_words = word_tokens(ai_text)
    human_words = word_tokens(human_text)
    if not ai_words or not human_words:
        return False, "empty_words"
    if not (cfg.human_ai_min_words <= len(ai_words) <= cfg.human_ai_max_words):
        return False, "ai_word_length"
    if not (cfg.human_ai_min_words <= len(human_words) <= cfg.human_ai_max_words):
        return False, "human_word_length"

    length_ratio = len(human_words) / max(len(ai_words), 1)
    if not (cfg.human_ai_min_length_ratio <= length_ratio <= cfg.human_ai_max_length_ratio):
        return False, "length_ratio"
    if lexical_overlap_ratio(ai_text, human_text) < cfg.human_ai_min_word_overlap:
        return False, "word_overlap"
    if char_similarity_ratio(ai_text, human_text) < cfg.human_ai_min_char_similarity:
        return False, "char_similarity"
    return True, "kept"


def make_entry(
    *,
    source: str,
    source_row_id: str,
    task_mode: str,
    input_text: str,
    output_text: str,
    cfg: DatasetBuildConfig,
) -> dict[str, Any] | None:
    input_text = normalize_space(input_text)
    output_text = normalize_space(output_text)
    if not input_text or not output_text or input_text == output_text:
        return None
    return {
        "source": source,
        "source_row_id": source_row_id,
        "task_mode": task_mode,
        "instruction": choose_instruction(task_mode, cfg, source, source_row_id),
        "input": input_text,
        "output": output_text,
    }


def limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or len(rows) <= limit:
        return rows
    return rows[:limit]


def load_grammar_rows(grammar_json: Path, cfg: DatasetBuildConfig) -> list[dict[str, Any]]:
    data = load_json(grammar_json)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(data):
        entry = make_entry(
            source="grammar_local",
            source_row_id=str(idx),
            task_mode="grammar",
            input_text=str(row.get("input", "")),
            output_text=str(row.get("output", "")),
            cfg=cfg,
        )
        if entry is not None:
            rows.append(entry)
    return limit_rows(rows, cfg.max_examples_per_source)


def load_hf_dataset_rows(dataset_name: str, split: str, streaming: bool = False) -> Any:
    from datasets import load_dataset

    return load_dataset(dataset_name, split=split, streaming=streaming)


def load_human_ai_rows(cfg: DatasetBuildConfig) -> tuple[list[dict[str, Any]], dict[str, int]]:
    dataset = load_hf_dataset_rows("dmitva/human_ai_generated_text", split="train", streaming=True)
    rows: list[dict[str, Any]] = []
    stats: dict[str, int] = {
        "seen": 0,
        "kept": 0,
        "blacklist": 0,
        "char_length": 0,
        "empty_words": 0,
        "ai_word_length": 0,
        "human_word_length": 0,
        "length_ratio": 0,
        "word_overlap": 0,
        "char_similarity": 0,
        "null_entry": 0,
    }
    for idx, row in enumerate(dataset):
        stats["seen"] += 1
        ai_text = str(row.get("ai_text", "")).strip()
        human_text = str(row.get("human_text", "")).strip()
        keep, reason = passes_human_ai_filters(ai_text, human_text, cfg)
        if not keep:
            stats[reason] = stats.get(reason, 0) + 1
            continue
        entry = make_entry(
            source="dmitva_human_ai_generated_text",
            source_row_id=str(row.get("id", idx)),
            task_mode="natural",
            input_text=ai_text,
            output_text=human_text,
            cfg=cfg,
        )
        if entry is not None:
            rows.append(entry)
            stats["kept"] += 1
            if cfg.max_examples_per_source is not None and len(rows) >= cfg.max_examples_per_source:
                break
        else:
            stats["null_entry"] += 1
    return rows, stats


def load_jfleg_rows(cfg: DatasetBuildConfig) -> list[dict[str, Any]]:
    dataset = load_hf_dataset_rows("jhu-clsp/jfleg", split="validation")
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        sentence = str(row.get("sentence", "")).strip()
        corrections = row.get("corrections") or []
        if not corrections:
            continue
        entry = make_entry(
            source="jfleg_validation",
            source_row_id=str(idx),
            task_mode="grammar",
            input_text=sentence,
            output_text=str(corrections[0]),
            cfg=cfg,
        )
        if entry is not None:
            rows.append(entry)
    return limit_rows(rows, cfg.max_examples_per_source)


def load_gyafc_rows(gyafc_dir: Path, cfg: DatasetBuildConfig) -> list[dict[str, Any]]:
    informal_files = sorted(gyafc_dir.rglob("*informal*"))
    rows: list[dict[str, Any]] = []
    for informal_path in informal_files:
        formal_name = informal_path.name.replace("informal", "formal")
        formal_path = informal_path.with_name(formal_name)
        if not formal_path.exists():
            continue
        informal_lines = informal_path.read_text(encoding="utf-8").splitlines()
        formal_lines = formal_path.read_text(encoding="utf-8").splitlines()
        for idx, (source_line, target_line) in enumerate(zip(informal_lines, formal_lines, strict=False)):
            entry = make_entry(
                source="gyafc",
                source_row_id=f"{informal_path.stem}:{idx}",
                task_mode="academic",
                input_text=source_line,
                output_text=target_line,
                cfg=cfg,
            )
            if entry is not None:
                rows.append(entry)
    return limit_rows(rows, cfg.max_examples_per_source)


def load_gyafc_hf_rows(cfg: DatasetBuildConfig) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    rows: list[dict[str, Any]] = []
    archive_names = ("gyafc_em.tgz", "gyafc_fr.tgz")
    for archive_name in archive_names:
        archive_path = hf_hub_download(
            repo_id="RUCAIBox/Style-Transfer",
            repo_type="dataset",
            filename=archive_name,
        )
        with tarfile.open(archive_path, "r:gz") as handle:
            members = {member.name: member for member in handle.getmembers() if member.isfile()}
            root_name = archive_name.removesuffix(".tgz")
            src_member = members.get(f"{root_name}/train.src")
            tgt_member = members.get(f"{root_name}/train.tgt")
            if src_member is None or tgt_member is None:
                continue
            src_handle = handle.extractfile(src_member)
            tgt_handle = handle.extractfile(tgt_member)
            if src_handle is None or tgt_handle is None:
                continue
            source_lines = src_handle.read().decode("utf-8").splitlines()
            target_lines = tgt_handle.read().decode("utf-8").splitlines()
            for idx, (source_line, target_line) in enumerate(zip(source_lines, target_lines, strict=False)):
                entry = make_entry(
                    source="gyafc_hf",
                    source_row_id=f"{root_name}:{idx}",
                    task_mode="academic",
                    input_text=source_line,
                    output_text=target_line,
                    cfg=cfg,
                )
                if entry is not None:
                    rows.append(entry)
                    if cfg.max_examples_per_source is not None and len(rows) >= cfg.max_examples_per_source:
                        return rows
    return rows


def parasci_iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_parasci_rows(parasci_dir: Path, cfg: DatasetBuildConfig) -> list[dict[str, Any]]:
    jsonl_files = sorted(parasci_dir.rglob("*.jsonl"))
    json_files = sorted(parasci_dir.rglob("*.json"))
    rows: list[dict[str, Any]] = []
    for jsonl_path in jsonl_files:
        for idx, row in enumerate(parasci_iter_jsonl(jsonl_path)):
            source_text = str(row.get("sentence_1", row.get("src", row.get("source", ""))))
            target_text = str(row.get("sentence_2", row.get("tgt", row.get("target", ""))))
            entry = make_entry(
                source="parasci",
                source_row_id=f"{jsonl_path.stem}:{idx}",
                task_mode="academic",
                input_text=source_text,
                output_text=target_text,
                cfg=cfg,
            )
            if entry is not None:
                rows.append(entry)
    for json_path in json_files:
        payload = load_json(json_path)
        if not isinstance(payload, list):
            continue
        for idx, row in enumerate(payload):
            source_text = str(row.get("sentence_1", row.get("src", row.get("source", ""))))
            target_text = str(row.get("sentence_2", row.get("tgt", row.get("target", ""))))
            entry = make_entry(
                source="parasci",
                source_row_id=f"{json_path.stem}:{idx}",
                task_mode="academic",
                input_text=source_text,
                output_text=target_text,
                cfg=cfg,
            )
            if entry is not None:
                rows.append(entry)
    return limit_rows(rows, cfg.max_examples_per_source)


def load_parasci_hf_rows(cfg: DatasetBuildConfig) -> list[dict[str, Any]]:
    dataset = load_hf_dataset_rows("HHousen/ParaSCI", split="train", streaming=True)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        source_text = str(row.get("sentence1", row.get("sentence_1", row.get("src", row.get("source", "")))))
        target_text = str(row.get("sentence2", row.get("sentence_2", row.get("tgt", row.get("target", "")))))
        entry = make_entry(
            source="parasci_hf",
            source_row_id=str(idx),
            task_mode="academic",
            input_text=source_text,
            output_text=target_text,
            cfg=cfg,
        )
        if entry is not None:
            rows.append(entry)
            if cfg.max_examples_per_source is not None and len(rows) >= cfg.max_examples_per_source:
                break
    return rows


def stratified_split(
    rows: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["source"], row["task_mode"]), []).append(row)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for key, group in grouped.items():
        rng = random.Random(f"{seed}:{key[0]}:{key[1]}")
        shuffled = group[:]
        rng.shuffle(shuffled)
        train_end = int(len(shuffled) * train_ratio)
        val_end = train_end + int(len(shuffled) * val_ratio)
        train_rows.extend(shuffled[:train_end])
        val_rows.extend(shuffled[train_end:val_end])
        test_rows.extend(shuffled[val_end:])

    random.Random(seed).shuffle(train_rows)
    random.Random(seed + 1).shuffle(val_rows)
    random.Random(seed + 2).shuffle(test_rows)
    return train_rows, val_rows, test_rows


def build_pilot(train_rows: list[dict[str, Any]], pilot_examples: int, seed: int) -> list[dict[str, Any]]:
    if len(train_rows) <= pilot_examples:
        return train_rows
    grouped: dict[str, list[dict[str, Any]]] = {"grammar": [], "academic": [], "natural": []}
    for row in train_rows:
        grouped[row["task_mode"]].append(row)
    rng = random.Random(seed)
    pilot: list[dict[str, Any]] = []
    target_per_group = max(pilot_examples // max(len(grouped), 1), 1)
    for group_rows in grouped.values():
        rng.shuffle(group_rows)
        pilot.extend(group_rows[:target_per_group])
    remaining = [row for row in train_rows if row not in pilot]
    rng.shuffle(remaining)
    if len(pilot) < pilot_examples:
        pilot.extend(remaining[: pilot_examples - len(pilot)])
    rng.shuffle(pilot)
    return pilot[:pilot_examples]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, int] = {}
    by_task: dict[str, int] = {}
    for row in rows:
        by_source[row["source"]] = by_source.get(row["source"], 0) + 1
        by_task[row["task_mode"]] = by_task.get(row["task_mode"], 0) + 1
    return {
        "count": len(rows),
        "by_source": dict(sorted(by_source.items())),
        "by_task_mode": dict(sorted(by_task.items())),
    }


def build_dataset(
    cfg: DatasetBuildConfig,
    paths: LocalPaths,
    sources: SourcePaths,
) -> dict[str, Any]:
    ensure_dir(paths.prepared_dir)
    rows: list[dict[str, Any]] = []
    dataset_notes: dict[str, Any] = {}

    if cfg.include_grammar and sources.grammar_json is not None:
        rows.extend(load_grammar_rows(sources.grammar_json, cfg))
    if cfg.include_gyafc:
        if sources.gyafc_dir is not None and sources.gyafc_dir.exists():
            rows.extend(load_gyafc_rows(sources.gyafc_dir, cfg))
        else:
            rows.extend(load_gyafc_hf_rows(cfg))
    if cfg.include_parasci:
        if sources.parasci_dir is not None and sources.parasci_dir.exists():
            rows.extend(load_parasci_rows(sources.parasci_dir, cfg))
        else:
            rows.extend(load_parasci_hf_rows(cfg))
    if cfg.include_human_ai:
        human_ai_rows, human_ai_stats = load_human_ai_rows(cfg)
        rows.extend(human_ai_rows)
        dataset_notes["human_ai_filtering"] = {
            "kept_rows": len(human_ai_rows),
            **human_ai_stats,
        }
    if cfg.include_jfleg_train_support:
        rows.extend(load_jfleg_rows(cfg))

    train_rows, val_rows, test_rows = stratified_split(rows, cfg.train_ratio, cfg.val_ratio, cfg.seed)
    pilot_rows = build_pilot(train_rows, cfg.pilot_examples, cfg.seed)

    write_jsonl(paths.prepared_dir / "train.jsonl", train_rows)
    write_jsonl(paths.prepared_dir / "val.jsonl", val_rows)
    write_jsonl(paths.prepared_dir / "test.jsonl", test_rows)
    write_jsonl(paths.prepared_dir / "pilot_train.jsonl", pilot_rows)

    summary = {
        "config": asdict(cfg),
        "sources": {
            "grammar_json": str(sources.grammar_json) if sources.grammar_json else None,
            "gyafc_dir": str(sources.gyafc_dir) if sources.gyafc_dir else None,
            "parasci_dir": str(sources.parasci_dir) if sources.parasci_dir else None,
        },
        "dataset_notes": dataset_notes,
        "train": summarize(train_rows),
        "val": summarize(val_rows),
        "test": summarize(test_rows),
        "pilot_train": summarize(pilot_rows),
    }
    (paths.prepared_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare mixed Chapter 7 rewrite datasets.")
    parser.add_argument("--grammar-json", type=Path, default=None)
    parser.add_argument("--gyafc-dir", type=Path, default=None)
    parser.add_argument("--parasci-dir", type=Path, default=None)
    parser.add_argument("--prepared-dir", type=Path, default=LocalPaths().prepared_dir)
    parser.add_argument("--pilot-examples", type=int, default=DatasetBuildConfig().pilot_examples)
    parser.add_argument("--max-examples-per-source", type=int, default=None)
    parser.add_argument("--human-ai-min-words", type=int, default=DatasetBuildConfig().human_ai_min_words)
    parser.add_argument("--human-ai-max-words", type=int, default=DatasetBuildConfig().human_ai_max_words)
    parser.add_argument(
        "--human-ai-min-word-overlap",
        type=float,
        default=DatasetBuildConfig().human_ai_min_word_overlap,
    )
    parser.add_argument(
        "--human-ai-min-char-similarity",
        type=float,
        default=DatasetBuildConfig().human_ai_min_char_similarity,
    )
    parser.add_argument(
        "--human-ai-min-length-ratio",
        type=float,
        default=DatasetBuildConfig().human_ai_min_length_ratio,
    )
    parser.add_argument(
        "--human-ai-max-length-ratio",
        type=float,
        default=DatasetBuildConfig().human_ai_max_length_ratio,
    )
    parser.add_argument("--include-jfleg-train-support", action="store_true")
    parser.add_argument("--include-grammar", dest="include_grammar", action="store_true")
    parser.add_argument("--exclude-grammar", dest="include_grammar", action="store_false")
    parser.add_argument("--include-gyafc", dest="include_gyafc", action="store_true")
    parser.add_argument("--exclude-gyafc", dest="include_gyafc", action="store_false")
    parser.add_argument("--include-parasci", dest="include_parasci", action="store_true")
    parser.add_argument("--exclude-parasci", dest="include_parasci", action="store_false")
    parser.add_argument("--include-human-ai", dest="include_human_ai", action="store_true")
    parser.add_argument("--exclude-human-ai", dest="include_human_ai", action="store_false")
    parser.set_defaults(
        include_grammar=DatasetBuildConfig().include_grammar,
        include_gyafc=DatasetBuildConfig().include_gyafc,
        include_parasci=DatasetBuildConfig().include_parasci,
        include_human_ai=DatasetBuildConfig().include_human_ai,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DatasetBuildConfig(
        pilot_examples=args.pilot_examples,
        max_examples_per_source=args.max_examples_per_source,
        include_grammar=args.include_grammar,
        include_gyafc=args.include_gyafc,
        include_parasci=args.include_parasci,
        include_human_ai=args.include_human_ai,
        include_jfleg_train_support=args.include_jfleg_train_support,
        human_ai_min_words=args.human_ai_min_words,
        human_ai_max_words=args.human_ai_max_words,
        human_ai_min_word_overlap=args.human_ai_min_word_overlap,
        human_ai_min_char_similarity=args.human_ai_min_char_similarity,
        human_ai_min_length_ratio=args.human_ai_min_length_ratio,
        human_ai_max_length_ratio=args.human_ai_max_length_ratio,
    )
    paths = LocalPaths(prepared_dir=args.prepared_dir)
    sources = SourcePaths(
        grammar_json=args.grammar_json,
        gyafc_dir=args.gyafc_dir,
        parasci_dir=args.parasci_dir,
    )
    set_seed(cfg.seed)
    summary = build_dataset(cfg, paths, sources)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
