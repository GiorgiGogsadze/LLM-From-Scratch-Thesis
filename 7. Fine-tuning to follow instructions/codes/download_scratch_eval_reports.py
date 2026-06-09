#!/usr/bin/env python3
"""Download Modal scratch cross-eval JSON reports into artifacts/evals_scratch_new/.

Requires Modal CLI auth (`modal token set`). Run after ``modal run modal_app.py::evaluate_scratch_main``.
"""
from __future__ import annotations

import io
from pathlib import Path

import modal

from modal_training_config import ModalConfig

MODAL_CFG = ModalConfig()

# Must match modal_app.SCRATCH_CROSS_EVALS run_name prefixes
RUN_NAMES = [
    "scratch_stage1_on_grammar",
    "scratch_stage1_on_humanizer",
    "scratch_stage1_on_academic",
    "scratch_stage2_on_grammar",
    "scratch_stage2_on_humanizer",
    "scratch_stage2_on_academic",
    "scratch_stage3_on_grammar",
    "scratch_stage3_on_humanizer",
    "scratch_stage3_on_academic",
]


def _write_remote_json(vol: modal.Volume, remote_rel: str, dest: Path) -> None:
    buf = io.BytesIO()
    for chunk in vol.read_file(remote_rel):
        buf.write(chunk)
    raw = buf.getvalue()
    if not raw:
        raise RuntimeError(f"Empty read: {remote_rel}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "artifacts" / "evals_scratch_new"
    vol = modal.Volume.from_name(MODAL_CFG.outputs_volume_name)

    for run_name in RUN_NAMES:
        remote_rel = f"{run_name}/evaluation_report.json"
        dest = out_dir / f"{run_name}_eval_report.json"
        print(f"{remote_rel} -> {dest}")
        _write_remote_json(vol, remote_rel, dest)

    print(f"Done. Wrote {len(RUN_NAMES)} reports under {out_dir}")


if __name__ == "__main__":
    main()
