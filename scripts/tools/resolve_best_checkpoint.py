#!/usr/bin/env python3
"""Resolve the latest successful rotating best checkpoint directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def has_checkpoint_weights(path: Path) -> bool:
    if any(
        (path / name).is_file()
        for name in (
            "model.safetensors",
            "pytorch_model.bin",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
            "adapter_model.safetensors.index.json",
            "adapter_model.bin.index.json",
        )
    ):
        return True
    for pattern in (
        "model-*-of-*.safetensors",
        "pytorch_model-*-of-*.bin",
        "adapter_model-*-of-*.safetensors",
        "adapter_model-*-of-*.bin",
    ):
        if next(path.glob(pattern), None) is not None:
            return True
    return False


def step_from_name(name: str) -> int:
    marker = "_step-"
    if marker not in name:
        return -1
    tail = name.split(marker, 1)[1]
    digits = []
    for char in tail:
        if not char.isdigit():
            break
        digits.append(char)
    return int("".join(digits)) if digits else -1


def load_metadata(path: Path) -> dict:
    for name in ("best_infer_index.json", "best_eval_loss.json", "best_train_loss.json", "best_reward.json"):
        metadata_path = path / name
        if metadata_path.is_file():
            try:
                return json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def candidate_dirs(output_dir: Path, best_name: str):
    root = output_dir / f"{best_name}_candidates"
    if not root.is_dir():
        return []
    candidates = []
    for path in root.iterdir():
        if not path.is_dir() or not (path / "_SUCCESS").is_file():
            continue
        metadata = load_metadata(path)
        step = -1
        for key in ("best_infer_index_step", "best_eval_loss_step", "best_train_loss_step", "best_reward_step", "global_step"):
            if key in metadata:
                try:
                    step = int(metadata[key])
                    break
                except (TypeError, ValueError):
                    pass
        if step < 0:
            step = step_from_name(path.name)
        candidates.append((step, path))
    return sorted(candidates, key=lambda item: (item[0], str(item[1])))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--best-name",
        action="append",
        default=None,
        help="Best checkpoint logical name to try, e.g. eval_best or best. Can be repeated.",
    )
    parser.add_argument("--allow-direct", action="store_true", help="Allow output_dir/best_name as a fallback.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    names = args.best_name or ["eval_best", "best"]
    direct_fallbacks = []
    all_candidates = []

    if args.allow_direct and output_dir.is_dir() and has_checkpoint_weights(output_dir):
        direct_fallbacks.append(output_dir)

    for name in names:
        all_candidates.extend(candidate_dirs(output_dir, name))
        direct = output_dir / name
        if args.allow_direct and direct.is_dir():
            direct_fallbacks.append(direct)

    if all_candidates:
        print(all_candidates[-1][1])
        return 0
    if direct_fallbacks:
        print(direct_fallbacks[0])
        return 0

    tried = ", ".join([f"{name}_candidates" for name in names])
    print(f"No successful best checkpoint found under {output_dir}; tried {tried}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
