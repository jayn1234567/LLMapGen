#!/usr/bin/env python3
"""Validate that checkpoint weight files are present and structurally readable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _index_shards(index_path: Path) -> list[Path]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Checkpoint index has no non-empty weight_map: {index_path}")
    return sorted({index_path.parent / str(name) for name in weight_map.values()})


def _weight_files(checkpoint_dir: Path) -> list[Path]:
    files: set[Path] = set()
    for index_name in (
        "model.safetensors.index.json",
        "adapter_model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "adapter_model.bin.index.json",
    ):
        index_path = checkpoint_dir / index_name
        if index_path.is_file():
            files.update(_index_shards(index_path))

    for direct_name in (
        "model.safetensors",
        "adapter_model.safetensors",
        "pytorch_model.bin",
        "adapter_model.bin",
    ):
        direct_path = checkpoint_dir / direct_name
        if direct_path.is_file():
            files.add(direct_path)

    if not files:
        for pattern in (
            "model-*-of-*.safetensors",
            "adapter_model-*-of-*.safetensors",
            "pytorch_model-*-of-*.bin",
            "adapter_model-*-of-*.bin",
        ):
            files.update(checkpoint_dir.glob(pattern))
    return sorted(files)


def _validate_safetensors(paths: Iterable[Path]) -> None:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required to validate this checkpoint") from exc

    for path in paths:
        if path.suffix != ".safetensors":
            continue
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
        except Exception as exc:
            raise ValueError(f"Unreadable safetensors file {path}: {exc}") from exc
        if not keys:
            raise ValueError(f"Safetensors file contains no tensors: {path}")


def validate_checkpoint(checkpoint_dir: Path) -> dict[str, object]:
    checkpoint_dir = checkpoint_dir.resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    files = _weight_files(checkpoint_dir)
    if not files:
        raise FileNotFoundError(f"No supported checkpoint weights found below: {checkpoint_dir}")

    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Checkpoint index references missing files: " + ", ".join(str(path) for path in missing)
        )
    empty = [path for path in files if path.stat().st_size <= 0]
    if empty:
        raise ValueError("Checkpoint contains empty files: " + ", ".join(str(path) for path in empty))

    _validate_safetensors(files)
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "num_weight_files": len(files),
        "total_weight_bytes": sum(path.stat().st_size for path in files),
        "weight_files": [str(path) for path in files],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_checkpoint(args.checkpoint_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
