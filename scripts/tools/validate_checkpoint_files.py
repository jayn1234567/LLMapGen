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
    safetensor_paths = [path for path in paths if path.suffix == ".safetensors"]
    if not safetensor_paths:
        return
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required to validate this checkpoint") from exc

    for path in safetensor_paths:
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
        except Exception as exc:
            raise ValueError(f"Unreadable safetensors file {path}: {exc}") from exc
        if not keys:
            raise ValueError(f"Safetensors file contains no tensors: {path}")


def _validate_expected_kind(checkpoint_dir: Path, expected_kind: str) -> str:
    expected_kind = str(expected_kind or "auto").strip().lower()
    if expected_kind not in {"auto", "full", "lora"}:
        raise ValueError(f"Unsupported expected checkpoint kind: {expected_kind!r}")

    adapter_path = next(
        (
            checkpoint_dir / name
            for name in ("adapter_model.safetensors", "adapter_model.bin")
            if (checkpoint_dir / name).is_file()
        ),
        None,
    )
    full_path = next(
        (
            checkpoint_dir / name
            for name in (
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            )
            if (checkpoint_dir / name).is_file()
        ),
        None,
    )

    if expected_kind == "lora":
        required = (
            checkpoint_dir / "adapter_config.json",
            checkpoint_dir / "non_lora_trainables.bin",
            checkpoint_dir / "config.json",
        )
        missing = [path for path in required if not path.is_file()]
        if adapter_path is None:
            missing.append(checkpoint_dir / "adapter_model.safetensors|adapter_model.bin")
        if missing:
            raise FileNotFoundError(
                "LoRA checkpoint is missing required files: "
                + ", ".join(str(path) for path in missing)
            )
        empty = [path for path in (*required, adapter_path) if path.stat().st_size <= 0]
        if empty:
            raise ValueError(
                "LoRA checkpoint contains empty required files: "
                + ", ".join(str(path) for path in empty)
            )
        adapter_config = json.loads(required[0].read_text(encoding="utf-8"))
        if not str(adapter_config.get("base_model_name_or_path") or "").strip():
            raise ValueError(
                f"LoRA adapter config has no base_model_name_or_path: {required[0]}"
            )
        return "lora"

    if expected_kind == "full" and full_path is None:
        raise FileNotFoundError(
            f"Full checkpoint weights were required but not found below: {checkpoint_dir}"
        )
    if expected_kind == "full":
        return "full"
    if adapter_path is not None:
        return "lora"
    if full_path is not None:
        return "full"
    return "unknown"


def validate_checkpoint(
    checkpoint_dir: Path,
    expected_kind: str = "auto",
) -> dict[str, object]:
    checkpoint_dir = checkpoint_dir.resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_kind = _validate_expected_kind(checkpoint_dir, expected_kind)
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
        "checkpoint_kind": checkpoint_kind,
        "expected_kind": str(expected_kind or "auto").strip().lower(),
        "num_weight_files": len(files),
        "total_weight_bytes": sum(path.stat().st_size for path in files),
        "weight_files": [str(path) for path in files],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-kind",
        choices=("auto", "full", "lora"),
        default="auto",
        help="Require a full-model or LoRA checkpoint layout instead of auto-detecting it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_checkpoint(args.checkpoint_dir, expected_kind=args.expected_kind)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
