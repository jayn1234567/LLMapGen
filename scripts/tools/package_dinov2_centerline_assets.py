#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


BRIDGE_CANDIDATE_NAMES = (
    "rc_dinov2_centerline_json_modules.pt",
    "rc_dinov2_caption_modules.pt",
    "pytorch_model.bin",
    "model.safetensors",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024 * 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_bridge_state(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"bridge modules path not found: {path}")
    for name in BRIDGE_CANDIDATE_NAMES:
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no supported bridge modules file found under {path}; expected one of {BRIDGE_CANDIDATE_NAMES}"
    )


def copy_or_link(source: Path, target: Path, *, symlink: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if symlink:
        os.symlink(source, target)
    else:
        shutil.copy2(source, target)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package the DINOv2 centerline visual assets needed on DI/NPU: "
            "a segmentation-tuned DINO checkpoint plus Qwen-aligned bridge modules."
        )
    )
    parser.add_argument("--visual-encoder-checkpoint", required=True, help="Segmentation-tuned DINO checkpoint, e.g. best.pt.")
    parser.add_argument(
        "--bridge-modules-state",
        required=True,
        help="Bridge/alignment modules file or directory containing rc_dinov2_caption_modules.pt.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to create the portable asset bundle in.")
    parser.add_argument("--dinov2-model-name-or-path", default="", help="Optional base DINOv2 path to record in manifest.")
    parser.add_argument("--qwen-model-name-or-path", default="", help="Optional Qwen path to record in manifest.")
    parser.add_argument("--vision-train-last-n-layers", type=int, default=2)
    parser.add_argument("--symlink", action="store_true", help="Symlink instead of copying; useful for local validation only.")
    parser.add_argument("--with-sha256", action="store_true", help="Compute SHA256 for copied assets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visual_source = Path(args.visual_encoder_checkpoint).expanduser().resolve()
    if not visual_source.is_file():
        raise FileNotFoundError(f"visual encoder checkpoint not found: {visual_source}")
    bridge_source = resolve_bridge_state(Path(args.bridge_modules_state).expanduser().resolve())

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    visual_target = output_dir / "visual_encoder_checkpoint.pt"
    bridge_target = output_dir / "bridge_modules_state.pt"
    copy_or_link(visual_source, visual_target, symlink=bool(args.symlink))
    copy_or_link(bridge_source, bridge_target, symlink=bool(args.symlink))

    files: Dict[str, Dict[str, Any]] = {
        "visual_encoder_checkpoint": {
            "path": visual_target.name,
            "source_path": str(visual_source),
            "size_bytes": visual_source.stat().st_size,
        },
        "bridge_modules_state": {
            "path": bridge_target.name,
            "source_path": str(bridge_source),
            "size_bytes": bridge_source.stat().st_size,
        },
    }
    if bool(args.with_sha256):
        files["visual_encoder_checkpoint"]["sha256"] = sha256_file(visual_target)
        files["bridge_modules_state"]["sha256"] = sha256_file(bridge_target)

    manifest: Dict[str, Any] = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "asset_type": "llmapgen_dinov2_centerline_visual_bridge",
        "files": files,
        "recommended_training_env": {
            "VISUAL_ENCODER_CHECKPOINT_PATH": "${ASSET_DIR}/visual_encoder_checkpoint.pt",
            "BRIDGE_MODULES_STATE_PATH": "${ASSET_DIR}/bridge_modules_state.pt",
            "DINOV2_MODEL_NAME_OR_PATH": str(args.dinov2_model_name_or_path).strip(),
            "MODEL_NAME_OR_PATH": str(args.qwen_model_name_or_path).strip(),
            "FREEZE_VISION_ENCODER": "true",
            "VISION_TRAIN_LAST_N_LAYERS": str(int(args.vision_train_last_n_layers)),
            "LOCAL_FILES_ONLY": "true",
        },
    }
    manifest_path = output_dir / "asset_manifest.json"
    write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    template = f"""#!/usr/bin/env bash
set -euo pipefail

ASSET_DIR="${{ASSET_DIR:-$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)}}"

export VISUAL_ENCODER_CHECKPOINT_PATH="${{VISUAL_ENCODER_CHECKPOINT_PATH:-${{ASSET_DIR}}/visual_encoder_checkpoint.pt}}"
export BRIDGE_MODULES_STATE_PATH="${{BRIDGE_MODULES_STATE_PATH:-${{ASSET_DIR}}/bridge_modules_state.pt}}"
export DINOV2_MODEL_NAME_OR_PATH="${{DINOV2_MODEL_NAME_OR_PATH:-{str(args.dinov2_model_name_or_path).strip()}}}"
export MODEL_NAME_OR_PATH="${{MODEL_NAME_OR_PATH:-{str(args.qwen_model_name_or_path).strip()}}}"
export FREEZE_VISION_ENCODER="${{FREEZE_VISION_ENCODER:-true}}"
export VISION_TRAIN_LAST_N_LAYERS="${{VISION_TRAIN_LAST_N_LAYERS:-{int(args.vision_train_last_n_layers)}}}"
export LOCAL_FILES_ONLY="${{LOCAL_FILES_ONLY:-true}}"
"""
    env_path = output_dir / "train_env_template.sh"
    write_text(env_path, template)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "manifest": str(manifest_path),
                "train_env_template": str(env_path),
                "visual_encoder_checkpoint": str(visual_target),
                "bridge_modules_state": str(bridge_target),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
