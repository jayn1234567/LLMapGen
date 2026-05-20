#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mllm.rl.export import export_merged_lora_checkpoint
from mllm.rl.modeling import load_policy_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a multimodal LoRA checkpoint into a full checkpoint.")
    parser.add_argument("--adapter-checkpoint", required=True, help="PEFT adapter checkpoint directory.")
    parser.add_argument("--model-base", default=None, help="Base multimodal checkpoint if not stored in adapter_config.json.")
    parser.add_argument("--output-dir", required=True, help="Merged full checkpoint output directory.")
    parser.add_argument("--vision-tower", default=None)
    parser.add_argument("--mm-vision-tower-type", default=None)
    parser.add_argument("--input-image-size", type=int, default=None)
    parser.add_argument("--disable-deepstack", type=str, default="True")
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    disable_deepstack = str(args.disable_deepstack).lower() in {"1", "true", "yes", "on"}
    model_args = SimpleNamespace(
        model_name_or_path=args.adapter_checkpoint,
        model_base=args.model_base,
        vision_tower=args.vision_tower,
        mm_vision_tower_type=args.mm_vision_tower_type,
        mm_vision_select_layer=-1,
        mm_vision_select_feature="patch",
        mm_projector_type="mlp2x_gelu",
        mm_patch_merge_type="flat",
        input_image_size=args.input_image_size,
        deepstack_visual_indexes=None,
        disable_deepstack=disable_deepstack,
        tokenizer_use_fast=False,
    )
    train_args = SimpleNamespace(
        bf16=args.bf16,
        fp16=args.fp16,
        model_max_length=args.model_max_length,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer, model, _ = load_policy_model(model_args, train_args, device)
    if device != "cpu":
        model.to(device)
    out = export_merged_lora_checkpoint(model, tokenizer, args.output_dir)
    print(f"Merged checkpoint written to: {out}")


if __name__ == "__main__":
    main()
