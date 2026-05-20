from __future__ import annotations

import json
import os
from pathlib import Path
import random

import torch
import transformers

from mllm import conversation as conversation_lib
from mllm.rl.config import GRPOArguments, RLDataArguments, RLModelArguments
from mllm.rl.grpo_trainer import GRPOCoordinator


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_paths(paths):
    if paths is None:
        return []
    if isinstance(paths, str):
        return [item for item in paths.replace(";", ",").split(",") if item]
    return list(paths)


def train():
    parser = transformers.HfArgumentParser((RLModelArguments, RLDataArguments, GRPOArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    _set_seed(train_args.seed)

    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        raise KeyError(f"Unknown conversation template: {model_args.version}")

    data_paths = _split_paths(data_args.data_path)
    image_folders = _split_paths(data_args.image_folder)
    if not data_paths:
        raise ValueError("--data_path is required for GRPO training")
    if not image_folders:
        raise ValueError("--image_folder is required for GRPO training")
    if data_args.map_task not in {"lane", "lane_intersection"}:
        raise ValueError("--map_task must be lane or lane_intersection")
    if train_args.rollout_backend != "vllm_prompt_embeds":
        raise ValueError(
            "This entrypoint is reserved for the formal vLLM rollout architecture. "
            "Use --rollout_backend vllm_prompt_embeds."
        )
    if not model_args.disable_deepstack:
        raise ValueError(
            "vLLM prompt-embed GRPO currently supports no-DeepStack only. "
            "Pass --disable_deepstack True."
        )
    if train_args.kl_beta > 0 and not train_args.lora_enable:
        raise ValueError("KL_BETA > 0 currently requires LoRA so the reference can disable the adapter.")

    output_dir = Path(train_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    startup = {
        "pid": os.getpid(),
        "model": model_args.model_name_or_path,
        "vision_tower": model_args.vision_tower,
        "map_task": data_args.map_task,
        "rollout_backend": train_args.rollout_backend,
        "vllm_model_path": train_args.vllm_model_path,
        "lora_enable": train_args.lora_enable,
        "kl_beta": train_args.kl_beta,
    }
    print("[mllm-rl] startup: " + json.dumps(startup, ensure_ascii=False))
    result = GRPOCoordinator(model_args, data_args, train_args).train()
    print("[mllm-rl] finished: " + json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    train()
