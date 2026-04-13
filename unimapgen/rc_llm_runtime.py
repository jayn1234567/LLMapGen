from __future__ import annotations

import inspect
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from transformers import TrainingArguments


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_meta_jsonl(dataset_jsonl: Path, explicit_meta_jsonl: str) -> Path | None:
    explicit = str(explicit_meta_jsonl).strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Meta jsonl not found: {path}")
        return path
    candidate = dataset_jsonl.with_name(f"meta_{dataset_jsonl.name}")
    if candidate.is_file():
        return candidate.resolve()
    return None


def load_json_dict(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict json at {path}, got {type(data)!r}")
    return data


def save_run_args(output_dir: Path, args: Any) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = vars(args) if hasattr(args, "__dict__") else dict(args)
    args_path = output_dir / "args.json"
    args_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return args_path


def inspect_visual_encoder_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    path = Path(str(checkpoint_path).strip()).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Visual encoder checkpoint not found: {path}")
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        return {}
    args = state.get("args", {})
    return dict(args) if isinstance(args, dict) else {}


def infer_visual_layout(image_size: int, encoder_input_pad_size: int, patch_size: int = 14) -> tuple[int, int]:
    effective_size = max(int(image_size), int(encoder_input_pad_size))
    visual_grid_size = max(1, int(effective_size) // int(patch_size))
    return visual_grid_size, visual_grid_size * visual_grid_size


def create_training_arguments(*, base_kwargs: Dict[str, Any], evaluation_strategy: str) -> TrainingArguments:
    supported_args = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    training_kwargs = dict(base_kwargs)
    if "overwrite_output_dir" in supported_args:
        training_kwargs["overwrite_output_dir"] = True
    if "save_safetensors" in supported_args:
        training_kwargs["save_safetensors"] = False

    filtered_training_kwargs = {
        key: value
        for key, value in training_kwargs.items()
        if key in supported_args
    }
    if "evaluation_strategy" in supported_args:
        filtered_training_kwargs["evaluation_strategy"] = str(evaluation_strategy)
    elif "eval_strategy" in supported_args:
        filtered_training_kwargs["eval_strategy"] = str(evaluation_strategy)
    return TrainingArguments(**filtered_training_kwargs)
