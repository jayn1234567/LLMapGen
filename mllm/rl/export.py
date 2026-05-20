from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any

import torch

from mllm.model.qwen_token_utils import normalize_qwen_config_dict
from mllm.train.checkpoint_metadata import (
    sync_qwen_multimodal_config,
    write_qwen_multimodal_checkpoint_metadata,
)


TEXT_DECODER_EXCLUDE_PREFIXES = (
    "model.vision_tower.",
    "model.mm_projector.",
    "model.image_newline",
    "model.deepstack",
    "model.deepstack_mergers",
    "vision_tower.",
    "mm_projector.",
    "deepstack",
)


def _copy_if_exists(src_dir: Path, dst_dir: Path, names: tuple[str, ...]) -> None:
    for name in names:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def _sanitize_tokenizer_config(tokenizer_config_path: Path) -> None:
    if not tokenizer_config_path.exists():
        return
    try:
        data = _load_json(tokenizer_config_path)
    except Exception:
        return
    changed = False
    # Qwen3-VL tokenizer_config from newer Transformers may store this as a
    # list. Transformers 4.56 expects a name->token mapping and crashes before
    # vLLM can start. The actual tokens are already carried by tokenizer.json.
    if isinstance(data.get("extra_special_tokens"), list):
        data.pop("extra_special_tokens", None)
        changed = True
    if changed:
        tokenizer_config_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_weight_files(checkpoint_dir: Path) -> list[Path]:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = checkpoint_dir / index_name
        if index_path.exists():
            index = _load_json(index_path)
            return sorted({checkpoint_dir / shard for shard in index.get("weight_map", {}).values()})
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = checkpoint_dir / name
        if path.exists():
            return [path]
    raise FileNotFoundError(f"No full model weights found under {checkpoint_dir}")


def _load_weight_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover
            raise ImportError("safetensors is required to export vLLM text decoder weights") from exc
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu")


def _save_weight_file(state: dict[str, torch.Tensor], path: Path) -> None:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover
            raise ImportError("safetensors is required to export vLLM text decoder weights") from exc
        save_file(state, str(path))
        return
    torch.save(state, path)


def _is_text_decoder_key(key: str) -> bool:
    if any(key.startswith(prefix) for prefix in TEXT_DECODER_EXCLUDE_PREFIXES):
        return False
    if key.startswith("model.") or key.startswith("lm_head."):
        return True
    return False


def _text_decoder_config(multimodal_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(multimodal_config)
    model_type = str(config.get("model_type") or "").lower()
    if "qwen3" in model_type:
        config["model_type"] = "qwen3"
        config["architectures"] = ["Qwen3ForCausalLM"]
    elif "qwen2" in model_type:
        config["model_type"] = "qwen2"
        config["architectures"] = ["Qwen2ForCausalLM"]
    else:
        raise ValueError(f"Cannot export vLLM text decoder for model_type={model_type!r}")

    for key in list(config):
        if key.startswith("mm_") or key in {
            "vision_tower",
            "input_image_size",
            "deepstack_visual_indexes",
            "disable_deepstack",
            "use_mm_proj",
            "freeze_mm_mlp_adapter",
            "tune_mm_mlp_adapter",
            "image_aspect_ratio",
            "image_grid_pinpoints",
            "tokenizer_model_max_length",
            "tokenizer_padding_side",
        }:
            config.pop(key, None)
    normalize_qwen_config_dict(config)
    return config


def export_text_decoder_checkpoint(
    multimodal_checkpoint: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """Export the text decoder part of a no-DeepStack multimodal checkpoint for vLLM.

    The actor computes DINO/projector prompt embeddings. vLLM only needs the
    Qwen decoder weights plus tokenizer files to continue generation from those
    prompt embeddings.
    """

    checkpoint_dir = Path(multimodal_checkpoint)
    output_dir = Path(output_dir)
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {checkpoint_dir}")
    config = _load_json(config_path)
    if not bool(config.get("disable_deepstack", True)):
        raise ValueError(
            "vLLM prompt-embed rollout currently supports no-DeepStack checkpoints only. "
            "DeepStack needs layer-level visual residual injection and cannot be represented "
            "by prompt embeddings alone."
        )

    done_path = output_dir / ".vllm_text_export_complete"
    if done_path.exists() and not overwrite:
        return output_dir

    tmp_dir = Path(f"{output_dir}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    text_config = _text_decoder_config(config)
    (tmp_dir / "config.json").write_text(json.dumps(text_config, ensure_ascii=False, indent=2), encoding="utf-8")

    state: dict[str, torch.Tensor] = {}
    for weight_file in _checkpoint_weight_files(checkpoint_dir):
        for key, value in _load_weight_file(weight_file).items():
            if _is_text_decoder_key(key):
                state[key] = value
    if not state:
        raise ValueError(f"No text decoder tensors found in {checkpoint_dir}")
    _save_weight_file(state, tmp_dir / "model.safetensors")

    _copy_if_exists(
        checkpoint_dir,
        tmp_dir,
        (
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "generation_config.json",
            "chat_template.json",
            "chat_template.jinja",
            "special_tokens_map.json",
        ),
    )
    _sanitize_tokenizer_config(tmp_dir / "tokenizer_config.json")
    if not (tmp_dir / "generation_config.json").exists():
        _, generation_config = normalize_qwen_config_dict(text_config, {})
        (tmp_dir / "generation_config.json").write_text(
            json.dumps(generation_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    done_path_tmp = tmp_dir / ".vllm_text_export_complete"
    done_path_tmp.write_text(f"source={checkpoint_dir.resolve()}\n", encoding="utf-8")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.replace(tmp_dir, output_dir)
    return output_dir


def export_merged_lora_checkpoint(model, tokenizer, output_dir: str | Path) -> Path:
    """Write a full merged checkpoint from a PEFT actor policy."""

    output_dir = Path(output_dir)
    tmp_dir = Path(f"{output_dir}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    model_to_merge = model
    if hasattr(model_to_merge, "module"):
        model_to_merge = model_to_merge.module
    if hasattr(model_to_merge, "merge_and_unload"):
        model_to_merge = model_to_merge.merge_and_unload()
    try:
        sync_qwen_multimodal_config(model_to_merge)
    except Exception:
        pass
    model_to_merge.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    write_qwen_multimodal_checkpoint_metadata(model_to_merge, tmp_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.replace(tmp_dir, output_dir)
    return output_dir
