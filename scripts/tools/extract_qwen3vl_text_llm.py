#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable

import torch

try:
    from safetensors.torch import load_file as load_safetensors_file
    from safetensors.torch import save_file as save_safetensors_file
except Exception as exc:  # pragma: no cover
    load_safetensors_file = None
    save_safetensors_file = None
    _SAFETENSORS_IMPORT_ERROR = exc
else:
    _SAFETENSORS_IMPORT_ERROR = None


DONE_FILE = ".extract_complete"
EXTRACT_VERSION = "1"


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.json",
    "chat_template.jinja",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the text Qwen3 LLM from a Qwen3-VL checkpoint.")
    parser.add_argument("--input-dir", required=True, help="Qwen3-VL checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for the extracted Qwen3 text LLM.")
    parser.add_argument("--overwrite", action="store_true", help="Remove an incomplete output directory before extraction.")
    return parser.parse_args()


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def is_ready(output_dir: Path) -> bool:
    done_path = output_dir / DONE_FILE
    if not done_path.is_file() or not (output_dir / "config.json").is_file():
        return False
    done_text = done_path.read_text(encoding="utf-8", errors="ignore")
    if f"version={EXTRACT_VERSION}" not in done_text:
        return False
    return (output_dir / "model.safetensors").is_file() or (output_dir / "model.safetensors.index.json").is_file()


def normalize_text_config(vl_config: Dict) -> Dict:
    text_config = dict(vl_config.get("text_config") or {})
    if not text_config:
        raise ValueError("Qwen3-VL config.json does not contain text_config.")
    text_config["model_type"] = "qwen3"
    text_config["architectures"] = ["Qwen3ForCausalLM"]
    for key in ("dtype", "torch_dtype"):
        text_config.pop(key, None)
    return text_config


def map_llm_key(key: str) -> str | None:
    prefixes = (
        ("model.language_model.", "model."),
        ("language_model.", "model."),
    )
    for prefix, replacement in prefixes:
        if key.startswith(prefix):
            return replacement + key[len(prefix) :]
    if key == "lm_head.weight":
        return key
    return None


def filter_llm_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        mapped_key = map_llm_key(str(key))
        if mapped_key:
            out[mapped_key] = value
    return out


def safetensor_shards(input_dir: Path) -> Iterable[Path]:
    single = input_dir / "model.safetensors"
    if single.is_file():
        yield single
        return
    index_path = input_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = read_json(index_path)
        shard_names = sorted(set(str(name) for name in index.get("weight_map", {}).values()))
        for shard_name in shard_names:
            shard = input_dir / shard_name
            if shard.is_file():
                yield shard
        return
    yield from sorted(input_dir.glob("model-*.safetensors"))


def extract_safetensors(input_dir: Path, output_dir: Path) -> None:
    if load_safetensors_file is None or save_safetensors_file is None:
        raise RuntimeError(f"safetensors import failed: {_SAFETENSORS_IMPORT_ERROR!r}")

    shards = list(safetensor_shards(input_dir))
    if not shards:
        raise FileNotFoundError(f"No safetensors model files found in {input_dir}")

    weight_map: Dict[str, str] = {}
    wrote_any = False
    for shard in shards:
        llm_state = filter_llm_state(dict(load_safetensors_file(str(shard))))
        if not llm_state:
            continue
        save_safetensors_file(llm_state, str(output_dir / shard.name))
        wrote_any = True
        for key in llm_state:
            weight_map[key] = shard.name

    if not wrote_any:
        raise ValueError(
            "Unable to find Qwen3-VL language_model weights. Expected keys like "
            "'model.language_model.layers.0...'."
        )

    if len([name for name in set(weight_map.values())]) == 1 and (output_dir / "model.safetensors").is_file():
        return
    write_json(output_dir / "model.safetensors.index.json", {"metadata": {}, "weight_map": weight_map})


def copy_tokenizer_files(input_dir: Path, output_dir: Path) -> None:
    for filename in TOKENIZER_FILES:
        src = input_dir / filename
        if src.is_file():
            shutil.copy2(src, output_dir / filename)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not (input_dir / "config.json").is_file():
        raise FileNotFoundError(f"Qwen3-VL config.json not found: {input_dir / 'config.json'}")

    if is_ready(output_dir):
        print(f"[qwen3vl-extract] already ready: {output_dir}", flush=True)
        return
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vl_config = read_json(input_dir / "config.json")
    if str(vl_config.get("model_type", "")).lower() != "qwen3_vl":
        print(
            f"[qwen3vl-extract] warning: input model_type={vl_config.get('model_type')!r}, "
            "continuing because text_config is present.",
            flush=True,
        )
    write_json(output_dir / "config.json", normalize_text_config(vl_config))
    copy_tokenizer_files(input_dir, output_dir)
    extract_safetensors(input_dir, output_dir)

    (output_dir / DONE_FILE).write_text(
        f"version={EXTRACT_VERSION}\nsource={input_dir}\n",
        encoding="utf-8",
    )
    print(f"[qwen3vl-extract] extracted text LLM: {input_dir} -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
