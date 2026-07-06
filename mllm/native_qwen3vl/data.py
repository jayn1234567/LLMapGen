from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100
IMAGE_TOKEN = "<image>"
SEQUENCE_EXTRA_KEYS = {"token_type_ids", "mm_token_type_ids", "position_ids"}


def load_json_or_jsonl(file_path: str | os.PathLike) -> list[dict[str, Any]]:
    path = Path(file_path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            return [item for item in payload["records"] if isinstance(item, dict)]
        if isinstance(payload, dict) and isinstance(payload.get("patch_results"), list):
            return [item for item in payload["patch_results"] if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]
    except json.JSONDecodeError:
        pass
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def normalize_path_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item for item in value.split(",") if item]
    return [str(item) for item in value if str(item)]


def strip_image_token(prompt: str) -> str:
    return str(prompt or "").replace(IMAGE_TOKEN, "").strip()


def resolve_image_path(image_value: str, image_folder: str | os.PathLike) -> Path:
    path = Path(image_value)
    if path.is_absolute() and path.exists():
        return path
    return (Path(image_folder) / path).resolve()


def build_qwen3vl_messages(prompt: str, image_path: str | os.PathLike, answer: str | None = None) -> list[dict[str, Any]]:
    user_message = {
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": strip_image_token(prompt)},
        ],
    }
    messages = [user_message]
    if answer is not None:
        messages.append({"role": "assistant", "content": str(answer)})
    return messages


def _tokenizer_from_processor(processor: Any):
    return getattr(processor, "tokenizer", processor)


def _pad_token_id(processor: Any) -> int:
    tokenizer = _tokenizer_from_processor(processor)
    value = getattr(tokenizer, "pad_token_id", None)
    if value is not None:
        return int(value)
    eos = getattr(tokenizer, "eos_token_id", None)
    return int(eos) if eos is not None else 0


def _apply_chat_template(processor: Any, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _processor_call(processor: Any, text: str, image: Image.Image, max_length: int | None):
    kwargs = {
        "text": [text],
        "images": [image],
        "return_tensors": "pt",
    }
    if max_length and max_length > 0:
        kwargs.update({"truncation": True, "max_length": max_length})
    return processor(**kwargs)


def _as_1d(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[0] if tensor.ndim >= 2 and tensor.shape[0] == 1 else tensor


def _maybe_squeeze_grid(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[0] if tensor.ndim == 3 and tensor.shape[0] == 1 else tensor


class NativeQwen3VLDataset(Dataset):
    """Dataset adapter for existing UniMapGen conversation JSONL files."""

    def __init__(
        self,
        data_paths: Sequence[str],
        image_folders: Sequence[str],
        sample_limit: int = 0,
        sample_seed: int = 42,
    ):
        if not data_paths:
            raise ValueError("data_paths must not be empty")
        if not image_folders:
            raise ValueError("image_folders must not be empty")
        if len(image_folders) == 1 and len(data_paths) > 1:
            image_folders = list(image_folders) * len(data_paths)
        if len(image_folders) != len(data_paths):
            raise ValueError("image_folders must have length 1 or match data_paths")

        records: list[dict[str, Any]] = []
        for data_idx, data_path in enumerate(data_paths):
            for record in load_json_or_jsonl(data_path):
                copied = dict(record)
                copied["_native_data_idx"] = data_idx
                records.append(copied)

        if sample_limit and sample_limit > 0 and len(records) > sample_limit:
            import random

            rng = random.Random(sample_seed)
            indices = sorted(rng.sample(range(len(records)), sample_limit))
            records = [records[idx] for idx in indices]

        self.records = records
        self.image_folders = list(image_folders)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        conversations = record.get("conversations") or []
        if not conversations:
            raise ValueError(f"record has no conversations: {record.get('id', index)}")
        prompt = conversations[0].get("value", "")
        answer = conversations[1].get("value", "") if len(conversations) > 1 else ""
        image_folder = self.image_folders[int(record.get("_native_data_idx", 0))]
        image_path = resolve_image_path(record["image"], image_folder)
        return {
            "record": copy.deepcopy(record),
            "image_path": image_path,
            "prompt": prompt,
            "answer": answer,
        }


class NativeQwen3VLDataCollator:
    """Collates native Qwen3-VL multimodal examples and masks user tokens."""

    def __init__(self, processor: Any, model_max_length: int = 4096):
        self.processor = processor
        self.model_max_length = model_max_length
        self.pad_token_id = _pad_token_id(processor)

    def _encode_one(self, feature: dict[str, Any]) -> dict[str, Any]:
        image = Image.open(feature["image_path"]).convert("RGB")
        full_messages = build_qwen3vl_messages(feature["prompt"], feature["image_path"], feature["answer"])
        prompt_messages = build_qwen3vl_messages(feature["prompt"], feature["image_path"], None)

        full_text = _apply_chat_template(self.processor, full_messages, add_generation_prompt=False)
        prompt_text = _apply_chat_template(self.processor, prompt_messages, add_generation_prompt=True)

        full_inputs = _processor_call(self.processor, full_text, image, self.model_max_length)
        prompt_inputs = _processor_call(self.processor, prompt_text, image, self.model_max_length)
        input_ids = _as_1d(full_inputs["input_ids"])
        labels = input_ids.clone()
        prompt_len = min(int(_as_1d(prompt_inputs["input_ids"]).numel()), int(labels.numel()))
        labels[:prompt_len] = IGNORE_INDEX
        if self.pad_token_id is not None:
            labels[input_ids == self.pad_token_id] = IGNORE_INDEX

        encoded: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": _as_1d(full_inputs.get("attention_mask", torch.ones_like(input_ids))),
            "labels": labels,
        }
        for key, value in full_inputs.items():
            if key in encoded:
                continue
            if torch.is_tensor(value):
                if key in SEQUENCE_EXTRA_KEYS:
                    encoded[key] = _as_1d(value)
                elif key.endswith("grid_thw"):
                    encoded[key] = _maybe_squeeze_grid(value)
                else:
                    encoded[key] = value
            else:
                encoded[key] = value
        return encoded

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode_one(feature) for feature in features]
        batch = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [item["input_ids"] for item in encoded],
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [item["attention_mask"] for item in encoded],
                batch_first=True,
                padding_value=0,
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                [item["labels"] for item in encoded],
                batch_first=True,
                padding_value=IGNORE_INDEX,
            ),
        }

        extra_keys = sorted(set().union(*(item.keys() for item in encoded)) - set(batch.keys()))
        for key in extra_keys:
            values = [item[key] for item in encoded if key in item]
            if not values or not torch.is_tensor(values[0]):
                continue
            if key in SEQUENCE_EXTRA_KEYS:
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    values,
                    batch_first=True,
                    padding_value=0,
                )
            elif key in {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}:
                batch[key] = torch.cat(values, dim=0)
            elif all(value.shape == values[0].shape for value in values):
                batch[key] = torch.stack(values, dim=0)
            else:
                batch[key] = values
        return batch
