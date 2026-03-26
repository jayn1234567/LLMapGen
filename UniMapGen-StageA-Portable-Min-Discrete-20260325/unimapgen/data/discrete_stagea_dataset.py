from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from unimapgen.discrete_map_token_format import DiscreteMapTokenFormatter


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_user_text(raw_text: str) -> str:
    return str(raw_text).replace("<image>\n", "", 1).replace("<image>", "", 1).strip()


def extract_assistant_lines(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    for msg in sample.get("messages", []):
        if str(msg.get("role", "")).strip().lower() != "assistant":
            continue
        payload = json.loads(str(msg.get("content", "")))
        return list(payload.get("lines", []))
    raise ValueError(f"No assistant message found in sample: {sample.get('id', '<unknown>')}")


def extract_gt_lines(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        return extract_assistant_lines(sample)
    except ValueError:
        return []


def build_prompt_conversation(
    sample: Dict[str, Any],
    image_path: str,
    formatter: DiscreteMapTokenFormatter,
) -> List[Dict[str, Any]]:
    conversation: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": formatter.build_system_prompt()}]}
    ]
    for msg in sample.get("messages", []):
        if str(msg.get("role", "")).strip().lower() != "user":
            continue
        conversation.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": build_user_text(msg.get("content", ""))},
                ],
            }
        )
    return conversation


def build_full_conversation(
    sample: Dict[str, Any],
    image_path: str,
    formatter: DiscreteMapTokenFormatter,
) -> List[Dict[str, Any]]:
    conversation = build_prompt_conversation(sample=sample, image_path=image_path, formatter=formatter)
    conversation.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": formatter.lines_to_text(extract_assistant_lines(sample))}],
        }
    )
    return conversation


def _coerce_xy_point(raw: Any) -> Optional[List[float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        x = float(raw[0])
        y = float(raw[1])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return [x, y]


def sanitize_lines(lines: Any) -> List[Dict[str, Any]]:
    if not isinstance(lines, list):
        return []
    sanitized: List[Dict[str, Any]] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        raw_points = line.get("points", [])
        cleaned_points: List[List[int]] = []
        if isinstance(raw_points, (list, tuple)):
            if len(raw_points) >= 2 and not isinstance(raw_points[0], (list, tuple)):
                point = _coerce_xy_point(raw_points)
                if point is not None:
                    cleaned_points.append([int(round(point[0])), int(round(point[1]))])
            else:
                for raw in raw_points:
                    point = _coerce_xy_point(raw)
                    if point is not None:
                        cleaned_points.append([int(round(point[0])), int(round(point[1]))])
        if len(cleaned_points) < 2:
            continue
        new_line = {
            "category": line.get("category", "road"),
            "start_type": line.get("start_type", "start"),
            "end_type": line.get("end_type", "end"),
            "points": cleaned_points,
        }
        for key in ("score", "confidence", "line_score"):
            if key in line:
                new_line[key] = line[key]
        sanitized.append(new_line)
    return sanitized


def select_items(
    items: Sequence[Dict[str, Any]],
    sample_ids: Sequence[str],
    id_prefixes: Sequence[str],
    max_samples: int,
) -> List[Dict[str, Any]]:
    rows = list(items)
    if sample_ids:
        wanted = set(sample_ids)
        selected = [item for item in rows if str(item.get("id", "")) in wanted]
        missing = [item for item in sample_ids if item not in {str(row.get("id", "")) for row in selected}]
        if missing:
            raise ValueError(f"sample ids not found: {missing}")
        return selected
    if id_prefixes:
        selected = [item for item in rows if any(str(item.get("id", "")).startswith(prefix) for prefix in id_prefixes)]
        if not selected:
            raise ValueError(f"id prefixes not found: {list(id_prefixes)}")
        return selected[: max_samples if max_samples > 0 else None]
    if max_samples <= 0:
        return rows
    return rows[:max_samples]


@dataclass
class RawSample:
    sample_id: str
    image_path: Path
    prompt_text: str
    full_text: str


class PatchOnlyDiscreteDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        media_dir: Path,
        processor: AutoProcessor,
        formatter: DiscreteMapTokenFormatter,
    ) -> None:
        self.rows = list(rows)
        self.media_dir = Path(media_dir)
        self.processor = processor
        self.formatter = formatter

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> RawSample:
        sample = self.rows[index]
        rel_image = str(sample.get("images", [""])[0])
        image_path = (self.media_dir / rel_image).resolve()
        prompt_conv = build_prompt_conversation(sample=sample, image_path=str(image_path), formatter=self.formatter)
        full_conv = build_full_conversation(sample=sample, image_path=str(image_path), formatter=self.formatter)
        prompt_text = self.processor.apply_chat_template(prompt_conv, tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(full_conv, tokenize=False, add_generation_prompt=False)
        return RawSample(
            sample_id=str(sample.get("id", index)),
            image_path=image_path,
            prompt_text=str(prompt_text),
            full_text=str(full_text),
        )


class PatchOnlyDiscreteCollator:
    def __init__(self, processor: AutoProcessor, cutoff_len: int) -> None:
        self.processor = processor
        self.cutoff_len = int(cutoff_len)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Processor does not expose a tokenizer.")
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[RawSample]) -> Dict[str, torch.Tensor]:
        images: List[Image.Image] = []
        prompt_texts: List[str] = []
        full_texts: List[str] = []
        for item in features:
            with Image.open(item.image_path) as image:
                images.append(image.convert("RGB"))
            prompt_texts.append(item.prompt_text)
            full_texts.append(item.full_text)

        full_batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.cutoff_len,
            return_tensors="pt",
        )
        prompt_text_batch = self.tokenizer(
            text=prompt_texts,
            padding=True,
            truncation=True,
            max_length=self.cutoff_len,
            return_tensors="pt",
        )
        full_text_batch = self.tokenizer(
            text=full_texts,
            padding=True,
            truncation=True,
            max_length=self.cutoff_len,
            return_tensors="pt",
        )

        input_ids = full_batch["input_ids"]
        attention_mask = full_batch["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        for batch_idx in range(len(features)):
            full_len = int(attention_mask[batch_idx].sum().item())
            prompt_text_len = int(prompt_text_batch["attention_mask"][batch_idx].sum().item())
            full_text_len = int(full_text_batch["attention_mask"][batch_idx].sum().item())
            prompt_len = max(0, full_len - max(0, full_text_len - prompt_text_len))
            labels[batch_idx, :prompt_len] = -100

        batch = dict(full_batch)
        batch["labels"] = labels
        return batch
