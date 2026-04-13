from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image

from unimapgen.data.rc_caption_short_dataset import (
    SIDE_ORDER,
    build_caption_short,
    extract_lines_from_meta,
    format_side_list,
    line_side_support,
    path_length,
)
from unimapgen.data.rc_centerline_cnn_prefix_dataset import (
    index_rows_by_id,
    load_jsonl,
    pil_to_tensor,
)

VALID_SCENE_LABELS = (
    "straight",
    "curved",
    "branching",
    "intersection-approach",
    "complex",
)


def normalize_scene_label(raw_label: Any) -> str:
    label = str(raw_label).strip().lower()
    return label if label in VALID_SCENE_LABELS else ""


def normalize_visible_sides(raw_sides: Sequence[Any]) -> List[str]:
    raw_set = {str(side).strip().lower() for side in raw_sides if str(side).strip()}
    return [side for side in SIDE_ORDER if side in raw_set]


def scene_article(scene_label: str) -> str:
    normalized = str(scene_label).strip().lower()
    if not normalized:
        return "a"
    return "an" if normalized[0] in {"a", "e", "i", "o", "u"} else "a"


def collect_visible_structure_sides(
    *,
    structure_lines: Sequence[Dict[str, Any]],
    patch_size: int,
    border_tol_px: float,
    min_support_length: float | None = None,
) -> List[str]:
    min_len = (
        float(min_support_length)
        if min_support_length is not None
        else float(max(48.0, float(patch_size) * 0.12))
    )
    side_scores: Dict[str, float] = {side: 0.0 for side in SIDE_ORDER}
    for line in structure_lines:
        points = np.asarray(line.get("points", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            continue
        length = path_length(points)
        if length < min_len:
            continue
        for side in set(line_side_support(points, int(patch_size), float(border_tol_px))):
            if side in side_scores:
                side_scores[side] += float(length)
    return [side for side in SIDE_ORDER if float(side_scores.get(side, 0.0)) > 0.0]


def build_semantic_text(scene_label: str, visible_sides: Sequence[str]) -> str:
    normalized_scene = normalize_scene_label(scene_label) or "complex"
    normalized_sides = normalize_visible_sides(visible_sides)
    prefix = (
        "This is a black-background BEV road-structure patch "
        f"showing {scene_article(normalized_scene)} {normalized_scene} road scene."
    )
    if normalized_sides:
        side_text = format_side_list(normalized_sides)
        noun = "side" if len(normalized_sides) == 1 else "sides"
        suffix = f"Visible road structure reaches the {side_text} {noun}."
    else:
        suffix = "Visible road structure stays inside the patch away from the patch borders."
    return f"{prefix} {suffix}"


def build_semantic_group_key(scene_label: str, visible_sides: Sequence[str]) -> str:
    normalized_scene = normalize_scene_label(scene_label) or "complex"
    normalized_sides = normalize_visible_sides(visible_sides)
    side_key = ",".join(normalized_sides) if normalized_sides else "inside"
    return f"{normalized_scene}|{side_key}"


def build_semantic_group_id(group_key: str) -> int:
    digest = hashlib.sha1(str(group_key).encode("utf-8")).digest()
    return int(int.from_bytes(digest[:8], byteorder="big", signed=False) & 0x7FFFFFFFFFFFFFFF)


def build_semantic_target(
    *,
    row: Dict[str, Any],
    meta: Dict[str, Any],
    media_dir: Path,
    patch_size: int,
    border_tol_px: float,
) -> Dict[str, Any]:
    # Stage 1 允许监督字段既来自显式 semantic 标注，也能从结构线/中心线元数据回推出默认目标。
    structure_lines, centerline_lines = extract_lines_from_meta(meta, media_dir=Path(media_dir))
    scene_label = normalize_scene_label(
        meta.get("semantic_scene_label")
        or row.get("semantic_scene_label")
        or meta.get("caption_label")
        or row.get("caption_label")
    )
    if not scene_label:
        scene_label, _ = build_caption_short(
            structure_lines=structure_lines,
            centerline_lines=centerline_lines,
            patch_size=int(patch_size),
            border_tol_px=float(border_tol_px),
        )
        scene_label = normalize_scene_label(scene_label) or "complex"

    visible_sides = normalize_visible_sides(
        meta.get("semantic_visible_sides")
        or row.get("semantic_visible_sides")
        or []
    )
    if not visible_sides:
        visible_sides = collect_visible_structure_sides(
            structure_lines=structure_lines,
            patch_size=int(patch_size),
            border_tol_px=float(border_tol_px),
        )

    semantic_text = str(meta.get("semantic_text") or row.get("semantic_text") or "").strip()
    if not semantic_text:
        semantic_text = build_semantic_text(scene_label, visible_sides)

    group_key = str(meta.get("semantic_group_key") or row.get("semantic_group_key") or "").strip()
    if not group_key:
        group_key = build_semantic_group_key(scene_label, visible_sides)

    raw_group_id = meta.get("semantic_group_id", row.get("semantic_group_id", 0))
    try:
        group_id = int(raw_group_id)
    except Exception:
        group_id = 0
    if group_id <= 0:
        group_id = build_semantic_group_id(group_key)

    return {
        "semantic_text": semantic_text,
        "semantic_scene_label": scene_label,
        "semantic_visible_sides": list(visible_sides),
        "semantic_group_key": group_key,
        "semantic_group_id": int(group_id),
        "semantic_visible_side_count": len(visible_sides),
        "semantic_source": str(meta.get("semantic_source", "")).strip() or "scene_plus_visible_sides_natural_language_v1",
    }


@dataclass
class RawRCSemanticAlignSample:
    sample_id: str
    image_path: Path
    pixel_values: torch.Tensor
    text: str
    group_id: int
    scene_label: str
    visible_sides: List[str]


class RCSemanticAlignDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        rows: Sequence[Dict[str, Any]],
        meta_rows: Sequence[Dict[str, Any]] | None,
        media_dir: Path,
        image_size: int,
        tokenizer: Any,
        cutoff_len: int,
        border_tol_px: float = 18.0,
    ) -> None:
        self.rows = list(rows)
        self.meta_by_id = index_rows_by_id(meta_rows)
        self.media_dir = Path(media_dir)
        self.image_size = int(image_size)
        self.tokenizer = tokenizer
        self.cutoff_len = int(cutoff_len)
        self.border_tol_px = float(border_tol_px)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> RawRCSemanticAlignSample:
        row = self.rows[index]
        sample_id = str(row.get("id", index))
        meta = self.meta_by_id.get(sample_id, {})
        rel_image = str(row.get("images", [""])[0] if row.get("images") else meta.get("image", "")).strip()
        image_path = (self.media_dir / rel_image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found for sample {sample_id}: {image_path}")

        with Image.open(image_path) as img:
            pixel_values = pil_to_tensor(img, image_size=self.image_size)

        # 统一在数据层把 scene / visible sides / semantic text 收拢好，训练阶段只消费标准字段。
        target = build_semantic_target(
            row=row,
            meta=meta,
            media_dir=self.media_dir,
            patch_size=int(self.image_size),
            border_tol_px=float(self.border_tol_px),
        )
        return RawRCSemanticAlignSample(
            sample_id=sample_id,
            image_path=image_path,
            pixel_values=pixel_values,
            text=str(row.get("text", "")).strip() or str(target["semantic_text"]),
            group_id=int(target["semantic_group_id"]),
            scene_label=str(target["semantic_scene_label"]),
            visible_sides=list(target["semantic_visible_sides"]),
        )


class RCSemanticAlignCollator:
    def __init__(
        self,
        *,
        tokenizer: Any,
        cutoff_len: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.cutoff_len = int(cutoff_len)

    def __call__(self, features: Sequence[RawRCSemanticAlignSample]) -> Dict[str, torch.Tensor]:
        # Stage 1 仍然是图文双塔：文本过 tokenizer，图像直接堆成 pixel batch。
        text_batch = self.tokenizer(
            [item.text for item in features],
            padding=True,
            truncation=True,
            max_length=self.cutoff_len,
            return_tensors="pt",
        )
        return {
            "input_ids": text_batch["input_ids"],
            "attention_mask": text_batch["attention_mask"],
            "pixel_values": torch.stack([item.pixel_values for item in features], dim=0),
            "group_ids": torch.tensor([int(item.group_id) for item in features], dtype=torch.long),
        }


__all__ = [
    "VALID_SCENE_LABELS",
    "build_semantic_group_id",
    "build_semantic_group_key",
    "build_semantic_target",
    "build_semantic_text",
    "collect_visible_structure_sides",
    "load_jsonl",
    "normalize_scene_label",
    "normalize_visible_sides",
    "RCSemanticAlignCollator",
    "RCSemanticAlignDataset",
    "RawRCSemanticAlignSample",
]
