from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from PIL import Image

from unimapgen.data.rc_centerline_cnn_prefix_dataset import (
    index_rows_by_id,
    load_jsonl,
    pil_to_tensor,
)


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert road-centerline reconstruction assistant for black-background BEV road-structure images.\n\n"
    "VISIBLE SEMANTICS:\n"
    "The visible road-structure classes are lane_boundary, lane_divider, and background.\n"
    "The image does not show centerlines directly.\n\n"
    "TASK DEFINITION:\n"
    "Your task is to infer the unseen road centerlines strictly from the visible road structure.\n"
    "1. A centerline is the geometric middle path of one valid drivable corridor.\n"
    "2. Do not trace lane_boundary or lane_divider themselves.\n"
    "3. Keep different lanes, branches, and intersecting paths as separate continuous polylines.\n"
    "4. If a centerline reaches the patch border, terminate it at the visible border.\n"
    "5. Predict all valid centerlines implied by the visible road structure in the current patch only.\n\n"
    "OUTPUT CONSTRAINTS:\n"
    "1. Return ONLY valid JSON.\n"
    "2. Do NOT wrap the JSON in markdown fences.\n"
    "3. Do NOT output explanations or extra text.\n"
    "4. Use the patch-local coordinate system.\n"
    "5. All x and y coordinates must be integers between 0 and 512 inclusive.\n"
    "6. Strictly use this JSON structure:\n"
    '{"lines":[]}\n'
    "or\n"
    '{"lines":[{"points":[[x1,y1],[x2,y2]]}]}'
)

DEFAULT_USER_PROMPT = (
    "This is a black-background BEV road-structure image.\n"
    "Predict the road centerlines for this patch from the visible lane_boundary and lane_divider structure.\n"
    "Return only the raw JSON object."
)

VISUAL_TOKENS = [
    "<vis_start>",
    "<vis_patch>",
    "<vis_end>",
]


def build_visual_placeholder(num_visual_tokens: int) -> str:
    return "<vis_start> " + " ".join(["<vis_patch>"] * int(num_visual_tokens)) + " <vis_end>"


def extract_message_content(sample: Dict[str, Any], role: str) -> str:
    wanted = str(role).strip().lower()
    for message in sample.get("messages", []):
        if str(message.get("role", "")).strip().lower() != wanted:
            continue
        content = str(message.get("content", "")).strip()
        if content:
            return content
    return ""


def normalize_int_coord(value: Any) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    rounded = int(round(numeric))
    return max(0, min(512, rounded))


def normalize_centerline_json_text(text: str) -> str:
    payload = json.loads(str(text).strip())
    raw_lines = payload.get("lines", [])
    normalized_lines: List[Dict[str, Any]] = []
    if isinstance(raw_lines, list):
        for raw_line in raw_lines:
            if not isinstance(raw_line, dict):
                continue
            raw_points = raw_line.get("points", [])
            if not isinstance(raw_points, list):
                continue
            points: List[List[int]] = []
            for point in raw_points:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                points.append(
                    [
                        normalize_int_coord(point[0]),
                        normalize_int_coord(point[1]),
                    ]
                )
            if len(points) >= 2:
                line_out: Dict[str, Any] = {}
                category = str(raw_line.get("category", "")).strip()
                if category:
                    line_out["category"] = category
                start_type = str(raw_line.get("start_type", "")).strip()
                end_type = str(raw_line.get("end_type", "")).strip()
                if start_type:
                    line_out["start_type"] = start_type
                if end_type:
                    line_out["end_type"] = end_type
                line_out["points"] = points
                normalized_lines.append(line_out)
    raw_intersections = payload.get("intersections", [])
    normalized_intersections: List[Dict[str, Any]] = []
    if isinstance(raw_intersections, list):
        for raw_item in raw_intersections:
            if not isinstance(raw_item, dict):
                continue
            raw_points = raw_item.get("points", [])
            if not isinstance(raw_points, list):
                continue
            points: List[List[int]] = []
            for point in raw_points:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                xy = [
                    normalize_int_coord(point[0]),
                    normalize_int_coord(point[1]),
                ]
                if not points or points[-1] != xy:
                    points.append(xy)
            if len(points) >= 3:
                if points[0] != points[-1]:
                    points.append(list(points[0]))
                if len(points) >= 4:
                    raw_type = raw_item.get("type", raw_item.get("label", "other"))
                    label = str(raw_type or "other").strip().lower().replace("-", "_").replace(" ", "_")
                    while "__" in label:
                        label = label.replace("__", "_")
                    if ("merge" in label and "split" in label) or (label.startswith("ra") and ("split" in label or "saplit" in label)):
                        label = "ramp_merge_split"
                    if label in {"0ther", "oother", "label", "null", "none", ""}:
                        label = "other"
                    if label == "round_about":
                        label = "roundabout"
                    if ("unsignal" in label or label.startswith("uns")) and "cross" in label:
                        label = "unsignalized_crossing"
                    if label.startswith("t_") and ("inter" in label or "section" in label):
                        label = "t_intersection"
                    if "cross" in label and ("inter" in label or "section" in label):
                        label = "cross_intersection"
                    if not label:
                        label = "other"
                    normalized_intersections.append({"type": label, "points": points})
    return json.dumps(
        {"lines": normalized_lines, "intersections": normalized_intersections},
        ensure_ascii=False,
        separators=(",", ":"),
    )


class RCCenterlineJSONSFTFormatter:
    def __init__(
        self,
        *,
        image_size: int,
        num_visual_tokens: int,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
    ) -> None:
        self.image_size = int(image_size)
        self.num_visual_tokens = int(num_visual_tokens)
        self.system_prompt = str(system_prompt).strip()
        self.user_prompt = str(user_prompt).strip()
        self.special_tokens = list(VISUAL_TOKENS)

    def register_tokens(self, tokenizer: Any) -> int:
        vocab = tokenizer.get_vocab()
        new_tokens = [tok for tok in self.special_tokens if tok not in vocab]
        if new_tokens:
            tokenizer.add_tokens(new_tokens, special_tokens=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return len(new_tokens)

    def build_user_text(self, user_prompt: str | None = None) -> str:
        prompt_body = str(user_prompt).strip() if user_prompt is not None else str(self.user_prompt).strip()
        return f"{build_visual_placeholder(self.num_visual_tokens)}\n{prompt_body}"

    def apply_chat_template(
        self,
        tokenizer: Any,
        *,
        system_text: str,
        user_text: str,
        assistant_text: str | None,
        add_generation_prompt: bool,
    ) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            conv = [
                {"role": "system", "content": str(system_text)},
                {"role": "user", "content": str(user_text)},
            ]
            if assistant_text is not None:
                conv.append({"role": "assistant", "content": str(assistant_text)})
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": bool(add_generation_prompt),
            }
            for extra_kwargs in ({"enable_thinking": False}, {"thinking": False}, {}):
                try:
                    return tokenizer.apply_chat_template(conv, **template_kwargs, **extra_kwargs)
                except TypeError:
                    continue
        pieces = [
            f"System:\n{system_text}",
            f"User:\n{user_text}",
        ]
        if assistant_text is not None:
            pieces.append(f"Assistant:\n{assistant_text}")
        elif add_generation_prompt:
            pieces.append("Assistant:\n")
        return "\n\n".join(pieces)


@dataclass
class RawRCCenterlineJSONSFTSample:
    sample_id: str
    image_path: Path
    pixel_values: torch.Tensor
    prompt_text: str
    full_text: str
    assistant_json: str


class RCCenterlineJSONSFTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        rows: Sequence[Dict[str, Any]],
        meta_rows: Sequence[Dict[str, Any]] | None,
        media_dir: Path,
        tokenizer: Any,
        formatter: RCCenterlineJSONSFTFormatter,
        image_size: int,
    ) -> None:
        self.rows = list(rows)
        self.meta_by_id = index_rows_by_id(meta_rows)
        self.media_dir = Path(media_dir)
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> RawRCCenterlineJSONSFTSample:
        sample = self.rows[index]
        sample_id = str(sample.get("id", index))
        meta = self.meta_by_id.get(sample_id, {})
        rel_image = str(sample.get("images", [""])[0] if sample.get("images") else meta.get("image", "")).strip()
        image_path = (self.media_dir / rel_image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found for sample {sample_id}: {image_path}")

        with Image.open(image_path) as img:
            pixel_values = pil_to_tensor(img, image_size=self.image_size)

        assistant_raw = extract_message_content(sample, "assistant")
        if not assistant_raw:
            assistant_raw = json.dumps({"lines": list(meta.get("target_lines", []))}, ensure_ascii=False, separators=(",", ":"))
        assistant_json = normalize_centerline_json_text(assistant_raw)

        system_text = str(self.formatter.system_prompt)
        user_text = self.formatter.build_user_text(self.formatter.user_prompt)
        prompt_text = self.formatter.apply_chat_template(
            self.tokenizer,
            system_text=system_text,
            user_text=user_text,
            assistant_text=None,
            add_generation_prompt=True,
        )
        full_text = self.formatter.apply_chat_template(
            self.tokenizer,
            system_text=system_text,
            user_text=user_text,
            assistant_text=assistant_json,
            add_generation_prompt=False,
        )
        return RawRCCenterlineJSONSFTSample(
            sample_id=sample_id,
            image_path=image_path,
            pixel_values=pixel_values,
            prompt_text=str(prompt_text),
            full_text=str(full_text),
            assistant_json=str(assistant_json),
        )


class RCCenterlineJSONSFTCollator:
    def __init__(
        self,
        *,
        tokenizer: Any,
        cutoff_len: int,
        num_visual_tokens: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.cutoff_len = int(cutoff_len)
        self.num_visual_tokens = int(num_visual_tokens)
        self.vis_patch_token_id = int(tokenizer.convert_tokens_to_ids("<vis_patch>"))
        if self.vis_patch_token_id < 0:
            raise ValueError("Tokenizer is missing <vis_patch>.")

    def __call__(self, features: Sequence[RawRCCenterlineJSONSFTSample]) -> Dict[str, torch.Tensor]:
        prompt_texts = [item.prompt_text for item in features]
        full_texts = [item.full_text for item in features]
        pixel_values = torch.stack([item.pixel_values for item in features], dim=0)

        full_batch = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.cutoff_len,
            return_tensors="pt",
        )
        prompt_batch = self.tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=self.cutoff_len,
            return_tensors="pt",
        )

        input_ids = full_batch["input_ids"]
        attention_mask = full_batch["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        vis_patch_mask = input_ids.eq(self.vis_patch_token_id)

        for batch_idx, item in enumerate(features):
            full_len = int(attention_mask[batch_idx].sum().item())
            prompt_text_len = int(prompt_batch["attention_mask"][batch_idx].sum().item())
            full_text_len = int(full_batch["attention_mask"][batch_idx].sum().item())
            prompt_len = max(0, full_len - max(0, full_text_len - prompt_text_len))
            labels[batch_idx, :prompt_len] = -100
            num_vis = int(vis_patch_mask[batch_idx].sum().item())
            if num_vis != self.num_visual_tokens:
                raise ValueError(
                    f"Visual token mismatch for sample={item.sample_id}: "
                    f"expected={self.num_visual_tokens} actual={num_vis}"
                )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "vis_patch_mask": vis_patch_mask,
        }


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_USER_PROMPT",
    "RCCenterlineJSONSFTFormatter",
    "RCCenterlineJSONSFTDataset",
    "RCCenterlineJSONSFTCollator",
    "RawRCCenterlineJSONSFTSample",
    "VISUAL_TOKENS",
    "build_visual_placeholder",
    "extract_message_content",
    "load_jsonl",
    "normalize_centerline_json_text",
]
