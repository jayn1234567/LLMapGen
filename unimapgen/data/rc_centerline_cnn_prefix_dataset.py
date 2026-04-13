from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

RC_BG_RGB = np.asarray([10.0, 12.0, 18.0], dtype=np.float32) / 255.0
RC_SEG_CLASS_COLORS = {
    "background": RC_BG_RGB,
    "lane_divider": np.asarray([74.0, 227.0, 255.0], dtype=np.float32) / 255.0,
    "road_edge": np.asarray([112.0, 255.0, 148.0], dtype=np.float32) / 255.0,
    "ped_edge": np.asarray([255.0, 96.0, 235.0], dtype=np.float32) / 255.0,
    "centerline": np.asarray([255.0, 72.0, 72.0], dtype=np.float32) / 255.0,
}
RC_MULTICLASS_CLASS_NAMES = [
    "background",
    "lane_divider",
    "road_edge",
    "ped_edge",
    "centerline",
]
RC_STRUCTURE_MULTICLASS_CLASS_NAMES = [
    "background",
    "lane_divider",
    "road_edge",
]
RC_DASH_SOLID_CLASS_NAMES = [
    "background",
    "dashed",
    "solid",
]


DEFAULT_SYSTEM_PROMPT = (
    "You are a road-centerline reconstruction assistant for RC-style road-structure images.\n"
    "Predict the road centerlines visible in the current RC patch.\n"
    "Use the reserved structure tokens only.\n"
    "Emit one count token first to indicate how many centerlines are present.\n"
    "Then emit the line structures.\n"
    "If there is no centerline, emit the count token only.\n"
    "Each <coord_pt> token corresponds to one continuous point in patch-local coordinates."
)

DEFAULT_USER_PROMPT = (
    "Please construct the road centerline map in the current RC patch.\n"
    "Keep all coordinates in the patch-local coordinate system."
)


STRUCT_TOKENS = [
    "<vis_start>",
    "<vis_patch>",
    "<vis_end>",
    "<line>",
    "<cat_centerline>",
    "<pts>",
    "<coord_pt>",
    "<eol>",
]
COUNT_BUCKET_TOKENS = [f"<count_{idx:02d}>" for idx in range(12)] + ["<count_12p>"]
STRUCT_TOKENS.extend(COUNT_BUCKET_TOKENS)


def load_jsonl(path: Path, max_samples: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if int(max_samples) > 0 and len(rows) >= int(max_samples):
                break
    return rows


def index_rows_by_id(rows: Sequence[Dict[str, Any]] | None) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if rows is None:
        return out
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        if row_id:
            out[row_id] = dict(row)
    return out


def normalize_coord(value: float, image_size: int) -> float:
    denom = float(max(int(image_size) - 1, 1))
    clipped = max(0.0, min(float(value), denom))
    return clipped / denom


def resize_rgb_image(
    image: Image.Image,
    image_size: int,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> Image.Image:
    return image.convert("RGB").resize((int(image_size), int(image_size)), resample)


def pil_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    img = resize_rgb_image(image, image_size=image_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).contiguous()


def count_to_bucket_token(num_lines: int) -> str:
    num = max(0, int(num_lines))
    if num <= 11:
        return f"<count_{num:02d}>"
    return "<count_12p>"


def extract_valid_lines(lines: Sequence[Dict[str, Any]]) -> List[List[List[float]]]:
    valid_lines: List[List[List[float]]] = []
    for line in lines:
        points = line.get("points", [])
        if not isinstance(points, list) or len(points) < 2:
            continue
        valid_points = [
            [float(point[0]), float(point[1])]
            for point in points
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if len(valid_points) < 2:
            continue
        valid_lines.append(valid_points)
    return valid_lines


def build_centerline_heatmap(
    lines: Sequence[Sequence[Sequence[float]]],
    image_size: int,
    line_width: int = 3,
    blur_radius: float = 1.0,
) -> torch.Tensor:
    canvas = Image.new("L", (int(image_size), int(image_size)), 0)
    draw = ImageDraw.Draw(canvas)
    width = max(1, int(line_width))
    for line in lines:
        if len(line) < 2:
            continue
        draw.line(
            [(float(point[0]), float(point[1])) for point in line],
            fill=255,
            width=width,
            joint="curve",
        )
    if float(blur_radius) > 0.0:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).contiguous()


def build_segmentation_label_map(
    image: Image.Image,
    image_size: int,
    supervision_mode: str = "binary",
    bg_distance_threshold: float = 0.10,
    min_line_intensity: float = 0.12,
) -> torch.Tensor:
    mode = str(supervision_mode).strip().lower()
    if mode not in {"none", "binary", "multiclass", "structure_multiclass", "dash_solid"}:
        raise ValueError(f"Unsupported seg supervision mode: {supervision_mode}")
    if mode == "none":
        return torch.zeros((int(image_size), int(image_size)), dtype=torch.int64)
    img = resize_rgb_image(image, image_size=image_size, resample=Image.Resampling.NEAREST)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    bg_dist = np.linalg.norm(arr - RC_BG_RGB.reshape(1, 1, 3), axis=-1)
    max_intensity = np.max(arr, axis=-1)
    mask = (bg_dist >= float(bg_distance_threshold)) & (max_intensity >= float(min_line_intensity))
    if mode == "binary":
        return torch.from_numpy(mask.astype(np.int64)).contiguous()

    if mode == "structure_multiclass":
        active_class_names = RC_STRUCTURE_MULTICLASS_CLASS_NAMES[1:]
    else:
        active_class_names = RC_MULTICLASS_CLASS_NAMES[1:]
    color_bank = np.stack([RC_SEG_CLASS_COLORS[name] for name in active_class_names], axis=0)
    dist = np.linalg.norm(arr[..., None, :] - color_bank.reshape(1, 1, color_bank.shape[0], 3), axis=-1)
    nearest_idx = np.argmin(dist, axis=-1).astype(np.int64) + 1
    label_map = np.zeros(mask.shape, dtype=np.int64)
    if mode == "dash_solid":
        # lane_divider -> dashed; all solid-colored structure classes -> solid
        dash_solid_map = np.asarray([1, 2, 2, 2], dtype=np.int64)
        label_map[mask] = dash_solid_map[nearest_idx[mask] - 1]
        return torch.from_numpy(label_map).contiguous()

    label_map[mask] = nearest_idx[mask]
    return torch.from_numpy(label_map).contiguous()


def load_segmentation_label_map_from_path(
    seg_path: Path,
    image_size: int,
    supervision_mode: str = "binary",
) -> torch.Tensor:
    mode = str(supervision_mode).strip().lower()
    if mode not in {"none", "binary", "multiclass", "structure_multiclass", "dash_solid"}:
        raise ValueError(f"Unsupported seg supervision mode: {supervision_mode}")
    if mode == "none":
        return torch.zeros((int(image_size), int(image_size)), dtype=torch.int64)

    with Image.open(seg_path) as img:
        if mode == "binary":
            mask = img.convert("L").resize((int(image_size), int(image_size)), Image.Resampling.NEAREST)
            arr = np.asarray(mask, dtype=np.uint8)
            return torch.from_numpy((arr > 0).astype(np.int64)).contiguous()

        if img.mode in {"L", "I", "I;16"}:
            label = img.convert("L").resize((int(image_size), int(image_size)), Image.Resampling.NEAREST)
            arr = np.asarray(label, dtype=np.int64)
            return torch.from_numpy(arr).contiguous()

        return build_segmentation_label_map(
            img,
            image_size=image_size,
            supervision_mode=mode,
        )


def build_visual_placeholder(num_visual_tokens: int) -> str:
    return "<vis_start> " + " ".join(["<vis_patch>"] * int(num_visual_tokens)) + " <vis_end>"


class RCCenterlineFormatter:
    def __init__(
        self,
        image_size: int,
        num_visual_tokens: int = 64,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
    ) -> None:
        self.image_size = int(image_size)
        self.num_visual_tokens = int(num_visual_tokens)
        self.system_prompt = str(system_prompt).strip()
        self.user_prompt = str(user_prompt).strip()
        self.special_tokens = list(STRUCT_TOKENS)

    def register_tokens(self, tokenizer: Any) -> int:
        vocab = tokenizer.get_vocab()
        new_tokens = [tok for tok in self.special_tokens if tok not in vocab]
        if new_tokens:
            tokenizer.add_tokens(new_tokens, special_tokens=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return len(new_tokens)

    def build_user_text(self) -> str:
        return f"{build_visual_placeholder(self.num_visual_tokens)}\n{self.user_prompt}"

    def extract_assistant_lines(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        for msg in sample.get("messages", []):
            if str(msg.get("role", "")).strip().lower() != "assistant":
                continue
            payload = json.loads(str(msg.get("content", "")))
            return list(payload.get("lines", []))
        raise ValueError(f"No assistant payload found for sample: {sample.get('id', '<unknown>')}")

    def lines_to_text_and_coords(
        self,
        lines: Sequence[Dict[str, Any]] | Sequence[Sequence[Sequence[float]]],
    ) -> tuple[str, List[List[float]], List[int], List[int], int]:
        valid_lines = (
            extract_valid_lines(lines)
            if lines and isinstance(lines[0], dict)
            else [[list(point[:2]) for point in line] for line in lines]  # type: ignore[index]
        )
        tokens: List[str] = [count_to_bucket_token(len(valid_lines))]
        coord_values: List[List[float]] = []
        coord_line_ids: List[int] = []
        coord_point_ids: List[int] = []
        for line_idx, valid_points in enumerate(valid_lines):
            tokens.extend(["<line>", "<cat_centerline>", "<pts>"])
            for point_idx, point in enumerate(valid_points):
                tokens.append("<coord_pt>")
                coord_values.append(
                    [
                        normalize_coord(float(point[0]), self.image_size),
                        normalize_coord(float(point[1]), self.image_size),
                    ]
                )
                coord_line_ids.append(int(line_idx))
                coord_point_ids.append(int(point_idx))
            tokens.append("<eol>")
        return " ".join(tokens), coord_values, coord_line_ids, coord_point_ids, int(len(valid_lines))

    def apply_chat_template(
        self,
        tokenizer: Any,
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
class RawRCCenterlineSample:
    sample_id: str
    image_path: Path
    pixel_values: torch.Tensor
    seg_target_labels: torch.Tensor
    centerline_heatmap: torch.Tensor
    prompt_text: str
    full_text: str
    coord_points: List[List[float]]
    coord_line_ids: List[int]
    coord_point_ids: List[int]
    num_lines: int


class RCCenterlinePrefixDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        meta_rows: Sequence[Dict[str, Any]] | None,
        media_dir: Path,
        tokenizer: Any,
        formatter: RCCenterlineFormatter,
        image_size: int,
        seg_supervision: str = "binary",
        seg_bg_distance_threshold: float = 0.10,
        seg_min_line_intensity: float = 0.12,
        max_num_lines: int = 0,
        centerline_heatmap_line_width: int = 3,
        centerline_heatmap_blur_radius: float = 1.0,
    ) -> None:
        self.meta_by_id = index_rows_by_id(meta_rows)
        self.media_dir = Path(media_dir)
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.image_size = int(image_size)
        self.seg_supervision = str(seg_supervision).strip().lower()
        self.seg_bg_distance_threshold = float(seg_bg_distance_threshold)
        self.seg_min_line_intensity = float(seg_min_line_intensity)
        self.max_num_lines = int(max_num_lines)
        self.centerline_heatmap_line_width = max(1, int(centerline_heatmap_line_width))
        self.centerline_heatmap_blur_radius = float(centerline_heatmap_blur_radius)
        self.rows = self._filter_rows(rows)

    def _sample_num_lines(self, sample: Dict[str, Any]) -> int:
        sample_id = str(sample.get("id", "")).strip()
        meta = self.meta_by_id.get(sample_id, {})
        meta_num_lines = meta.get("num_target_lines")
        if meta_num_lines is not None:
            try:
                return max(0, int(meta_num_lines))
            except (TypeError, ValueError):
                pass
        return len(extract_valid_lines(self.formatter.extract_assistant_lines(sample)))

    def _filter_rows(self, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if int(self.max_num_lines) <= 0:
            return list(rows)
        kept_rows: List[Dict[str, Any]] = []
        filtered = 0
        for row in rows:
            num_lines = self._sample_num_lines(row)
            if int(num_lines) > int(self.max_num_lines):
                filtered += 1
                continue
            kept_rows.append(row)
        print(
            (
                "[rc-centerline-dataset] "
                f"max_num_lines={self.max_num_lines} kept={len(kept_rows)} filtered={filtered}"
            ),
            flush=True,
        )
        return kept_rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> RawRCCenterlineSample:
        sample = self.rows[index]
        sample_id = str(sample.get("id", index))
        meta = self.meta_by_id.get(sample_id, {})
        rel_image = str(sample.get("images", [""])[0])
        image_path = (self.media_dir / rel_image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found for sample {sample_id}: {image_path}")

        with Image.open(image_path) as img:
            pixel_values = pil_to_tensor(img, image_size=self.image_size)
            seg_rel = str(meta.get("seg_binary", "")).strip()
            seg_path = (self.media_dir / seg_rel).resolve() if seg_rel else None
            if seg_path is not None and seg_path.is_file():
                seg_target_labels = load_segmentation_label_map_from_path(
                    seg_path,
                    image_size=self.image_size,
                    supervision_mode=self.seg_supervision,
                )
            else:
                seg_target_labels = build_segmentation_label_map(
                    img,
                    image_size=self.image_size,
                    supervision_mode=self.seg_supervision,
                    bg_distance_threshold=self.seg_bg_distance_threshold,
                    min_line_intensity=self.seg_min_line_intensity,
                )

        valid_lines = extract_valid_lines(self.formatter.extract_assistant_lines(sample))
        assistant_text, coord_values, coord_line_ids, coord_point_ids, num_lines = self.formatter.lines_to_text_and_coords(
            valid_lines
        )
        centerline_heatmap = build_centerline_heatmap(
            valid_lines,
            image_size=self.image_size,
            line_width=self.centerline_heatmap_line_width,
            blur_radius=self.centerline_heatmap_blur_radius,
        )
        user_text = self.formatter.build_user_text()
        prompt_text = self.formatter.apply_chat_template(
            tokenizer=self.tokenizer,
            system_text=self.formatter.system_prompt,
            user_text=user_text,
            assistant_text=None,
            add_generation_prompt=True,
        )
        full_text = self.formatter.apply_chat_template(
            tokenizer=self.tokenizer,
            system_text=self.formatter.system_prompt,
            user_text=user_text,
            assistant_text=assistant_text,
            add_generation_prompt=False,
        )
        return RawRCCenterlineSample(
            sample_id=sample_id,
            image_path=image_path,
            pixel_values=pixel_values,
            seg_target_labels=seg_target_labels,
            centerline_heatmap=centerline_heatmap,
            prompt_text=str(prompt_text),
            full_text=str(full_text),
            coord_points=list(coord_values),
            coord_line_ids=list(coord_line_ids),
            coord_point_ids=list(coord_point_ids),
            num_lines=int(num_lines),
        )


class RCCenterlinePrefixCollator:
    def __init__(
        self,
        tokenizer: Any,
        cutoff_len: int,
        num_visual_tokens: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.cutoff_len = int(cutoff_len)
        self.num_visual_tokens = int(num_visual_tokens)
        self.vis_patch_token_id = int(tokenizer.convert_tokens_to_ids("<vis_patch>"))
        self.coord_token_id = int(tokenizer.convert_tokens_to_ids("<coord_pt>"))
        if self.vis_patch_token_id < 0 or self.coord_token_id < 0:
            raise ValueError("Special tokens must be registered before collator construction.")

    def __call__(self, features: Sequence[RawRCCenterlineSample]) -> Dict[str, torch.Tensor]:
        prompt_texts = [item.prompt_text for item in features]
        full_texts = [item.full_text for item in features]
        pixel_values = torch.stack([item.pixel_values for item in features], dim=0)
        seg_target_labels = torch.stack([item.seg_target_labels for item in features], dim=0)
        centerline_heatmap = torch.stack([item.centerline_heatmap for item in features], dim=0)

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

        coord_target_values = torch.full((*input_ids.shape, 2), fill_value=-1.0, dtype=torch.float32)
        coord_target_line_ids = torch.full(input_ids.shape, fill_value=-1, dtype=torch.long)
        coord_target_point_ids = torch.full(input_ids.shape, fill_value=-1, dtype=torch.long)
        vis_patch_mask = input_ids.eq(self.vis_patch_token_id)

        for batch_idx, item in enumerate(features):
            full_len = int(attention_mask[batch_idx].sum().item())
            prompt_text_len = int(prompt_batch["attention_mask"][batch_idx].sum().item())
            full_text_len = int(full_batch["attention_mask"][batch_idx].sum().item())
            prompt_len = max(0, full_len - max(0, full_text_len - prompt_text_len))
            labels[batch_idx, :prompt_len] = -100

            coord_positions = [
                token_idx
                for token_idx in range(prompt_len, full_len)
                if int(input_ids[batch_idx, token_idx].item()) == self.coord_token_id
            ]
            if len(coord_positions) > len(item.coord_points):
                raise ValueError(
                    f"Coordinate placeholder mismatch for {item.sample_id}: "
                    f"positions={len(coord_positions)} values={len(item.coord_points)}"
                )
            kept_coord_points = item.coord_points[: len(coord_positions)]
            kept_line_ids = item.coord_line_ids[: len(coord_positions)]
            kept_point_ids = item.coord_point_ids[: len(coord_positions)]
            for coord_idx, token_idx in enumerate(coord_positions):
                coord_target_values[batch_idx, token_idx] = torch.tensor(
                    kept_coord_points[coord_idx],
                    dtype=torch.float32,
                )
                coord_target_line_ids[batch_idx, token_idx] = int(kept_line_ids[coord_idx])
                coord_target_point_ids[batch_idx, token_idx] = int(kept_point_ids[coord_idx])
            num_vis = int(vis_patch_mask[batch_idx].sum().item())
            if num_vis != self.num_visual_tokens:
                raise ValueError(
                    f"Visual token mismatch for {item.sample_id}: expected={self.num_visual_tokens} actual={num_vis}"
                )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "seg_target_labels": seg_target_labels,
            "centerline_heatmap": centerline_heatmap,
            "coord_target_values": coord_target_values,
            "coord_target_line_ids": coord_target_line_ids,
            "coord_target_point_ids": coord_target_point_ids,
            "vis_patch_mask": vis_patch_mask,
        }
