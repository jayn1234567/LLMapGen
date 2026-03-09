import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .serialization import normalize_opensatmap_category


@dataclass
class LaneSegDatasetConfig:
    root_dir: str
    ann_json_path: str
    split: str
    image_size: int
    max_samples: Optional[int]
    positive_categories: List[str]
    mask_line_width: int
    train_augment: bool
    aug_rot90_prob: float
    aug_hflip_prob: float
    aug_vflip_prob: float


class LaneSegDataset(Dataset):
    def __init__(self, cfg: LaneSegDatasetConfig) -> None:
        self.cfg = cfg
        with open(cfg.ann_json_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        if not isinstance(ann, dict):
            raise ValueError(f"Annotation file must be dict-json: {cfg.ann_json_path}")

        split_dir = os.path.join(cfg.root_dir, cfg.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"LaneSeg split dir not found: {split_dir}")

        self.positive_categories = {normalize_opensatmap_category(x) for x in cfg.positive_categories}
        self.items: List[Dict] = []
        for name in sorted(os.listdir(split_dir)):
            img_path = os.path.join(split_dir, name)
            rec = ann.get(name)
            if not os.path.isfile(img_path) or not isinstance(rec, dict):
                continue
            self.items.append(
                {
                    "token": name,
                    "img_path": img_path,
                    "lines": list(rec.get("lines", [])),
                    "src_w": int(rec.get("image_width", cfg.image_size)),
                    "src_h": int(rec.get("image_height", cfg.image_size)),
                }
            )

        if cfg.max_samples is not None and int(cfg.max_samples) > 0:
            self.items = self.items[: int(cfg.max_samples)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        img = Image.open(item["img_path"]).convert("RGB")
        img = img.resize((self.cfg.image_size, self.cfg.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        image = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        mask = self._build_mask(
            raw_lines=item["lines"],
            src_w=int(item["src_w"]),
            src_h=int(item["src_h"]),
        )
        if self.cfg.train_augment:
            image, mask = self._apply_augment(image, mask)

        return {
            "image": image,
            "mask": mask.unsqueeze(0),
            "token": item["token"],
        }

    def _build_mask(self, raw_lines: Sequence[Dict], src_w: int, src_h: int) -> torch.Tensor:
        size = int(self.cfg.image_size)
        canvas = Image.new("L", (size, size), color=0)
        draw = ImageDraw.Draw(canvas)
        sx = float(max(1, size - 1)) / float(max(1, src_w - 1))
        sy = float(max(1, size - 1)) / float(max(1, src_h - 1))
        width = max(1, int(round(float(self.cfg.mask_line_width) * float(size) / float(max(1, src_w)))))

        for line in raw_lines:
            cat = normalize_opensatmap_category(line.get("category", ""))
            if cat not in self.positive_categories:
                continue
            pts = line.get("points", [])
            if not isinstance(pts, list) or len(pts) < 2:
                continue
            scaled = []
            for p in pts:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                scaled.append((float(p[0]) * sx, float(p[1]) * sy))
            if len(scaled) >= 2:
                draw.line(scaled, fill=1, width=width)

        mask = np.asarray(canvas, dtype=np.float32)
        return torch.from_numpy(mask).contiguous()

    def _apply_augment(self, image: torch.Tensor, mask: torch.Tensor):
        if np.random.rand() < float(self.cfg.aug_rot90_prob):
            k = int(np.random.randint(1, 4))
            image = torch.rot90(image, k=k, dims=(1, 2))
            mask = torch.rot90(mask, k=k, dims=(0, 1))

        if np.random.rand() < float(self.cfg.aug_hflip_prob):
            image = torch.flip(image, dims=(2,))
            mask = torch.flip(mask, dims=(1,))

        if np.random.rand() < float(self.cfg.aug_vflip_prob):
            image = torch.flip(image, dims=(1,))
            mask = torch.flip(mask, dims=(0,))
        return image, mask


def lane_seg_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "image": torch.stack([x["image"] for x in batch], dim=0),
        "mask": torch.stack([x["mask"] for x in batch], dim=0),
        "token_strs": [x["token"] for x in batch],
    }
