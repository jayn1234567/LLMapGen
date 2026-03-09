import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .serialization import MapSequenceTokenizer, serialize_opensatmap_lines


@dataclass
class OpenSatMapDatasetConfig:
    opensatmap_root: str
    ann_json_path: str
    split: str
    image_size: int
    max_samples: Optional[int]
    sample_interval_meter: float
    meter_per_pixel: float
    max_lines: int
    max_points_per_line: int
    categories: List[str]
    line_types: List[str]
    max_seq_len: int
    coord_num_bins: Optional[int]
    angle_num_bins: int
    train_augment: bool
    aug_rot90_prob: float
    aug_hflip_prob: float
    aug_vflip_prob: float
    split_dir: Optional[str] = None


class OpenSatMapDataset(Dataset):
    def __init__(self, cfg: OpenSatMapDatasetConfig) -> None:
        self.cfg = cfg
        with open(cfg.ann_json_path, "r", encoding="utf-8") as f:
            ann_dict = json.load(f)
        if not isinstance(ann_dict, dict):
            raise ValueError(f"Annotation file must be dict-json: {cfg.ann_json_path}")

        split_root = cfg.split_dir or os.path.join(cfg.opensatmap_root, "picuse20trainvaltest")
        split_dir = os.path.join(split_root, cfg.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"OpenSatMap split dir not found: {split_dir}")

        image_names = sorted(os.listdir(split_dir))
        self.items: List[Dict] = []
        for name in image_names:
            ann = ann_dict.get(name)
            if not isinstance(ann, dict):
                continue
            img_path = os.path.join(split_dir, name)
            if not os.path.isfile(img_path):
                continue
            self.items.append(
                {
                    "token": name,
                    "img_path": img_path,
                    "raw_lines": list(ann.get("lines", [])),
                    "src_w": int(ann.get("image_width", 4096)),
                    "src_h": int(ann.get("image_height", 4096)),
                }
            )

        if cfg.max_samples is not None and cfg.max_samples > 0:
            self.items = self.items[: int(cfg.max_samples)]

        self.tokenizer = MapSequenceTokenizer(
            image_size=cfg.image_size,
            categories=cfg.categories,
            line_types=cfg.line_types,
            max_seq_len=cfg.max_seq_len,
            coord_num_bins=cfg.coord_num_bins,
            angle_num_bins=cfg.angle_num_bins,
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        img = Image.open(item["img_path"]).convert("RGB")
        img = img.resize((self.cfg.image_size, self.cfg.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        lines = serialize_opensatmap_lines(
            raw_lines=item["raw_lines"],
            categories=self.cfg.categories,
            line_types=self.cfg.line_types,
            src_w=int(item["src_w"]),
            src_h=int(item["src_h"]),
            image_size=self.cfg.image_size,
            interval_meter=float(self.cfg.sample_interval_meter),
            meter_per_pixel=float(self.cfg.meter_per_pixel),
            max_lines=int(self.cfg.max_lines),
            max_points_per_line=int(self.cfg.max_points_per_line),
        )
        if self.cfg.train_augment:
            img_t, lines = self._apply_augment(image=img_t, lines=lines)

        cur_ids = self.tokenizer.encode_lines(lines)
        token_ids = torch.tensor(cur_ids, dtype=torch.long)

        out = {
            "image": img_t,
            "tokens": token_ids,
            "current_tokens": token_ids.clone(),
            "current_start_idx": torch.tensor(1, dtype=torch.long),
            "token": item["token"],
        }
        return out

    def _apply_augment(self, image: torch.Tensor, lines: List[Dict]):
        size = int(self.cfg.image_size)
        out_lines = []
        for line in lines:
            out_lines.append(
                {
                    "category": line["category"],
                    "line_type": line.get("line_type", ""),
                    "start_type": line.get("start_type", "start"),
                    "end_type": line.get("end_type", "end"),
                    "points": np.asarray(line.get("points", []), dtype=np.float32).copy(),
                }
            )

        if np.random.rand() < float(self.cfg.aug_rot90_prob):
            k = int(np.random.randint(1, 4))
            image = torch.rot90(image, k=k, dims=(1, 2))
            for line in out_lines:
                pts = line["points"]
                if pts.ndim != 2 or pts.shape[0] == 0:
                    continue
                x = pts[:, 0].copy()
                y = pts[:, 1].copy()
                if k == 1:
                    pts[:, 0] = y
                    pts[:, 1] = (size - 1) - x
                elif k == 2:
                    pts[:, 0] = (size - 1) - x
                    pts[:, 1] = (size - 1) - y
                else:
                    pts[:, 0] = (size - 1) - y
                    pts[:, 1] = x

        if np.random.rand() < float(self.cfg.aug_hflip_prob):
            image = torch.flip(image, dims=(2,))
            for line in out_lines:
                pts = line["points"]
                if pts.ndim == 2 and pts.shape[0] > 0:
                    pts[:, 0] = (size - 1) - pts[:, 0]

        if np.random.rand() < float(self.cfg.aug_vflip_prob):
            image = torch.flip(image, dims=(1,))
            for line in out_lines:
                pts = line["points"]
                if pts.ndim == 2 and pts.shape[0] > 0:
                    pts[:, 1] = (size - 1) - pts[:, 1]

        return image, out_lines
