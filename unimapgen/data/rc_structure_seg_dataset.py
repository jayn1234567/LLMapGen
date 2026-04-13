from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from unimapgen.data.rc_centerline_cnn_prefix_dataset import (
    build_segmentation_label_map,
    index_rows_by_id,
    load_jsonl,
    load_segmentation_label_map_from_path,
    resize_rgb_image,
)


@dataclass
class RCStructureSegDatasetConfig:
    dataset_jsonl: str
    dataset_meta_jsonl: str
    media_dir: str
    image_size: int = 512
    mask_size: int = 512
    supervision_mode: str = "binary"
    max_samples: Optional[int] = None
    train_augment: bool = False
    aug_rot90_prob: float = 0.0
    aug_hflip_prob: float = 0.0
    aug_vflip_prob: float = 0.0


class RCStructureSegDataset(Dataset):
    def __init__(self, cfg: RCStructureSegDatasetConfig) -> None:
        self.cfg = cfg
        self.media_dir = Path(cfg.media_dir).resolve()
        # 主 jsonl 提供样本列表，meta_jsonl 提供预计算 mask 路径等补充信息。
        self.rows = load_jsonl(Path(cfg.dataset_jsonl).resolve(), max_samples=int(cfg.max_samples or 0))
        self.meta_rows = load_jsonl(Path(cfg.dataset_meta_jsonl).resolve())
        self.meta_by_id = index_rows_by_id(self.meta_rows)
        self.image_size = int(cfg.image_size)
        self.mask_size = int(cfg.mask_size)
        self.supervision_mode = str(cfg.supervision_mode).strip().lower()

    def _precomputed_mask_rel_path(self, meta: Dict[str, Any]) -> str:
        if self.supervision_mode == "binary":
            return str(meta.get("seg_binary", "")).strip()
        if self.supervision_mode == "structure_multiclass":
            return str(meta.get("seg_structure_multiclass", "")).strip()
        return ""

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        sample_id = str(row.get("id", idx))
        meta = self.meta_by_id.get(sample_id, {})
        rel_image = str(row.get("images", [""])[0])
        image_path = (self.media_dir / rel_image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found for sample {sample_id}: {image_path}")

        with Image.open(image_path) as img:
            rgb = resize_rgb_image(img, image_size=self.image_size)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
            image = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

            seg_rel = self._precomputed_mask_rel_path(meta)
            seg_path = (self.media_dir / seg_rel).resolve() if seg_rel else None
            use_precomputed_mask = seg_path is not None and seg_path.is_file()
            if use_precomputed_mask:
                # 正式训练优先读预导出的 mask，保证训练和评估看到完全一致的像素监督。
                mask = load_segmentation_label_map_from_path(
                    seg_path,
                    image_size=self.mask_size,
                    supervision_mode=self.supervision_mode,
                )
            else:
                # 最小版仓库允许从源图即时重建 mask，方便在没有完整导出物时直接验证链路。
                mask = build_segmentation_label_map(
                    img,
                    image_size=self.mask_size,
                    supervision_mode=self.supervision_mode,
                )

        if self.cfg.train_augment:
            image, mask = self._apply_augment(image, mask)

        if self.supervision_mode == "binary":
            mask = mask.float().unsqueeze(0)
        else:
            mask = mask.long()

        return {
            "image": image,
            "mask": mask,
            "sample_id": sample_id,
            "image_rel_path": rel_image,
        }

    def _apply_augment(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 这里只做图像和 mask 完全同步的几何增强，避免像素监督错位。
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


def rc_structure_seg_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image": torch.stack([x["image"] for x in batch], dim=0),
        "mask": torch.stack([x["mask"] for x in batch], dim=0),
        "sample_ids": [str(x["sample_id"]) for x in batch],
        "image_rel_paths": [str(x["image_rel_path"]) for x in batch],
    }
