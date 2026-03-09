import glob
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .serialization import MapSequenceTokenizer, serialize_annotation


@dataclass
class NuScenesSDMapDatasetConfig:
    nuscenes_root: str
    temporal_pkl_path: str
    sdmap_root: str
    satmap_root: str
    image_size: int
    use_pv: bool
    pv_camera: str
    pv_num_frames: int
    pv_image_size: List[int]
    max_samples: Optional[int]
    sample_interval_meter: float
    meter_range_half: float
    max_lines: int
    max_points_per_line: int
    categories: List[str]
    max_seq_len: int
    coord_num_bins: Optional[int]
    angle_num_bins: int
    train_augment: bool
    aug_rot90_prob: float
    aug_hflip_prob: float
    aug_vflip_prob: float


class NuScenesSDMapDataset(Dataset):
    def __init__(self, cfg: NuScenesSDMapDatasetConfig) -> None:
        self.cfg = cfg
        with open(cfg.temporal_pkl_path, "rb") as f:
            raw = pickle.load(f)
        infos = raw.get("infos", [])
        self.info_by_token = {x.get("token", ""): x for x in infos}

        self.sdmap_by_token = self._build_sdmap_index(cfg.sdmap_root)
        self.items: List[Dict] = []
        for info in infos:
            token = str(info.get("token", ""))
            if not token:
                continue
            sat_path = os.path.join(cfg.satmap_root, f"{token}_satellite.png")
            ann_path = self.sdmap_by_token.get(token, "")
            if not os.path.exists(sat_path) or not ann_path:
                continue
            self.items.append(
                {
                    "token": token,
                    "sat_path": sat_path,
                    "ann_path": ann_path,
                    "cams": info.get("cams", {}),
                    "prev_token": info.get("prev", ""),
                }
            )

        if cfg.max_samples is not None and cfg.max_samples > 0:
            self.items = self.items[: int(cfg.max_samples)]

        self.tokenizer = MapSequenceTokenizer(
            image_size=cfg.image_size,
            categories=cfg.categories,
            max_seq_len=cfg.max_seq_len,
            coord_num_bins=cfg.coord_num_bins,
            angle_num_bins=cfg.angle_num_bins,
        )

    @staticmethod
    def _build_sdmap_index(sdmap_root: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        paths = glob.glob(os.path.join(sdmap_root, "**", "*.pkl"), recursive=True)
        for p in paths:
            name = os.path.basename(p)
            if len(name) == 36:
                tok = name[:-4]
                out[tok] = p
        return out

    def __len__(self) -> int:
        return len(self.items)

    def _read_sdmap_annotation(self, ann_path: str) -> Dict[str, List[np.ndarray]]:
        with open(ann_path, "rb") as f:
            x = pickle.load(f)
        if not isinstance(x, dict):
            return {c: [] for c in self.cfg.categories}
        out = {c: [] for c in self.cfg.categories}
        for c in self.cfg.categories:
            arrs = x.get(c, [])
            if not isinstance(arrs, list):
                continue
            keep = []
            for a in arrs:
                aa = np.asarray(a, dtype=np.float32)
                if aa.ndim == 2 and aa.shape[0] >= 2 and aa.shape[1] == 2:
                    keep.append(aa)
            out[c] = keep
        return out

    def _load_single_pv_image(self, rel_path: str, h: int, w: int) -> torch.Tensor:
        if rel_path.startswith("./data/nuscenes/"):
            rel_path = rel_path[len("./data/nuscenes/") :]
        pv_path = os.path.join(self.cfg.nuscenes_root, rel_path)
        if os.path.exists(pv_path):
            img = Image.open(pv_path).convert("RGB")
            img = img.resize((w, h), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return torch.zeros((3, h, w), dtype=torch.float32)

    def _load_pv_images(self, item: Dict) -> torch.Tensor:
        h, w = int(self.cfg.pv_image_size[0]), int(self.cfg.pv_image_size[1])
        max_l = max(1, int(self.cfg.pv_num_frames))
        frame_tokens = [item["token"]]
        prev_token = item.get("prev_token", "")
        while len(frame_tokens) < max_l and prev_token:
            prev_info = self.info_by_token.get(prev_token)
            if prev_info is None:
                break
            frame_tokens.append(prev_token)
            prev_token = prev_info.get("prev", "")
        frame_tokens = list(reversed(frame_tokens))

        frames: List[torch.Tensor] = []
        for tok in frame_tokens:
            info = self.info_by_token.get(tok)
            cams = {} if info is None else info.get("cams", {})
            cam = cams.get(self.cfg.pv_camera, {}) if isinstance(cams, dict) else {}
            rel_path = cam.get("data_path", "") if isinstance(cam, dict) else ""
            if rel_path:
                frames.append(self._load_single_pv_image(rel_path=rel_path, h=h, w=w))
            else:
                frames.append(torch.zeros((3, h, w), dtype=torch.float32))
        while len(frames) < max_l:
            frames.insert(0, torch.zeros((3, h, w), dtype=torch.float32))
        return torch.stack(frames[:max_l], dim=0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        img = Image.open(item["sat_path"]).convert("RGB")
        img = img.resize((self.cfg.image_size, self.cfg.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        ann = self._read_sdmap_annotation(item["ann_path"])
        lines = serialize_annotation(
            annotation=ann,
            categories=self.cfg.categories,
            image_size=self.cfg.image_size,
            interval_meter=self.cfg.sample_interval_meter,
            max_lines=self.cfg.max_lines,
            max_points_per_line=self.cfg.max_points_per_line,
            meter_range_half=self.cfg.meter_range_half,
        )
        if self.cfg.train_augment:
            img_t, lines = self._apply_augment(img_t, lines)

        cur_ids = self.tokenizer.encode_lines(lines)
        token_ids = torch.tensor(cur_ids, dtype=torch.long)
        out = {
            "image": img_t,
            "tokens": token_ids,
            "current_tokens": token_ids.clone(),
            "current_start_idx": torch.tensor(1, dtype=torch.long),
            "token": item["token"],
        }
        if self.cfg.use_pv:
            out["pv_images"] = self._load_pv_images(item)
        return out

    def _apply_augment(self, image: torch.Tensor, lines: List[Dict]):
        size = int(self.cfg.image_size)
        out_lines = []
        for line in lines:
            out_lines.append(
                {
                    "category": line["category"],
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
