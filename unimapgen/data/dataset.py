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
class DatasetConfig:
    nuscenes_root: str
    pkl_dir: str
    satmap_root: str
    split: str
    image_size: int
    use_pv: bool
    pv_camera: str
    pv_num_frames: int
    pv_image_size: List[int]
    max_samples: Optional[int]
    sample_interval_meter: float
    max_lines: int
    max_points_per_line: int
    categories: List[str]
    max_seq_len: int
    coord_num_bins: Optional[int]
    angle_num_bins: int
    use_state_update: bool
    state_update_mode: str
    state_prefix_mode: str
    use_text_prompt: bool
    text_prompt_mode: str
    text_num_trace_points: int
    train_augment: bool
    aug_rot90_prob: float
    aug_hflip_prob: float
    aug_vflip_prob: float


class NuScenesSatelliteMapDataset(Dataset):
    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg
        pkl_path = os.path.join(cfg.pkl_dir, f"nuscenes_map_infos_temporal_{cfg.split}.pkl")
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
        infos = raw["infos"]
        self.info_by_token = {x.get("token", ""): x for x in infos}

        self.items: List[Dict] = []
        for info in infos:
            token = info["token"]
            sat_path = os.path.join(cfg.satmap_root, f"{token}_satellite.png")
            if not os.path.exists(sat_path):
                continue
            ann = info.get("annotation", {})
            if not isinstance(ann, dict):
                continue
            self.items.append(
                {
                    "token": token,
                    "sat_path": sat_path,
                    "annotation": ann,
                    "cams": info.get("cams", {}),
                    "prev_token": info.get("prev", ""),
                    "scene_token": info.get("scene_token", ""),
                    "map_location": info.get("map_location", ""),
                    "ego_xy": tuple((info.get("ego2global_translation") or [0.0, 0.0])[:2]),
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
        self.prompt_type_to_id = {"full_map": 0, "target_xy": 1, "trace_points": 2, "pv_assisted_target": 3}
        # For state update: quickly locate previous state annotation by token.
        self.item_by_token = {x["token"]: x for x in self.items}
        self.scan_prev_by_token = {}
        if cfg.use_state_update and cfg.state_update_mode == "patch_scan":
            self.scan_prev_by_token = self._build_patch_scan_prev_map()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        img = Image.open(item["sat_path"]).convert("RGB")
        img = img.resize((self.cfg.image_size, self.cfg.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        lines = serialize_annotation(
            annotation=item["annotation"],
            categories=self.cfg.categories,
            image_size=self.cfg.image_size,
            interval_meter=self.cfg.sample_interval_meter,
            max_lines=self.cfg.max_lines,
            max_points_per_line=self.cfg.max_points_per_line,
        )
        if self.cfg.train_augment:
            img_t, lines = self._apply_augment(image=img_t, lines=lines)
        current_token_ids = self.tokenizer.encode_lines(lines)

        # Build prefix + current sequence.
        # Prefix can include: previous map state tokens and/or text prompt tokens.
        full_ids = [self.tokenizer.bos_id]
        if self.cfg.use_state_update:
            prev_ref = item["prev_token"]
            if self.cfg.state_update_mode == "patch_scan":
                prev_ref = self.scan_prev_by_token.get(item["token"], "")
            prev_token_ids = self._build_prev_state_tokens(prev_ref)
            if len(prev_token_ids) > 2:
                full_ids.extend(prev_token_ids[1:-1])
            full_ids.append(self.tokenizer.state_id)
        prompt_token_ids = []
        if self.cfg.use_text_prompt:
            prompt_token_ids = self._build_text_prompt_tokens(lines)
            full_ids.extend(prompt_token_ids)
        prompt_len = len(full_ids)
        full_ids.extend(current_token_ids[1:])
        full_ids = full_ids[: self.cfg.max_seq_len]
        token_ids = torch.tensor(full_ids, dtype=torch.long)
        current_start_idx = min(prompt_len, token_ids.shape[0] - 1)

        current_tokens = torch.tensor(current_token_ids, dtype=torch.long)
        out = {
            "image": img_t,
            "tokens": token_ids,
            "current_tokens": current_tokens,
            "current_start_idx": torch.tensor(current_start_idx, dtype=torch.long),
            "token": item["token"],
        }
        if self.cfg.use_pv:
            pv_images = self._load_pv_images(item)
            out["pv_images"] = pv_images
        if self.cfg.use_text_prompt:
            prompt_id = self.prompt_type_to_id.get(self.cfg.text_prompt_mode, 0)
            out["prompt_type"] = torch.tensor(prompt_id, dtype=torch.long)
            out["prompt_tokens"] = torch.tensor(prompt_token_ids, dtype=torch.long)
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

        # Build a short temporal chain: [oldest ... current].
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
            cam = cams.get(self.cfg.pv_camera, {})
            rel_path = cam.get("data_path", "")
            if rel_path:
                frames.append(self._load_single_pv_image(rel_path=rel_path, h=h, w=w))
            else:
                frames.append(torch.zeros((3, h, w), dtype=torch.float32))

        # Left-pad with empty frames if the temporal chain is short.
        while len(frames) < max_l:
            frames.insert(0, torch.zeros((3, h, w), dtype=torch.float32))
        return torch.stack(frames[:max_l], dim=0)  # [L, C, H, W]

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

        # Random 90-degree rotations (paper-inspired lightweight approximation).
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

    def _build_prev_state_tokens(self, prev_token: str) -> List[int]:
        if not prev_token:
            return [self.tokenizer.bos_id, self.tokenizer.eos_id]
        prev_item = self.item_by_token.get(prev_token)
        if prev_item is None:
            return [self.tokenizer.bos_id, self.tokenizer.eos_id]
        prev_lines = serialize_annotation(
            annotation=prev_item["annotation"],
            categories=self.cfg.categories,
            image_size=self.cfg.image_size,
            interval_meter=self.cfg.sample_interval_meter,
            max_lines=self.cfg.max_lines,
            max_points_per_line=self.cfg.max_points_per_line,
        )
        prev_lines = self._filter_prefix_lines(prev_lines)
        return self.tokenizer.encode_lines(prev_lines)

    def _build_text_prompt_tokens(self, lines: List[Dict]) -> List[int]:
        mode = self.cfg.text_prompt_mode
        if mode == "target_xy":
            line = self._select_prompt_line(lines)
            if line is None:
                return [self.tokenizer.txt_xy_id, self.tokenizer.txt_end_id]
            pts = np.asarray(line.get("points", []), dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 2:
                return [self.tokenizer.txt_xy_id, self.tokenizer.txt_end_id]
            return self.tokenizer.encode_text_target_xy(pts[0], pts[-1])
        if mode in {"trace_points", "pv_assisted_target"}:
            points, angles = self._build_trace_prompt_points(lines, int(self.cfg.text_num_trace_points))
            return self.tokenizer.encode_text_trace_points(points_xy=points, angles_deg=angles)
        # full_map: minimal terminal token so model can learn this mode explicitly.
        return [self.tokenizer.txt_end_id]

    def _select_prompt_line(self, lines: List[Dict]):
        if len(lines) == 0:
            return None
        best = None
        best_len = -1.0
        for line in lines:
            pts = np.asarray(line.get("points", []), dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
            ln = float(np.sum(seg))
            if ln > best_len:
                best = line
                best_len = ln
        return best

    def _build_trace_prompt_points(self, lines: List[Dict], n_points: int):
        n_points = max(2, n_points)
        line = self._select_prompt_line(lines)
        if line is None:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        pts = np.asarray(line.get("points", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        idx = np.linspace(0, pts.shape[0] - 1, num=min(n_points, pts.shape[0]), dtype=np.int32)
        sel = pts[idx]
        angles = []
        for i in idx:
            i0 = max(0, i - 1)
            i1 = min(pts.shape[0] - 1, i + 1)
            v = pts[i1] - pts[i0]
            a = (np.degrees(np.arctan2(-v[1], v[0])) + 360.0) % 360.0
            angles.append(a)
        return sel, np.asarray(angles, dtype=np.float32)

    def _build_patch_scan_prev_map(self) -> Dict[str, str]:
        """
        Approximate left->right, top->bottom scan order by global XY:
        - group by scene/map location
        - sort by y descending, then x ascending
        """
        groups: Dict[str, List[Dict]] = {}
        for item in self.items:
            scene_key = item.get("scene_token", "")
            city = item.get("map_location", "")
            key = f"{city}::{scene_key}"
            groups.setdefault(key, []).append(item)

        prev_map: Dict[str, str] = {}
        for _, arr in groups.items():
            arr.sort(key=lambda z: (-float(z["ego_xy"][1]), float(z["ego_xy"][0])))
            for i, cur in enumerate(arr):
                prev_map[cur["token"]] = arr[i - 1]["token"] if i > 0 else ""
        return prev_map

    def _filter_prefix_lines(self, lines: List[Dict]) -> List[Dict]:
        mode = self.cfg.state_prefix_mode
        if mode == "all":
            return lines
        if mode == "cut_only":
            filtered = [x for x in lines if x.get("start_type") == "cut" or x.get("end_type") == "cut"]
            # Avoid empty prefix collapse in sparse cases.
            if len(filtered) == 0:
                return lines[: min(8, len(lines))]
            return filtered
        if mode == "cut_points":
            points = []
            for x in lines:
                arr = np.asarray(x.get("points", []), dtype=np.float32)
                if arr.ndim != 2 or arr.shape[0] == 0:
                    continue
                cat = x.get("category", "divider")
                if x.get("start_type") == "cut":
                    points.append(
                        {
                            "category": cat,
                            "start_type": "cut",
                            "end_type": "cut",
                            "points": arr[:1],
                        }
                    )
                if x.get("end_type") == "cut":
                    points.append(
                        {
                            "category": cat,
                            "start_type": "cut",
                            "end_type": "cut",
                            "points": arr[-1:],
                        }
                    )
            if len(points) == 0:
                return lines[: min(4, len(lines))]
            return points[: self.cfg.max_lines]
        return lines


def collate_fn(batch: List[Dict], pad_id: int) -> Dict[str, torch.Tensor]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    lengths = [b["tokens"].shape[0] for b in batch]
    max_len = max(lengths)
    tokens = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    # loss_mask aligns with target side tokens[:, 1:].
    loss_mask = torch.zeros((len(batch), max_len - 1), dtype=torch.bool)
    for i, b in enumerate(batch):
        cur = b["tokens"]
        tokens[i, : cur.shape[0]] = cur
        start = int(b.get("current_start_idx", torch.tensor(1)).item())
        # Target index j predicts tokens[:, j+1], so current starts at target start-1.
        t_start = max(0, start - 1)
        t_end = max(0, cur.shape[0] - 1)
        if t_end > t_start:
            loss_mask[i, t_start:t_end] = True

    cur_lens = [b["current_tokens"].shape[0] for b in batch]
    cur_max = max(cur_lens)
    current_tokens = torch.full((len(batch), cur_max), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        current_tokens[i, : b["current_tokens"].shape[0]] = b["current_tokens"]

    out = {
        "image": images,
        "tokens": tokens,
        "current_tokens": current_tokens,
        "loss_mask": loss_mask,
        "lengths": torch.tensor(lengths),
        "token_strs": [b["token"] for b in batch],
    }
    if "pv_images" in batch[0]:
        out["pv_images"] = torch.stack([b["pv_images"] for b in batch], dim=0)  # [B, L, C, H, W]
    if "prompt_type" in batch[0]:
        out["prompt_type"] = torch.stack([b["prompt_type"] for b in batch], dim=0)  # [B]
    if "prompt_tokens" in batch[0]:
        p_lens = [b["prompt_tokens"].shape[0] for b in batch]
        p_max = max(p_lens) if len(p_lens) > 0 else 1
        p_tok = torch.full((len(batch), p_max), pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            cur = b["prompt_tokens"]
            p_tok[i, : cur.shape[0]] = cur
        out["prompt_tokens"] = p_tok
    return out
