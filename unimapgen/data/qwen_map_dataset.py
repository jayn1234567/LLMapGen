import json
import os
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .serialization import MapSequenceTokenizer, serialize_opensatmap_lines
from unimapgen.state_geometry import (
    build_patch_scan_order,
    build_state_lines_from_global,
    load_patch_geometry_map,
    merge_global_lines,
    patch_lines_to_global,
)


def filter_state_prefix_lines(lines: List[Dict], prefix_mode: str, max_lines: int, trace_num_points: int = 3) -> List[Dict]:
    mode = str(prefix_mode)
    if mode == "all":
        out = list(lines)
    elif mode == "cut_only":
        out = [x for x in lines if x.get("start_type") == "cut" or x.get("end_type") == "cut"]
        if not out:
            out = list(lines[: min(8, len(lines))])
    elif mode == "cut_traces":
        out = []
        trace_num_points = max(2, int(trace_num_points))
        for x in lines:
            arr = np.asarray(x.get("points", []), dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] == 0:
                continue
            cat = x.get("category", "lane_line")
            line_type = x.get("line_type", "")
            if x.get("start_type") == "cut":
                seg = arr[: min(trace_num_points, arr.shape[0])].copy()
                out.append(
                    {
                        "category": cat,
                        "line_type": line_type,
                        "start_type": "cut",
                        "end_type": "cut",
                        "points": seg,
                    }
                )
            if x.get("end_type") == "cut":
                seg = arr[max(0, arr.shape[0] - trace_num_points) :].copy()[::-1]
                out.append(
                    {
                        "category": cat,
                        "line_type": line_type,
                        "start_type": "cut",
                        "end_type": "cut",
                        "points": seg,
                    }
                )
        if not out:
            out = list(lines[: min(4, len(lines))])
    elif mode == "cut_points":
        out = []
        for x in lines:
            arr = np.asarray(x.get("points", []), dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] == 0:
                continue
            cat = x.get("category", "lane_line")
            line_type = x.get("line_type", "")
            if x.get("start_type") == "cut":
                out.append(
                    {
                        "category": cat,
                        "line_type": line_type,
                        "start_type": "cut",
                        "end_type": "cut",
                        "points": arr[:1],
                    }
                )
            if x.get("end_type") == "cut":
                out.append(
                    {
                        "category": cat,
                        "line_type": line_type,
                        "start_type": "cut",
                        "end_type": "cut",
                        "points": arr[-1:],
                    }
                )
        if not out:
            out = list(lines[: min(4, len(lines))])
    else:
        out = list(lines)
    return out[: int(max_lines)]


def build_state_token_ids_from_lines(
    map_tokenizer: MapSequenceTokenizer,
    lines: List[Dict],
    prefix_mode: str,
    max_lines: int,
    trace_num_points: int = 3,
) -> List[int]:
    filtered = filter_state_prefix_lines(
        lines=lines,
        prefix_mode=prefix_mode,
        max_lines=max_lines,
        trace_num_points=trace_num_points,
    )
    prev_ids = map_tokenizer.encode_lines(filtered)
    out: List[int] = []
    if len(prev_ids) > 2:
        out.extend(int(x) for x in prev_ids[1:-1])
    out.append(int(map_tokenizer.state_id))
    return out


@dataclass
class OpenSatMapQwenDatasetConfig:
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
    prompt_template: str
    use_state_update: bool = False
    state_update_mode: str = "sample_prev"
    state_prefix_mode: str = "cut_points"
    splits_meta_path: Optional[str] = None
    patch_geometry_json: Optional[str] = None
    geometry_border_tol_px: float = 4.0
    geometry_overlap_margin_px: float = 32.0
    geometry_endpoint_margin_px: float = 96.0
    geometry_densify_step_m: float = 1.0
    geometry_connect_radius_m: float = 3.0
    geometry_trace_num_points: int = 3
    geometry_adjacent_source_margin_px: float = 96.0
    geometry_adjacent_center_margin_m: float = 96.0
    split_dir: Optional[str] = None


class OpenSatMapQwenDataset(Dataset):
    def __init__(self, cfg: OpenSatMapQwenDatasetConfig) -> None:
        self.cfg = cfg
        print(
            f"[QwenDataset] Loading split={cfg.split} ann_json={cfg.ann_json_path}",
            flush=True,
        )
        with open(cfg.ann_json_path, "r", encoding="utf-8") as f:
            ann_dict = json.load(f)
        if not isinstance(ann_dict, dict):
            raise ValueError(f"Annotation file must be dict-json: {cfg.ann_json_path}")

        split_root = cfg.split_dir or os.path.join(cfg.opensatmap_root, "picuse20trainvaltest")
        split_dir = os.path.join(split_root, cfg.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"OpenSatMap split dir not found: {split_dir}")

        image_names = sorted(os.listdir(split_dir))
        split_token_order = self._load_split_token_order(cfg=cfg, image_names=image_names)
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

        if split_token_order:
            order_map = {tok: i for i, tok in enumerate(split_token_order)}
            self.items.sort(key=lambda x: order_map.get(self._token_key(x["token"]), 10**9))

        if cfg.max_samples is not None and cfg.max_samples > 0:
            self.items = self.items[: int(cfg.max_samples)]
        print(
            f"[QwenDataset] Split={cfg.split} matched_items={len(self.items)} image_size={cfg.image_size} "
            f"use_state_update={bool(cfg.use_state_update)}",
            flush=True,
        )

        self.map_tokenizer = MapSequenceTokenizer(
            image_size=cfg.image_size,
            categories=cfg.categories,
            line_types=cfg.line_types,
            max_seq_len=cfg.max_seq_len,
            coord_num_bins=cfg.coord_num_bins,
            angle_num_bins=cfg.angle_num_bins,
        )
        categories_text = ", ".join(str(x).replace("_", " ") for x in cfg.categories)
        self.prompt_template = str(cfg.prompt_template)
        self.categories_text = categories_text
        self.line_types_text = ", ".join(str(x).replace("_", " ") for x in cfg.line_types)
        self.lines_by_token: Dict[str, List[Dict]] = {}
        for idx, item in enumerate(self.items, start=1):
            tok = self._token_key(item["token"])
            self.lines_by_token[tok] = serialize_opensatmap_lines(
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
            if len(self.items) >= 256 and (idx == 1 or idx % 256 == 0 or idx == len(self.items)):
                print(
                    f"[QwenDataset] Split={cfg.split} serialized_lines {idx}/{len(self.items)}",
                    flush=True,
                )
        self.prev_token_by_token = self._build_prev_token_map()
        self.geom_map = {}
        if cfg.patch_geometry_json and os.path.isfile(cfg.patch_geometry_json):
            self.geom_map = load_patch_geometry_map(cfg.patch_geometry_json)
            print(
                f"[QwenDataset] Split={cfg.split} loaded geometry records={len(self.geom_map)}",
                flush=True,
            )
        self.state_lines_by_token = self._load_or_build_state_lines_by_token()
        print(f"[QwenDataset] Split={cfg.split} dataset ready", flush=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        img = Image.open(item["img_path"]).convert("RGB")
        img = img.resize((self.cfg.image_size, self.cfg.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        image = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

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
            image, lines = self._apply_augment(image=image, lines=lines)

        map_token_ids = torch.tensor(self.map_tokenizer.encode_lines(lines), dtype=torch.long)
        prev_lines = self.state_lines_by_token.get(self._token_key(item["token"]), [])
        state_token_ids = self._build_prev_state_tokens(token=self._token_key(item["token"]))
        return {
            "image": image,
            "token": item["token"],
            "prompt_text": self._build_prompt_text(prev_lines=prev_lines),
            "map_token_ids": map_token_ids,
            "state_token_ids": torch.tensor(state_token_ids, dtype=torch.long),
            "lines": lines,
        }

    def _build_prev_state_tokens(self, token: str) -> List[int]:
        if not bool(self.cfg.use_state_update):
            return [self.map_tokenizer.state_id]
        prev_lines = self.state_lines_by_token.get(str(token), [])
        if not prev_lines:
            return [self.map_tokenizer.state_id]
        return build_state_token_ids_from_lines(
            map_tokenizer=self.map_tokenizer,
            lines=prev_lines,
            prefix_mode=str(self.cfg.state_prefix_mode),
            max_lines=int(self.cfg.max_lines),
            trace_num_points=int(self.cfg.geometry_trace_num_points),
        )

    def _build_state_lines_by_token(self) -> Dict[str, List[Dict]]:
        out: Dict[str, List[Dict]] = {}
        if not bool(self.cfg.use_state_update):
            return out
        ordered = [self._token_key(x["token"]) for x in self.items]
        if str(self.cfg.state_update_mode) == "patch_scan" and self.geom_map:
            global_lines = []
            total = len(ordered)
            for idx, tok in enumerate(ordered, start=1):
                geom_rec = self.geom_map.get(tok)
                if geom_rec is not None:
                    state_lines, _ = build_state_lines_from_global(
                        global_lines=global_lines,
                        geom_rec=geom_rec,
                        image_size=int(self.cfg.image_size),
                        meter_per_pixel=float(self.cfg.meter_per_pixel),
                        max_lines=int(self.cfg.max_lines),
                        border_tol_px=float(self.cfg.geometry_border_tol_px),
                        overlap_margin_px=float(self.cfg.geometry_overlap_margin_px),
                        endpoint_margin_px=float(self.cfg.geometry_endpoint_margin_px),
                        densify_step_m=float(self.cfg.geometry_densify_step_m),
                        trace_num_points=int(self.cfg.geometry_trace_num_points),
                        adjacent_source_margin_px=float(self.cfg.geometry_adjacent_source_margin_px),
                        adjacent_center_margin_m=float(self.cfg.geometry_adjacent_center_margin_m),
                    )
                    out[tok] = state_lines
                    cur_lines = self.lines_by_token.get(tok, [])
                    cur_global = patch_lines_to_global(
                        lines=cur_lines,
                        geom_rec=geom_rec,
                        image_size=int(self.cfg.image_size),
                        meter_per_pixel=float(self.cfg.meter_per_pixel),
                    )
                    global_lines = merge_global_lines(
                        global_lines=global_lines,
                        new_lines=cur_global,
                        cell_m=1.0,
                        connect_radius_m=float(self.cfg.geometry_connect_radius_m),
                    )
                else:
                    prev_tok = self.prev_token_by_token.get(tok, "")
                    out[tok] = self.lines_by_token.get(prev_tok, []) if prev_tok else []
                if total >= 256 and (idx == 1 or idx % 256 == 0 or idx == total):
                    print(
                        f"[QwenDataset] Split={self.cfg.split} build_state_lines {idx}/{total}",
                        flush=True,
                    )
            return out

        for tok in ordered:
            prev_tok = self.prev_token_by_token.get(tok, "")
            out[tok] = self.lines_by_token.get(prev_tok, []) if prev_tok else []
        return out

    def _load_or_build_state_lines_by_token(self) -> Dict[str, List[Dict]]:
        cache_path = self._state_lines_cache_path()
        if cache_path:
            cached = self._try_load_state_lines_cache(cache_path)
            if cached is not None:
                print(
                    f"[QwenDataset] Split={self.cfg.split} loaded state_lines cache={cache_path}",
                    flush=True,
                )
                return cached
        out = self._build_state_lines_by_token()
        if cache_path:
            self._write_state_lines_cache(cache_path=cache_path, state_lines_by_token=out)
        return out

    def _state_lines_cache_path(self) -> Optional[str]:
        if not bool(self.cfg.use_state_update):
            return None
        if str(self.cfg.state_update_mode) != "patch_scan":
            return None
        if not self.geom_map:
            return None
        cache_root = self.cfg.opensatmap_root or os.path.dirname(self.cfg.ann_json_path)
        if not cache_root:
            return None
        cache_dir = os.path.join(str(cache_root), ".cache")
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError:
            return None
        signature = self._state_lines_cache_signature()
        return os.path.join(cache_dir, f"state_lines_{self.cfg.split}_{signature}.json")

    def _state_lines_cache_signature(self) -> str:
        payload = {
            "split": str(self.cfg.split),
            "max_samples": int(self.cfg.max_samples) if self.cfg.max_samples is not None else None,
            "image_size": int(self.cfg.image_size),
            "sample_interval_meter": float(self.cfg.sample_interval_meter),
            "meter_per_pixel": float(self.cfg.meter_per_pixel),
            "max_lines": int(self.cfg.max_lines),
            "max_points_per_line": int(self.cfg.max_points_per_line),
            "categories": list(self.cfg.categories),
            "line_types": list(self.cfg.line_types),
            "state_update_mode": str(self.cfg.state_update_mode),
            "state_prefix_mode": str(self.cfg.state_prefix_mode),
            "geometry_border_tol_px": float(self.cfg.geometry_border_tol_px),
            "geometry_overlap_margin_px": float(self.cfg.geometry_overlap_margin_px),
            "geometry_endpoint_margin_px": float(self.cfg.geometry_endpoint_margin_px),
            "geometry_densify_step_m": float(self.cfg.geometry_densify_step_m),
            "geometry_connect_radius_m": float(self.cfg.geometry_connect_radius_m),
            "geometry_trace_num_points": int(self.cfg.geometry_trace_num_points),
            "geometry_adjacent_source_margin_px": float(self.cfg.geometry_adjacent_source_margin_px),
            "geometry_adjacent_center_margin_m": float(self.cfg.geometry_adjacent_center_margin_m),
            "tokens": [self._token_key(x["token"]) for x in self.items],
            "ann_json_path": str(self.cfg.ann_json_path),
            "patch_geometry_json": str(self.cfg.patch_geometry_json or ""),
        }
        for key, path in (
            ("ann_json_stat", self.cfg.ann_json_path),
            ("patch_geometry_stat", self.cfg.patch_geometry_json),
            ("splits_meta_stat", self.cfg.splits_meta_path),
        ):
            if path and os.path.isfile(path):
                st = os.stat(path)
                payload[key] = {
                    "size": int(st.st_size),
                    "mtime_ns": int(st.st_mtime_ns),
                }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:16]

    def _try_load_state_lines_cache(self, cache_path: str) -> Optional[Dict[str, List[Dict]]]:
        if not os.path.isfile(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        raw_state = obj.get("state_lines_by_token", obj)
        if not isinstance(raw_state, dict):
            return None
        out: Dict[str, List[Dict]] = {}
        for tok, lines in raw_state.items():
            if not isinstance(tok, str) or not isinstance(lines, list):
                continue
            out[tok] = self._deserialize_cached_lines(lines)
        return out

    def _write_state_lines_cache(self, cache_path: str, state_lines_by_token: Dict[str, List[Dict]]) -> None:
        payload = {
            "split": str(self.cfg.split),
            "state_update_mode": str(self.cfg.state_update_mode),
            "state_prefix_mode": str(self.cfg.state_prefix_mode),
            "state_lines_by_token": {
                str(tok): self._serialize_cached_lines(lines)
                for tok, lines in state_lines_by_token.items()
            },
        }
        tmp_path = f"{cache_path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp_path, cache_path)
            print(
                f"[QwenDataset] Split={self.cfg.split} saved state_lines cache={cache_path}",
                flush=True,
            )
        except Exception:
            try:
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _serialize_cached_lines(lines: List[Dict]) -> List[Dict]:
        out = []
        for line in lines:
            pts = np.asarray(line.get("points", []), dtype=np.float32)
            out.append(
                {
                    "category": line.get("category", "lane_line"),
                    "line_type": line.get("line_type", ""),
                    "start_type": line.get("start_type", "start"),
                    "end_type": line.get("end_type", "end"),
                    "points": pts.tolist(),
                }
            )
        return out

    @staticmethod
    def _deserialize_cached_lines(lines: List[Dict]) -> List[Dict]:
        out = []
        for line in lines:
            if not isinstance(line, dict):
                continue
            pts = np.asarray(line.get("points", []), dtype=np.float32)
            out.append(
                {
                    "category": line.get("category", "lane_line"),
                    "line_type": line.get("line_type", ""),
                    "start_type": line.get("start_type", "start"),
                    "end_type": line.get("end_type", "end"),
                    "points": pts,
                }
            )
        return out

    def _build_prev_token_map(self) -> Dict[str, str]:
        prev = {}
        ordered = [self._token_key(x["token"]) for x in self.items]
        for i, tok in enumerate(ordered):
            prev[tok] = ordered[i - 1] if i > 0 else ""
        return prev

    def _filter_prefix_lines(self, lines: List[Dict]) -> List[Dict]:
        return filter_state_prefix_lines(
            lines=lines,
            prefix_mode=str(self.cfg.state_prefix_mode),
            max_lines=int(self.cfg.max_lines),
            trace_num_points=int(self.cfg.geometry_trace_num_points),
        )

    def _build_prompt_text(self, prev_lines: List[Dict]) -> str:
        prefix_lines = self._filter_prefix_lines(prev_lines) if bool(self.cfg.use_state_update) else []
        if bool(self.cfg.use_state_update):
            if prefix_lines:
                state_instruction = (
                    "A serialized previous map state is provided before the current map tokens. "
                    "Tokens <s_cut> and <e_cut> mark connections crossing patch borders. "
                    "Use these cut endpoints to continue connectivity inside the current patch, "
                    "and avoid duplicating already-updated segments."
                )
            else:
                state_instruction = (
                    "No previous map state is available for this patch. "
                    "Start a new serialized map for the current patch."
                )
        else:
            state_instruction = "Generate the serialized vector map for the current patch."
        line_type_instruction = ""
        if self.cfg.line_types:
            line_type_instruction = (
                f"For each polyline, also output its line type using one of: {self.line_types_text}."
            )
        template_uses_state = "{state_instruction}" in self.prompt_template
        template_uses_line_type = "{line_type_instruction}" in self.prompt_template
        try:
            text = self.prompt_template.format(
                categories=self.categories_text,
                line_types=self.line_types_text,
                image_size=int(self.cfg.image_size),
                max_lines=int(self.cfg.max_lines),
                max_points_per_line=int(self.cfg.max_points_per_line),
                state_instruction=state_instruction,
                line_type_instruction=line_type_instruction,
                state_prefix_mode=str(self.cfg.state_prefix_mode),
                state_prefix_count=int(len(prefix_lines)),
                state_available="yes" if prefix_lines else "no",
            )
        except KeyError:
            text = self.prompt_template.format(
                categories=self.categories_text,
                line_types=self.line_types_text,
                image_size=int(self.cfg.image_size),
                max_lines=int(self.cfg.max_lines),
                max_points_per_line=int(self.cfg.max_points_per_line),
            )
            template_uses_state = False
            template_uses_line_type = False
        extras = []
        if state_instruction and not template_uses_state:
            extras.append(state_instruction)
        if line_type_instruction and not template_uses_line_type:
            extras.append(line_type_instruction)
        if extras:
            text = text.rstrip() + " " + " ".join(extras)
        return text

    def _load_split_token_order(self, cfg: OpenSatMapQwenDatasetConfig, image_names: List[str]) -> List[str]:
        if bool(cfg.use_state_update) and str(cfg.state_update_mode) == "patch_scan" and cfg.patch_geometry_json:
            if os.path.isfile(cfg.patch_geometry_json):
                geom_map = load_patch_geometry_map(cfg.patch_geometry_json)
                if geom_map:
                    return build_patch_scan_order(image_names, geom_map)
        cand = cfg.splits_meta_path
        if cand is None:
            cand = os.path.join(cfg.opensatmap_root, "splits_meta.json")
        if not os.path.isfile(cand):
            return []
        with open(cand, "r", encoding="utf-8") as f:
            meta = json.load(f)
        key = f"{cfg.split}_tokens"
        arr = meta.get(key, [])
        if not isinstance(arr, list):
            return []
        return [str(x) for x in arr]

    @staticmethod
    def _token_key(name: str) -> str:
        name = str(name)
        if name.endswith("_satellite.png"):
            return name[: -len("_satellite.png")]
        if name.endswith(".png"):
            return os.path.splitext(name)[0]
        return name

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


class QwenMapCollator:
    def __init__(self, qwen_map_tokenizer) -> None:
        self.qwen_map_tokenizer = qwen_map_tokenizer
        self.map_pad_id = int(qwen_map_tokenizer.map_tokenizer.pad_id)
        self.qwen_pad_id = int(qwen_map_tokenizer.pad_token_id)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = torch.stack([b["image"] for b in batch], dim=0)
        prompt_ids = [self.qwen_map_tokenizer.encode_prompt(b["prompt_text"]) for b in batch]
        state_qwen_ids = [
            self.qwen_map_tokenizer.encode_map_token_ids(b["state_token_ids"].tolist())
            for b in batch
        ]
        map_qwen_ids = [
            self.qwen_map_tokenizer.encode_map_token_ids(b["map_token_ids"].tolist())
            for b in batch
        ]

        prompt_max = max((len(x) for x in prompt_ids), default=0)
        state_max = max((len(x) for x in state_qwen_ids), default=0)
        map_max = max((len(x) for x in map_qwen_ids), default=0)

        prompt_tensor = torch.full((len(batch), prompt_max), self.qwen_pad_id, dtype=torch.long)
        prompt_mask = torch.zeros((len(batch), prompt_max), dtype=torch.long)
        state_tensor = torch.full((len(batch), state_max), self.qwen_pad_id, dtype=torch.long)
        state_mask = torch.zeros((len(batch), state_max), dtype=torch.long)
        map_tensor = torch.full((len(batch), map_max), self.qwen_pad_id, dtype=torch.long)
        map_mask = torch.zeros((len(batch), map_max), dtype=torch.long)
        gt_custom = torch.full((len(batch), map_max), self.map_pad_id, dtype=torch.long)
        state_gt_custom = torch.full((len(batch), state_max), self.map_pad_id, dtype=torch.long)

        for i, (p_ids, s_ids, m_ids, sample) in enumerate(zip(prompt_ids, state_qwen_ids, map_qwen_ids, batch)):
            if p_ids:
                prompt_tensor[i, : len(p_ids)] = torch.tensor(p_ids, dtype=torch.long)
                prompt_mask[i, : len(p_ids)] = 1
            if s_ids:
                state_tensor[i, : len(s_ids)] = torch.tensor(s_ids, dtype=torch.long)
                state_mask[i, : len(s_ids)] = 1
            if m_ids:
                map_tensor[i, : len(m_ids)] = torch.tensor(m_ids, dtype=torch.long)
                map_mask[i, : len(m_ids)] = 1
            state_custom_ids = sample["state_token_ids"]
            state_gt_custom[i, : state_custom_ids.shape[0]] = state_custom_ids
            custom_ids = sample["map_token_ids"]
            gt_custom[i, : custom_ids.shape[0]] = custom_ids

        return {
            "image": images,
            "prompt_input_ids": prompt_tensor,
            "prompt_attention_mask": prompt_mask,
            "state_input_ids": state_tensor,
            "state_attention_mask": state_mask,
            "map_input_ids": map_tensor,
            "map_attention_mask": map_mask,
            "gt_state_token_ids": state_gt_custom,
            "gt_map_token_ids": gt_custom,
            "token_strs": [b["token"] for b in batch],
            "gt_lines": [b["lines"] for b in batch],
            "prompt_texts": [b["prompt_text"] for b in batch],
        }
