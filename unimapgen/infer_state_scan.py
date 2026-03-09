import argparse
import json
import math
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from unimapgen.data.serialization import MapSequenceTokenizer, pixel_to_world, serialize_annotation, world_to_pixel
from unimapgen.models import build_model_from_cfg
from unimapgen.utils import ensure_dir, load_yaml


def quat_to_yaw(q) -> float:
    # nuScenes quaternion is [w, x, y, z].
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def local_to_global(points_local: np.ndarray, ego_xy: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points_local @ rot.T + ego_xy[None, :]


def global_to_local(points_global: np.ndarray, ego_xy: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    rot_t = np.array([[c, s], [-s, c]], dtype=np.float32)  # R(-yaw)
    return (points_global - ego_xy[None, :]) @ rot_t.T


def load_infos(cfg, split: str):
    pkl_path = os.path.join(cfg["data"]["nuscenes_map_pkl_dir"], f"nuscenes_map_infos_temporal_{split}.pkl")
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    infos = raw["infos"]
    sat_root = cfg["data"]["satmap_root"]
    valid = []
    for info in infos:
        token = info["token"]
        sat_path = os.path.join(sat_root, f"{token}_satellite.png")
        if not os.path.exists(sat_path):
            continue
        valid.append(info)
    return valid


def build_scene_scan_order(infos: List[Dict]) -> Dict[str, List[Dict]]:
    groups = defaultdict(list)
    for info in infos:
        city = info.get("map_location", "")
        scene = info.get("scene_token", "")
        groups[f"{city}::{scene}"].append(info)
    for key in groups:
        groups[key].sort(
            key=lambda z: (
                -float((z.get("ego2global_translation") or [0.0, 0.0])[1]),
                float((z.get("ego2global_translation") or [0.0, 0.0])[0]),
            )
        )
    return groups


def load_image_tensor(path: str, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def project_global_lines_to_current_patch(
    global_lines: List[Dict],
    ego_xy: np.ndarray,
    yaw: float,
    image_size: int,
    max_lines: int,
) -> List[Dict]:
    out = []
    border_tol = max(2.0, float(image_size) * 0.02)
    for line in global_lines:
        pts_g = np.asarray(line["points_global"], dtype=np.float32)
        if pts_g.shape[0] < 2:
            continue
        pts_local = global_to_local(pts_g, ego_xy=ego_xy, yaw=yaw)
        inside = (
            (pts_local[:, 0] >= -30.0)
            & (pts_local[:, 0] <= 30.0)
            & (pts_local[:, 1] >= -30.0)
            & (pts_local[:, 1] <= 30.0)
        )
        if int(inside.sum()) < 2:
            continue
        pts_local = pts_local[inside]
        pts_pix = world_to_pixel(pts_local, image_size=image_size)
        start_type = "cut" if _is_border_point(pts_pix[0], image_size=image_size, tol=border_tol) else "start"
        end_type = "cut" if _is_border_point(pts_pix[-1], image_size=image_size, tol=border_tol) else "end"
        out.append(
            {
                "category": line["category"],
                "line_type": line.get("line_type", ""),
                "start_type": start_type,
                "end_type": end_type,
                "points": pts_pix,
            }
        )
        if len(out) >= max_lines:
            break
    return out


def filter_prefix_lines(lines: List[Dict], mode: str, max_lines: int) -> List[Dict]:
    if mode == "cut_only":
        cut_lines = [x for x in lines if x.get("start_type") == "cut" or x.get("end_type") == "cut"]
        # Fallback: keep a tiny prefix instead of empty prefix.
        if len(cut_lines) == 0 and len(lines) > 0:
            cut_lines = lines[: min(4, len(lines))]
        lines = cut_lines
    elif mode == "cut_points":
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
                        "line_type": x.get("line_type", ""),
                        "start_type": "cut",
                        "end_type": "cut",
                        "points": arr[:1],
                    }
                )
            if x.get("end_type") == "cut":
                points.append(
                    {
                        "category": cat,
                        "line_type": x.get("line_type", ""),
                        "start_type": "cut",
                        "end_type": "cut",
                        "points": arr[-1:],
                    }
                )
        if len(points) == 0 and len(lines) > 0:
            points = lines[: min(4, len(lines))]
        lines = points
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    return lines


def _is_border_point(p: np.ndarray, image_size: int, tol: float) -> bool:
    x = float(p[0])
    y = float(p[1])
    lo = tol
    hi = (image_size - 1) - tol
    return x <= lo or x >= hi or y <= lo or y >= hi


def pred_lines_to_global(pred_lines: List[Dict], ego_xy: np.ndarray, yaw: float, image_size: int) -> List[Dict]:
    out = []
    for line in pred_lines:
        pts = np.asarray(line.get("points", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        pts_local = pixel_to_world(pts, image_size=image_size)
        pts_global = local_to_global(pts_local, ego_xy=ego_xy, yaw=yaw)
        out.append(
            {
                "category": line.get("category", "divider"),
                "line_type": line.get("line_type", ""),
                "start_type": line.get("start_type", "start"),
                "end_type": line.get("end_type", "end"),
                "points_global": pts_global.tolist(),
            }
        )
    return out


def _line_signature(points_global: List[List[float]], cell: float = 1.0) -> Tuple:
    if len(points_global) == 0:
        return tuple()
    arr = np.asarray(points_global, dtype=np.float32)
    p0 = np.round(arr[0] / cell).astype(np.int32)
    p1 = np.round(arr[-1] / cell).astype(np.int32)
    ln = int(arr.shape[0])
    # Use order-invariant endpoint key to catch reversed duplicates.
    a = tuple(p0.tolist())
    b = tuple(p1.tolist())
    if a > b:
        a, b = b, a
    return (a, b, ln)


def merge_global_lines(global_lines: List[Dict], new_lines: List[Dict], cell: float = 1.0) -> List[Dict]:
    existing = set()
    for line in global_lines:
        existing.add(
            (
                line.get("category", "unknown"),
                line.get("line_type", ""),
                _line_signature(line.get("points_global", []), cell=cell),
            )
        )
    for line in new_lines:
        sig = (
            line.get("category", "unknown"),
            line.get("line_type", ""),
            _line_signature(line.get("points_global", []), cell=cell),
        )
        if sig in existing:
            continue
        existing.add(sig)
        global_lines.append(line)
    return global_lines


def build_model(cfg, tokenizer, checkpoint_path: str, device: torch.device):
    model = build_model_from_cfg(cfg, vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--scene_limit", type=int, default=1)
    parser.add_argument("--max_patches_per_scene", type=int, default=16)
    parser.add_argument("--state_prefix_mode", type=str, default=None, choices=["all", "cut_only", "cut_points"])
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--min_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--output", type=str, default="outputs/state_scan_global.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    image_size = int(cfg["data"]["image_size"])
    scfg = cfg["serialization"]
    prefix_mode = args.state_prefix_mode or cfg["data"].get("state_prefix_mode", "all")
    dec = cfg.get("decode", {})
    min_new_tokens = int(args.min_new_tokens if args.min_new_tokens is not None else dec.get("min_new_tokens", 0))
    temperature = float(args.temperature if args.temperature is not None else dec.get("temperature", 1.0))
    top_k = int(args.top_k if args.top_k is not None else dec.get("top_k", 1))
    repetition_penalty = float(
        args.repetition_penalty if args.repetition_penalty is not None else dec.get("repetition_penalty", 1.0)
    )
    tokenizer = MapSequenceTokenizer(
        image_size=image_size,
        categories=scfg["categories"],
        line_types=list(scfg.get("line_types", [])),
        max_seq_len=int(scfg["max_seq_len"]),
        coord_num_bins=scfg.get("coord_num_bins"),
        angle_num_bins=int(scfg.get("angle_num_bins", 360)),
    )

    infos = load_infos(cfg, split=args.split)
    scene_groups = build_scene_scan_order(infos)
    scene_keys = list(scene_groups.keys())[: int(args.scene_limit)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, tokenizer=tokenizer, checkpoint_path=args.checkpoint, device=device)
    sat_root = cfg["data"]["satmap_root"]

    result = {"split": args.split, "scene_results": []}
    for sk in scene_keys:
        arr = scene_groups[sk][: int(args.max_patches_per_scene)]
        global_lines: List[Dict] = []
        patch_results = []
        for info in arr:
            token = info["token"]
            sat_path = os.path.join(sat_root, f"{token}_satellite.png")
            image = load_image_tensor(sat_path, image_size=image_size).to(device)

            ego_xy = np.asarray((info.get("ego2global_translation") or [0.0, 0.0])[:2], dtype=np.float32)
            yaw = quat_to_yaw(info.get("ego2global_rotation", [1.0, 0.0, 0.0, 0.0]))

            base_prev_lines = project_global_lines_to_current_patch(
                global_lines=global_lines,
                ego_xy=ego_xy,
                yaw=yaw,
                image_size=image_size,
                max_lines=int(scfg["max_lines"]),
            )

            used_mode = prefix_mode
            pred_lines = []
            prev_lines_local = []
            # Adaptive fallback for sparse prefix modes:
            # cut_points -> cut_only -> all
            trial_modes = [prefix_mode]
            if prefix_mode == "cut_points":
                trial_modes.extend(["cut_only", "all"])
            elif prefix_mode == "cut_only":
                trial_modes.append("all")

            for mode_try in trial_modes:
                prev_lines_local = filter_prefix_lines(base_prev_lines, mode=mode_try, max_lines=int(scfg["max_lines"]))
                prev_ids = tokenizer.encode_lines(prev_lines_local)
                prompt_ids = [tokenizer.bos_id]
                if len(prev_ids) > 2:
                    prompt_ids.extend(prev_ids[1:-1])
                prompt_ids.append(tokenizer.state_id)
                prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

                pred_ids = model.generate(
                    image=image,
                    bos_id=tokenizer.bos_id,
                    eos_id=tokenizer.eos_id,
                    max_new_tokens=max(16, min(int(args.max_new_tokens), int(scfg["max_seq_len"]) - prompt.shape[1])),
                    prompt_ids=prompt,
                    min_new_tokens=min_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )[0].detach().cpu().tolist()
                pred_lines = tokenizer.decode_to_lines(pred_ids)
                used_mode = mode_try
                if len(pred_lines) > 0:
                    break

            # Cold-start fallback: if state-conditioned generation is empty,
            # retry with BOS-only prompt to avoid whole-scene collapse.
            if len(pred_lines) == 0:
                bos_prompt = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
                pred_ids = model.generate(
                    image=image,
                    bos_id=tokenizer.bos_id,
                    eos_id=tokenizer.eos_id,
                    max_new_tokens=max(16, min(int(args.max_new_tokens), int(scfg["max_seq_len"]) - 1)),
                    prompt_ids=bos_prompt,
                    min_new_tokens=min_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )[0].detach().cpu().tolist()
                pred_lines = tokenizer.decode_to_lines(pred_ids)
                used_mode = f"{used_mode}+bos_fallback"
            cur_global = pred_lines_to_global(pred_lines, ego_xy=ego_xy, yaw=yaw, image_size=image_size)
            global_lines = merge_global_lines(global_lines, cur_global, cell=1.0)

            # Optional GT info for debugging.
            gt_lines = serialize_annotation(
                annotation=info.get("annotation", {}),
                categories=scfg["categories"],
                image_size=image_size,
                interval_meter=float(scfg["sample_interval_meter"]),
                max_lines=int(scfg["max_lines"]),
                max_points_per_line=int(scfg["max_points_per_line"]),
            )
            patch_results.append(
                {
                    "token": token,
                    "state_prefix_mode": used_mode,
                    "num_prev_lines": len(prev_lines_local),
                    "num_pred_lines": len(pred_lines),
                    "num_gt_lines": len(gt_lines),
                }
            )

        result["scene_results"].append(
            {
                "scene_key": sk,
                "num_patches": len(arr),
                "num_global_lines": len(global_lines),
                "patches": patch_results,
                "global_lines": global_lines,
            }
        )

    ensure_dir(os.path.dirname(args.output) or ".")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved state-scan result to {args.output}")


if __name__ == "__main__":
    main()
