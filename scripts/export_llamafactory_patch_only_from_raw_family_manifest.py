import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_PROMPT_TEMPLATE = """<image>
Please construct the complete road map in the current satellite patch."""

DEFAULT_SYSTEM_PROMPT = (
    "You are a road-map reconstruction assistant for satellite-image patches.\n"
    "Predict the complete road map in the current patch from the satellite image.\n"
    "Return only valid JSON in the required schema.\n"
    "Do not output markdown fences or extra explanation.\n"
    "Keep all coordinates in the patch-local coordinate system."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export LLaMAFactory patch-only SFT data from raw OpenSatMap family manifests."
    )
    parser.add_argument("--ann-json", type=str, required=True)
    parser.add_argument("--family-manifest", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--max-families-per-split", type=int, default=0)
    parser.add_argument("--accepted-categories", type=str, nargs="+", default=["lane_line", "virtual_line", "curb"])
    parser.add_argument("--output-category", type=str, default="road")
    parser.add_argument("--resample-step-px", type=float, default=12.0)
    parser.add_argument("--boundary-tol-px", type=float, default=2.5)
    parser.add_argument("--export-visualizations", action="store_true")
    parser.add_argument("--viz-patch-ids", type=int, nargs="+", default=[])
    parser.add_argument("--use-system-prompt", action="store_true")
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--system-prompt-file", type=str, default="")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    count = 0
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_opensatmap_category(name: str) -> str:
    value = str(name).strip().lower()
    if value == "lane line":
        return "lane_line"
    if value == "virtual line":
        return "virtual_line"
    if value == "curb":
        return "curb"
    return value.replace(" ", "_")


def dedup_points(points: Sequence[np.ndarray], eps: float = 1e-3) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    out = [arr[0]]
    for idx in range(1, arr.shape[0]):
        if float(np.linalg.norm(arr[idx] - out[-1])) > float(eps):
            out.append(arr[idx])
    return np.asarray(out, dtype=np.float32)


def clamp_points(points_xy: np.ndarray, patch_size: int) -> np.ndarray:
    arr = np.asarray(points_xy, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    arr[:, 0] = np.clip(arr[:, 0], 0.0, float(patch_size - 1))
    arr[:, 1] = np.clip(arr[:, 1], 0.0, float(patch_size - 1))
    return arr


def simplify_for_json(points_xy: np.ndarray, patch_size: int) -> List[List[int]]:
    arr = clamp_points(points_xy, patch_size=patch_size)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return []
    arr = np.rint(arr).astype(np.int32)
    arr = dedup_points(arr.astype(np.float32)).astype(np.int32)
    return [[int(x), int(y)] for x, y in arr.tolist()]


def resample_polyline(points_xy: np.ndarray, step_px: float, max_points: Optional[int] = None) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return pts
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    total = float(np.sum(seg))
    if total < 1e-6:
        return pts[:1]
    step = max(float(step_px), 1.0)
    n = max(2, int(math.floor(total / step)) + 1)
    if max_points is not None:
        n = min(n, int(max_points))
    target = np.linspace(0.0, total, n, dtype=np.float32)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    sampled: List[np.ndarray] = []
    for dist in target:
        j = int(np.searchsorted(cum, dist, side="right") - 1)
        j = min(max(j, 0), len(seg) - 1)
        t0 = float(cum[j])
        t1 = float(cum[j + 1])
        ratio = 0.0 if t1 <= t0 else (float(dist) - t0) / (t1 - t0)
        sampled.append(pts[j] * (1.0 - ratio) + pts[j + 1] * ratio)
    return dedup_points(sampled)


def point_in_rect(point_xy: np.ndarray, rect: Tuple[float, float, float, float], eps: float = 1e-6) -> bool:
    x_min, y_min, x_max, y_max = rect
    x = float(point_xy[0])
    y = float(point_xy[1])
    return (x_min - eps) <= x <= (x_max + eps) and (y_min - eps) <= y <= (y_max + eps)


def clip_segment_liang_barsky(
    p0: np.ndarray,
    p1: np.ndarray,
    rect: Tuple[float, float, float, float],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    x_min, y_min, x_max, y_max = rect
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    p = [-dx, dx, -dy, dy]
    q = [float(p0[0] - x_min), float(x_max - p0[0]), float(p0[1] - y_min), float(y_max - p0[1])]
    u1 = 0.0
    u2 = 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-8:
            if qi < 0.0:
                return None
            continue
        t = qi / pi
        if pi < 0.0:
            if t > u2:
                return None
            if t > u1:
                u1 = t
        else:
            if t < u1:
                return None
            if t < u2:
                u2 = t
    c0 = np.asarray([p0[0] + u1 * dx, p0[1] + u1 * dy], dtype=np.float32)
    c1 = np.asarray([p0[0] + u2 * dx, p0[1] + u2 * dy], dtype=np.float32)
    return c0, c1


def clip_polyline_to_rect(points_xy: np.ndarray, rect: Tuple[float, float, float, float]) -> List[np.ndarray]:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return []
    pieces: List[np.ndarray] = []
    current: List[np.ndarray] = []
    for idx in range(pts.shape[0] - 1):
        clipped = clip_segment_liang_barsky(pts[idx], pts[idx + 1], rect)
        if clipped is None:
            if len(current) >= 2:
                pieces.append(dedup_points(current))
            current = []
            continue
        c0, c1 = clipped
        if not current:
            current = [c0, c1]
        else:
            if float(np.linalg.norm(current[-1] - c0)) <= 1e-3:
                current.append(c1)
            else:
                if len(current) >= 2:
                    pieces.append(dedup_points(current))
                current = [c0, c1]
        if not point_in_rect(pts[idx + 1], rect):
            if len(current) >= 2:
                pieces.append(dedup_points(current))
            current = []
    if len(current) >= 2:
        pieces.append(dedup_points(current))
    return [piece for piece in pieces if piece.shape[0] >= 2]


def point_boundary_side(point_xy: np.ndarray, rect: Tuple[float, float, float, float], tol_px: float) -> Optional[str]:
    x_min, y_min, x_max, y_max = rect
    x = float(point_xy[0])
    y = float(point_xy[1])
    if abs(x - x_min) <= tol_px:
        return "left"
    if abs(y - y_min) <= tol_px:
        return "top"
    if abs(x - x_max) <= tol_px:
        return "right"
    if abs(y - y_max) <= tol_px:
        return "bottom"
    return None


def point_origin_sort_key(point_xy: Sequence[float]) -> Tuple[float, float, float]:
    x = float(point_xy[0])
    y = float(point_xy[1])
    return (x * x + y * y, y, x)


def canonicalize_line_direction(
    points_xy: np.ndarray,
    start_type: str,
    end_type: str,
) -> Tuple[np.ndarray, str, str]:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return pts, start_type, end_type
    start_is_cut = str(start_type) == "cut"
    end_is_cut = str(end_type) == "cut"
    reverse = False
    if start_is_cut and not end_is_cut:
        reverse = False
    elif end_is_cut and not start_is_cut:
        reverse = True
    else:
        if point_origin_sort_key(pts[-1]) < point_origin_sort_key(pts[0]):
            reverse = True
    if not reverse:
        return pts, start_type, end_type
    return pts[::-1].copy(), end_type, start_type


def sort_lines(lines: List[Dict]) -> List[Dict]:
    return sorted(
        lines,
        key=lambda item: (*point_origin_sort_key(item.get("points", [[1e9, 1e9]])[0]),),
    )


def collect_global_lines(ann: Dict, accepted_categories: Sequence[str]) -> List[np.ndarray]:
    cat_set = set(str(x) for x in accepted_categories)
    out: List[np.ndarray] = []
    for rec in ann.get("lines", []):
        cat = normalize_opensatmap_category(rec.get("category", ""))
        if cat_set and cat not in cat_set:
            continue
        arr = np.asarray(rec.get("points", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
            continue
        out.append(dedup_points(arr))
    return out


def build_full_patch_segments_global(
    global_lines: Sequence[np.ndarray],
    patch_rect_global: Tuple[float, float, float, float],
    resample_step_px: float,
    boundary_tol_px: float,
) -> List[Dict]:
    out: List[Dict] = []
    for line in global_lines:
        pieces = clip_polyline_to_rect(line, patch_rect_global)
        for piece in pieces:
            piece = resample_polyline(piece, step_px=resample_step_px)
            if piece.ndim != 2 or piece.shape[0] < 2:
                continue
            start_side = point_boundary_side(piece[0], patch_rect_global, boundary_tol_px)
            end_side = point_boundary_side(piece[-1], patch_rect_global, boundary_tol_px)
            start_type = "cut" if start_side is not None else "start"
            end_type = "cut" if end_side is not None else "end"
            piece, start_type, end_type = canonicalize_line_direction(piece, start_type=start_type, end_type=end_type)
            out.append(
                {
                    "points_global": piece,
                    "start_type": start_type,
                    "end_type": end_type,
                }
            )
    return out


def build_full_patch_target_lines(
    full_segments_global: Sequence[Dict],
    patch: Dict,
    output_category: str,
) -> List[Dict]:
    crop_box = patch["crop_box"]
    patch_size = int(crop_box["x_max"] - crop_box["x_min"])
    out: List[Dict] = []
    offset = np.asarray([crop_box["x_min"], crop_box["y_min"]], dtype=np.float32)[None, :]
    for segment in full_segments_global:
        local = np.asarray(segment["points_global"], dtype=np.float32) - offset
        local = clamp_points(local, patch_size=patch_size)
        if local.ndim != 2 or local.shape[0] < 2:
            continue
        points_json = simplify_for_json(local, patch_size=patch_size)
        if len(points_json) < 2:
            continue
        out.append(
            {
                "category": output_category,
                "start_type": str(segment["start_type"]),
                "end_type": str(segment["end_type"]),
                "points": points_json,
            }
        )
    return sort_lines([x for x in out if len(x.get("points", [])) >= 2])


def draw_endpoint(draw: ImageDraw.ImageDraw, point: Sequence[int], color: Tuple[int, int, int], radius: int = 3) -> None:
    x = int(point[0])
    y = int(point[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def save_visualization(
    patch_image: Image.Image,
    target_lines: Sequence[Dict],
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)
    image = patch_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    patch_size = int(image.size[0])
    draw.rectangle((0, 0, patch_size - 1, patch_size - 1), outline=(255, 0, 180), width=2)
    for line in target_lines:
        pts = [tuple(int(v) for v in p) for p in line.get("points", [])]
        if len(pts) >= 2:
            draw.line(pts, fill=(40, 220, 255), width=3)
            draw_endpoint(draw, pts[0], (0, 180, 220), 3)
            draw_endpoint(draw, pts[-1], (0, 180, 220), 3)
    image.save(out_path)


def to_message_record_with_system(
    image_rel_path: str,
    target_lines: Sequence[Dict],
    sample_id: str,
    system_prompt: str,
) -> Dict:
    target_json = json.dumps({"lines": list(target_lines)}, ensure_ascii=False, separators=(",", ":"))
    messages: List[Dict] = []
    if str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": DEFAULT_PROMPT_TEMPLATE})
    messages.append({"role": "assistant", "content": target_json})
    return {
        "id": sample_id,
        "messages": messages,
        "images": [image_rel_path],
    }


def export_split(
    split: str,
    families: Sequence[Dict],
    annotations: Dict,
    output_root: Path,
    max_families_per_split: int,
    accepted_categories: Sequence[str],
    output_category: str,
    resample_step_px: float,
    boundary_tol_px: float,
    export_visualizations: bool,
    viz_patch_ids: Sequence[int],
    system_prompt: str,
) -> Dict[str, int]:
    rows: List[Dict] = []
    meta_rows: List[Dict] = []
    family_count = 0
    for family in families:
        if str(family.get("split")) != split:
            continue
        family_count += 1
        if max_families_per_split > 0 and family_count > max_families_per_split:
            break
        image_name = str(family["source_image"])
        ann = annotations.get(image_name)
        if not isinstance(ann, dict):
            continue
        global_lines = collect_global_lines(ann=ann, accepted_categories=accepted_categories)
        with Image.open(str(family["source_image_path"])) as raw_img:
            raw_image = raw_img.convert("RGB")
            patches = sorted(list(family["patches"]), key=lambda x: int(x["patch_id"]))
            for patch in patches:
                patch_id = int(patch["patch_id"])
                crop_box = patch["crop_box"]
                patch_rect_global = (
                    float(crop_box["x_min"]),
                    float(crop_box["y_min"]),
                    float(crop_box["x_max"]),
                    float(crop_box["y_max"]),
                )
                patch_image = raw_image.crop(
                    (
                        int(crop_box["x_min"]),
                        int(crop_box["y_min"]),
                        int(crop_box["x_max"]),
                        int(crop_box["y_max"]),
                    )
                )
                full_segments_global = build_full_patch_segments_global(
                    global_lines=global_lines,
                    patch_rect_global=patch_rect_global,
                    resample_step_px=resample_step_px,
                    boundary_tol_px=boundary_tol_px,
                )
                target_lines = build_full_patch_target_lines(
                    full_segments_global=full_segments_global,
                    patch=patch,
                    output_category=output_category,
                )

                image_rel = Path("images") / split / str(family["family_id"]) / f"p{patch_id:02d}.png"
                out_image = output_root / image_rel
                ensure_dir(out_image.parent)
                patch_image.save(out_image)

                if bool(export_visualizations) and (not viz_patch_ids or patch_id in viz_patch_ids):
                    save_visualization(
                        patch_image=patch_image,
                        target_lines=target_lines,
                        out_path=output_root / "visualizations" / split / str(family["family_id"]) / f"p{patch_id:02d}.png",
                    )

                sample_id = f"{family['family_id']}_p{patch_id:02d}"
                rows.append(
                    to_message_record_with_system(
                        image_rel_path=image_rel.as_posix(),
                        target_lines=target_lines,
                        sample_id=sample_id,
                        system_prompt=system_prompt,
                    )
                )
                meta_rows.append(
                    {
                        "id": sample_id,
                        "split": split,
                        "family_id": family["family_id"],
                        "source_image": family["source_image"],
                        "patch_id": patch_id,
                        "row": int(patch["row"]),
                        "col": int(patch["col"]),
                        "scan_index": patch_id,
                        "image": image_rel.as_posix(),
                        "crop_box": crop_box,
                        "num_target_lines": len(target_lines),
                        "target_mode": "full_patch_map",
                        "coord_system": "patch_local_896",
                        "serialization_mode": "paper_structured",
                        "line_direction_mode": "canonical_cut_then_origin",
                        "line_sort_mode": "first_point_distance_to_patch_origin",
                        "resample_mode": "equal_distance",
                        "resample_step_px": float(resample_step_px),
                        "has_system_prompt": bool(str(system_prompt).strip()),
                        "target_lines": target_lines,
                    }
                )
    count_main = write_jsonl(output_root / f"{split}.jsonl", rows)
    count_meta = write_jsonl(output_root / f"meta_{split}.jsonl", meta_rows)
    return {
        "families": family_count if max_families_per_split <= 0 else min(family_count, max_families_per_split),
        "samples": count_main,
        "meta_samples": count_meta,
    }


def main() -> None:
    args = parse_args()
    annotations = load_json(Path(args.ann_json).resolve())
    families = load_jsonl(Path(args.family_manifest).resolve())
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    system_prompt = ""
    if str(args.system_prompt_file).strip():
        system_prompt = Path(args.system_prompt_file).resolve().read_text(encoding="utf-8").strip()
    elif str(args.system_prompt).strip():
        system_prompt = str(args.system_prompt).strip()
    elif bool(args.use_system_prompt):
        system_prompt = DEFAULT_SYSTEM_PROMPT

    summary: Dict[str, Dict[str, int]] = {}
    for split in args.splits:
        summary[str(split)] = export_split(
            split=str(split),
            families=families,
            annotations=annotations,
            output_root=output_root,
            max_families_per_split=int(args.max_families_per_split),
            accepted_categories=[str(x) for x in args.accepted_categories],
            output_category=str(args.output_category),
            resample_step_px=float(args.resample_step_px),
            boundary_tol_px=float(args.boundary_tol_px),
            export_visualizations=bool(args.export_visualizations),
            viz_patch_ids=[int(x) for x in args.viz_patch_ids],
            system_prompt=system_prompt,
        )

    dataset_info = {
        "dataset_name": "unimapgen_paper16_patch_only_sft",
        "source_ann_json": str(Path(args.ann_json).resolve()),
        "source_family_manifest": str(Path(args.family_manifest).resolve()),
        "target_mode": "full_patch_map",
        "coord_system": "patch_local_896",
        "serialization_mode": "paper_structured",
        "line_direction_mode": "canonical_cut_then_origin",
        "line_sort_mode": "first_point_distance_to_patch_origin",
        "resample_mode": "equal_distance",
        "resample_step_px": float(args.resample_step_px),
        "use_system_prompt": bool(system_prompt),
        "system_prompt": system_prompt,
        "summary": summary,
    }
    with (output_root / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    for split, info in summary.items():
        print(f"[{split}] families={info['families']} samples={info['samples']} meta={info['meta_samples']}")
    print(f"Saved dataset to {output_root}")


if __name__ == "__main__":
    main()
