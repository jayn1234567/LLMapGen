#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mllm.coord_utils import COORD_MODE_PIXEL, convert_payload_text, record_coord_config
from scripts.tools.map_visualization import (
    count_invalid_geometry,
    render_whole_map_visualizations,
    resolve_image_path as resolve_map_image_path,
    sanitize_points,
)


def load_json_maybe(text: str):
    try:
        return json.loads(text)
    except Exception:
        return []


def normalize_lines(payload):
    if isinstance(payload, dict) and isinstance(payload.get("lines"), list):
        return payload["lines"]
    if isinstance(payload, list):
        return payload
    return []


def first_text(record: dict, keys: list[str], default: str = "[]") -> str:
    for key in keys:
        value = record.get(key)
        if value:
            return value
    return default


def payload_for_draw(record: dict, pixel_keys: list[str], raw_keys: list[str]):
    pixel_text = first_text(record, pixel_keys, default="")
    if pixel_text:
        return load_json_maybe(pixel_text)
    raw_text = first_text(record, raw_keys)
    coord_cfg = record_coord_config(record, default_mode=COORD_MODE_PIXEL)
    if coord_cfg["coord_mode"] != COORD_MODE_PIXEL:
        try:
            raw_text = convert_payload_text(
                raw_text,
                coord_cfg["coord_mode"],
                COORD_MODE_PIXEL,
                coord_cfg["patch_width"],
                coord_cfg["patch_height"],
                coord_range=coord_cfg["coord_range"],
                clamp=True,
            )
        except Exception:
            pass
    return load_json_maybe(raw_text)


def payload_has_intersection(payload) -> bool:
    return any(
        str(item.get("category", "centerline")).strip().lower() == "intersection"
        for item in normalize_lines(payload)
        if isinstance(item, dict)
    )


def record_has_intersection(record: dict) -> bool:
    gt_payload = payload_for_draw(record, ["ground_truth_pixel", "labels_pixel"], ["ground_truth", "labels"])
    pred_payload = payload_for_draw(record, ["prediction_json_pixel", "response_pixel", "prediction_pixel"], ["prediction_json", "response", "prediction"])
    return payload_has_intersection(gt_payload) or payload_has_intersection(pred_payload)


def record_has_ground_truth(record: dict) -> bool:
    return any(record.get(key) for key in ("ground_truth", "labels", "ground_truth_pixel", "labels_pixel"))


def safe_output_name(value) -> str:
    text = str(value or "record")
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text) or "record"


def warn_invalid_geometry(record_id: str, label: str, payload) -> None:
    invalid_lines, invalid_points = count_invalid_geometry(payload)
    if invalid_lines or invalid_points:
        print(
            f"[WARN] {record_id} {label}: skipped "
            f"{invalid_lines} malformed lines and {invalid_points} malformed points"
        )


def draw_map_lines(image: Image.Image, payload, centerline_color: tuple, intersection_color: tuple, width: int = 3) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for item in normalize_lines(payload):
        if not isinstance(item, dict):
            continue
        xy_points = sanitize_points(item.get("points", []))
        if not xy_points:
            continue
        category = str(item.get("category", "centerline")).lower()
        color = intersection_color if category == "intersection" else centerline_color
        for i in range(len(xy_points) - 1):
            draw.line([xy_points[i], xy_points[i + 1]], fill=color, width=width)
        for x, y in xy_points:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
    return image


def resolve_image_path(raw_path: str, image_folder: Path) -> Path:
    return resolve_map_image_path(raw_path, image_folder)


def add_title(image: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 40), "black")
    canvas.paste(image, (0, 40))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 8), text, fill="white", font=font)
    return canvas


def records_from_payload(payload) -> list:
    if isinstance(payload, dict):
        for key in ("patch_results", "results", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    if isinstance(payload, list):
        records = []
        for item in payload:
            if isinstance(item, dict):
                records.extend(records_from_payload(item))
        return records
    return []


def load_json_array_or_lines(file_path: Path) -> list:
    """读取 JSON 数组格式 或 JSON Lines 格式（每行一个 JSON 对象）"""
    content = file_path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    if content.startswith('{') and content.endswith('}'):
        try:
            payload = json.loads(content)
            return records_from_payload(payload)
        except json.JSONDecodeError:
            pass
    # 如果以 '[' 开头且以 ']' 结尾，视为标准 JSON 数组
    if content.startswith('[') and content.endswith(']'):
        try:
            return records_from_payload(json.loads(content))
        except json.JSONDecodeError:
            pass  # 解析失败则尝试按行处理
    # 按行解析 JSON Lines
    results = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            results.extend(records_from_payload(json.loads(line)))
        except json.JSONDecodeError:
            # 安静跳过无效行，可改为打印警告
            continue
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--color-gt", default="green")
    parser.add_argument("--color-pred", default="red")
    parser.add_argument("--map-task", choices=["auto", "lane", "lane_intersection"], default="auto")
    parser.add_argument("--no-eval-centerline", action="store_true")
    parser.add_argument("--eval-output-json", default="")
    parser.add_argument("--eval-meter-per-pixel", type=float, default=0.2)
    parser.add_argument("--eval-buffer-size", type=float, default=1.0)
    parser.add_argument("--eval-match-threshold", type=float, default=0.33)
    parser.add_argument("--whole-map-viz-dir", default="", help="Directory for stitched whole-map visualizations. Defaults to input-dir/whole_map_viz.")
    parser.add_argument("--skip-whole-map-viz", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0, help="Visualize at most this many records; 0 means all.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    image_folder = Path(args.image_folder)
    summary_path = input_dir / "summary.json"
    if not summary_path.exists():
        summary_jsonl_path = input_dir / "summary.jsonl"
        if summary_jsonl_path.exists():
            summary_path = summary_jsonl_path
        else:
            raise FileNotFoundError(f"Summary file not found: {summary_path}")

    results = load_json_array_or_lines(summary_path)
    if not results:
        raise ValueError(f"No valid records found in {summary_path}")
    if args.max_samples > 0:
        results = results[: args.max_samples]

    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "green": (0, 255, 0),
        "red": (255, 0, 0),
        "blue": (0, 128, 255),
        "yellow": (255, 255, 0),
    }
    gt_color = colors.get(args.color_gt, colors["green"])
    pred_color = colors.get(args.color_pred, colors["red"])

    for result in results:
        record_id = str(result.get("record_id") or result.get("id") or result.get("image") or "record")
        try:
            image_path = resolve_map_image_path(result.get("image", "") or result.get("image_path", ""), image_folder, result)
            if not image_path.exists():
                print(f"[WARN] Image not found for {record_id}: {image_path}")
                continue

            base_image = Image.open(image_path).convert("RGB")
            gt_payload = payload_for_draw(result, ["ground_truth_pixel", "labels_pixel"], ["ground_truth", "labels"])
            pred_payload = payload_for_draw(result, ["prediction_json_pixel", "response_pixel", "prediction_pixel"], ["prediction_json", "response", "prediction"])
            warn_invalid_geometry(record_id, "ground_truth", gt_payload)
            warn_invalid_geometry(record_id, "prediction", pred_payload)
            gt_image = draw_map_lines(base_image.copy(), gt_payload, gt_color, colors["yellow"])
            pred_image = draw_map_lines(base_image.copy(), pred_payload, pred_color, colors["blue"])

            gt_panel = add_title(gt_image, "Ground Truth")
            pred_panel = add_title(pred_image, "Prediction")

            merged = Image.new("RGB", (gt_panel.width + pred_panel.width + 10, gt_panel.height), "black")
            merged.paste(gt_panel, (0, 0))
            merged.paste(pred_panel, (gt_panel.width + 10, 0))

            out_path = output_dir / f"{safe_output_name(record_id or image_path.stem)}_compare.png"
            merged.save(out_path)
            print(f"Saved: {out_path}")
        except Exception as exc:
            print(f"[WARN] Visualization failed for {record_id}: {type(exc).__name__}: {exc}")
            continue

    print(f"Done. Visualizations saved to {output_dir}")
    if not args.skip_whole_map_viz:
        whole_map_viz_dir = Path(args.whole_map_viz_dir) if args.whole_map_viz_dir else input_dir / "whole_map_viz"
        try:
            rendered = render_whole_map_visualizations(results, image_folder, whole_map_viz_dir)
            print(json.dumps({"whole_map_viz_dir": str(whole_map_viz_dir), "whole_map_visualizations": rendered}, ensure_ascii=False))
        except Exception as exc:
            print(f"[WARN] Whole-map visualization failed: {type(exc).__name__}: {exc}")
    if not args.no_eval_centerline and any(record_has_ground_truth(result) for result in results):
        from infer_index.line_eval import (
            evaluate_records,
            evaluate_lane_intersection_records,
            print_eval_table,
            print_lane_intersection_eval_tables,
        )

        eval_kwargs = dict(
            meter_per_pixel=args.eval_meter_per_pixel,
            buffer_size=args.eval_buffer_size,
            match_threshold=args.eval_match_threshold,
        )
        map_task = args.map_task
        if map_task == "auto":
            map_task = "lane_intersection" if any(record_has_intersection(result) for result in results) else "lane"
        if map_task == "lane_intersection":
            map_eval = evaluate_lane_intersection_records(results, **eval_kwargs)
            eval_summary = {
                "centerline_eval": map_eval["lane"],
                "intersection_eval": map_eval["intersection"],
                "lane_intersection_eval": map_eval["lane_intersection"],
                "map_eval": map_eval,
            }
        else:
            eval_summary = evaluate_records(results, **eval_kwargs)
        eval_path = Path(args.eval_output_json) if args.eval_output_json else input_dir / "eval.json"
        eval_path.write_text(json.dumps(eval_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if map_task == "lane_intersection":
            print_lane_intersection_eval_tables(eval_summary["map_eval"])
        else:
            print_eval_table(eval_summary)
        print(json.dumps({"centerline_eval_json": str(eval_path), "centerline_eval": eval_summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
