#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_index.line_eval import evaluate_records, print_eval_table
from mllm.coord_utils import COORD_MODE_PIXEL, convert_payload_text, record_coord_config


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


def draw_map_lines(image: Image.Image, payload, centerline_color: tuple, intersection_color: tuple, width: int = 3) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for item in normalize_lines(payload):
        points = item.get("points", [])
        if not points:
            continue
        xy_points = [(int(pt[0]), int(pt[1])) for pt in points if isinstance(pt, list) and len(pt) == 2]
        category = str(item.get("category", "centerline")).lower()
        color = intersection_color if category == "intersection" else centerline_color
        for i in range(len(xy_points) - 1):
            draw.line([xy_points[i], xy_points[i + 1]], fill=color, width=width)
        for x, y in xy_points:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
    return image


def resolve_image_path(raw_path: str, image_folder: Path) -> Path:
    image_path = Path(raw_path)
    if image_path.is_absolute() and image_path.exists():
        return image_path
    candidate = image_folder / image_path
    if candidate.exists():
        return candidate
    fallback = image_folder / image_path.name
    if fallback.exists():
        return fallback
    return image_path


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


def load_json_array_or_lines(file_path: Path) -> list:
    """读取 JSON 数组格式 或 JSON Lines 格式（每行一个 JSON 对象）"""
    content = file_path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    if content.startswith('{') and content.endswith('}'):
        try:
            payload = json.loads(content)
            if isinstance(payload, dict) and isinstance(payload.get("patch_results"), list):
                return payload["patch_results"]
        except json.JSONDecodeError:
            pass
    # 如果以 '[' 开头且以 ']' 结尾，视为标准 JSON 数组
    if content.startswith('[') and content.endswith(']'):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass  # 解析失败则尝试按行处理
    # 按行解析 JSON Lines
    results = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
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
    parser.add_argument("--no-eval-centerline", action="store_true")
    parser.add_argument("--eval-output-json", default="")
    parser.add_argument("--eval-meter-per-pixel", type=float, default=0.2)
    parser.add_argument("--eval-buffer-size", type=float, default=1.0)
    parser.add_argument("--eval-match-threshold", type=float, default=0.33)
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
        image_path = resolve_image_path(result.get("image", ""), image_folder)
        if not image_path.exists():
            print(f"[WARN] Image not found: {image_path}")
            continue

        base_image = Image.open(image_path).convert("RGB")
        gt_payload = payload_for_draw(result, ["ground_truth_pixel", "labels_pixel"], ["ground_truth", "labels"])
        pred_payload = payload_for_draw(result, ["prediction_json_pixel", "response_pixel", "prediction_pixel"], ["prediction_json", "response", "prediction"])
        gt_image = draw_map_lines(base_image.copy(), gt_payload, gt_color, colors["yellow"])
        pred_image = draw_map_lines(base_image.copy(), pred_payload, pred_color, colors["blue"])

        gt_panel = add_title(gt_image, "Ground Truth")
        pred_panel = add_title(pred_image, "Prediction")

        merged = Image.new("RGB", (gt_panel.width + pred_panel.width + 10, gt_panel.height), "black")
        merged.paste(gt_panel, (0, 0))
        merged.paste(pred_panel, (gt_panel.width + 10, 0))

        record_id = result.get("record_id", image_path.stem)
        out_path = output_dir / f"{record_id}_compare.png"
        merged.save(out_path)
        print(f"Saved: {out_path}")

    print(f"Done. Visualizations saved to {output_dir}")
    if not args.no_eval_centerline and any("ground_truth" in result for result in results):
        eval_summary = evaluate_records(
            results,
            meter_per_pixel=args.eval_meter_per_pixel,
            buffer_size=args.eval_buffer_size,
            match_threshold=args.eval_match_threshold,
        )
        eval_path = Path(args.eval_output_json) if args.eval_output_json else input_dir / "eval.json"
        eval_path.write_text(json.dumps(eval_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print_eval_table(eval_summary)
        print(json.dumps({"centerline_eval_json": str(eval_path), "centerline_eval": eval_summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
