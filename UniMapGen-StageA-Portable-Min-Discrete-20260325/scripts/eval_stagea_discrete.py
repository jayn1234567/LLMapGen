from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unimapgen.data.discrete_stagea_dataset import (  # noqa: E402
    append_jsonl,
    build_prompt_conversation,
    extract_gt_lines,
    load_jsonl,
    sanitize_lines,
    select_items,
)
from unimapgen.discrete_map_token_format import DiscreteMapTokenFormatter  # noqa: E402
from unimapgen.models.qwen2_5vl_discrete import load_inference_model, load_processor  # noqa: E402
from unimapgen.paper_metrics import evaluate_prediction_items  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable Stage A discrete-token eval/infer/viz entry.")
    parser.add_argument("--dataset-jsonl", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--model-or-checkpoint", type=str, required=True)
    parser.add_argument("--processor-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--sample-ids", type=str, default="")
    parser.add_argument("--id-prefixes", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--meter-per-pixel", type=float, default=0.15)
    parser.add_argument("--line-width-px", type=int, default=6)
    parser.add_argument("--paper-categories", type=str, default="")
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--discrete-categories", type=str, default="road")
    parser.add_argument("--discrete-coord-num-bins", type=int, default=896)
    parser.add_argument("--discrete-token-schema", type=str, default="shared_numbers", choices=["legacy_xy", "shared_numbers"])
    parser.add_argument("--disable-legacy-text-prompt-tokens", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def sample_metrics(pred_lines: List[Dict[str, Any]], gt_lines: List[Dict[str, Any]], thresholds: Sequence[float]) -> Dict[str, float]:
    def to_np(line: Dict[str, Any]) -> np.ndarray:
        points = sanitize_lines([line])
        if not points:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(points[0]["points"], dtype=np.float32)

    def chamfer(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) == 0 or len(b) == 0:
            return 1e6
        d1 = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1))
        return float(d1.min(axis=1).mean() + d1.min(axis=0).mean()) * 0.5

    def continuity(lines: Sequence[Dict[str, Any]], tol: float = 12.0) -> float:
        endpoints: List[Tuple[int, np.ndarray]] = []
        for idx, line in enumerate(lines):
            pts = to_np(line)
            if len(pts) < 2:
                continue
            endpoints.append((idx, pts[0]))
            endpoints.append((idx, pts[-1]))
        if not endpoints:
            return 0.0
        connected = 0
        for idx, (line_id, point) in enumerate(endpoints):
            best = 1e9
            for jdx, (other_line_id, other_point) in enumerate(endpoints):
                if idx == jdx or line_id == other_line_id:
                    continue
                best = min(best, float(np.linalg.norm(point - other_point)))
            if best <= tol:
                connected += 1
        return connected / max(1, len(endpoints))

    cat_to_gt: Dict[str, List[np.ndarray]] = {}
    cat_to_pred: Dict[str, List[np.ndarray]] = {}
    for line in gt_lines:
        cat_to_gt.setdefault(str(line.get("category", "unknown")), []).append(to_np(line))
    for line in pred_lines:
        cat_to_pred.setdefault(str(line.get("category", "unknown")), []).append(to_np(line))

    out: Dict[str, float] = {}
    for threshold in thresholds:
        tp = 0
        fp = 0
        fn = 0
        for category in set(cat_to_gt.keys()) | set(cat_to_pred.keys()):
            gts = cat_to_gt.get(category, [])
            preds = cat_to_pred.get(category, [])
            used = set()
            for pred in preds:
                best_j = -1
                best_dist = 1e9
                for gt_idx, gt in enumerate(gts):
                    if gt_idx in used:
                        continue
                    dist = chamfer(pred, gt)
                    if dist < best_dist:
                        best_dist = dist
                        best_j = gt_idx
                if best_j >= 0 and best_dist <= float(threshold):
                    used.add(best_j)
                    tp += 1
                else:
                    fp += 1
            fn += max(0, len(gts) - len(used))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        out[f"APC@{int(threshold)}px"] = 2 * precision * recall / max(1e-6, precision + recall)

    chamfers = []
    for category in set(cat_to_gt.keys()) | set(cat_to_pred.keys()):
        gts = cat_to_gt.get(category, [])
        preds = cat_to_pred.get(category, [])
        for pred in preds:
            chamfers.append(min(chamfer(pred, gt) for gt in gts) if gts else 64.0)
    out["mean_chamfer_px"] = float(np.mean(chamfers)) if chamfers else 64.0
    out["continuity_pred"] = continuity(pred_lines, tol=12.0)
    out["continuity_gt"] = continuity(gt_lines, tol=12.0)
    out["continuity_gap"] = abs(out["continuity_pred"] - out["continuity_gt"])
    out["pred_num_lines"] = float(len(pred_lines))
    out["gt_num_lines"] = float(len(gt_lines))
    return out


def get_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 20)
    except OSError:
        return ImageFont.load_default()


def draw_polyline(draw: ImageDraw.ImageDraw, points: List[List[int]], color: Tuple[int, int, int], width: int) -> None:
    if len(points) < 2:
        return
    draw.line([tuple(map(int, point)) for point in points], fill=color, width=width)


def render_panel(image: Image.Image, lines: List[Dict[str, Any]], title: str, line_color: Tuple[int, int, int]) -> Image.Image:
    panel = image.copy().convert("RGB")
    draw = ImageDraw.Draw(panel)
    for line in lines:
        draw_polyline(draw, line.get("points", []), line_color, width=4)
    draw.rectangle([(0, 0), (image.width - 1, image.height - 1)], outline=(180, 0, 255), width=3)
    font = get_font()
    draw.rectangle([(8, 8), (360, 42)], fill=(0, 0, 0))
    draw.text((16, 14), title[:80], fill=(255, 255, 255), font=font)
    return panel


def stack_panels(left: Image.Image, right: Image.Image) -> Image.Image:
    width, height = left.size
    canvas = Image.new("RGB", (width * 2, height), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (width, 0))
    return canvas


def generate_prediction(
    sample: Dict[str, Any],
    image: Image.Image,
    image_path: Path,
    processor: Any,
    model: torch.nn.Module,
    formatter: DiscreteMapTokenFormatter,
    max_new_tokens: int,
) -> Tuple[str, Optional[float]]:
    conversation = build_prompt_conversation(sample=sample, image_path=str(image_path), formatter=formatter)
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
    prompt_len = int(inputs["input_ids"].shape[1])
    gen_ids = generated.sequences[:, prompt_len:]
    text = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    sample_score: Optional[float] = None
    if generated.scores:
        log_probs: List[float] = []
        for step_idx, step_scores in enumerate(generated.scores):
            if step_idx >= gen_ids.shape[1]:
                break
            token_id = int(gen_ids[0, step_idx].item())
            step_log_probs = torch.log_softmax(step_scores[0], dim=-1)
            log_probs.append(float(step_log_probs[token_id].item()))
        if log_probs:
            sample_score = float(math.exp(sum(log_probs) / len(log_probs)))
    return text, sample_score


def main() -> None:
    args = parse_args()
    dataset_jsonl = Path(args.dataset_jsonl).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    model_or_checkpoint = Path(args.model_or_checkpoint).resolve()
    processor_path = Path(args.processor_path).resolve() if str(args.processor_path).strip() else model_or_checkpoint
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "viz"
    if not args.skip_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    formatter = DiscreteMapTokenFormatter(
        image_size=int(args.image_size),
        categories=[item.strip() for item in str(args.discrete_categories).split(",") if item.strip()],
        max_seq_len=8192,
        coord_num_bins=int(args.discrete_coord_num_bins),
        coordinate_token_style=str(args.discrete_token_schema),
        include_text_prompt_tokens=not bool(args.disable_legacy_text_prompt_tokens),
    )

    items = load_jsonl(dataset_jsonl)
    selected = select_items(
        items=items,
        sample_ids=[item.strip() for item in str(args.sample_ids).split(",") if item.strip()],
        id_prefixes=[item.strip() for item in str(args.id_prefixes).split(",") if item.strip()],
        max_samples=int(args.max_samples),
    )
    predictions_jsonl_path = output_dir / "predictions.jsonl"
    if predictions_jsonl_path.exists():
        predictions_jsonl_path.unlink()

    print(f"[portable-dtok-eval] model_or_checkpoint={model_or_checkpoint}", flush=True)
    print(f"[portable-dtok-eval] processor_path={processor_path}", flush=True)
    print(f"[portable-dtok-eval] dataset_jsonl={dataset_jsonl}", flush=True)
    print(f"[portable-dtok-eval] dataset_root={dataset_root}", flush=True)
    print(f"[portable-dtok-eval] output_dir={output_dir}", flush=True)
    print(f"[portable-dtok-eval] selected_samples={len(selected)}", flush=True)

    processor = load_processor(str(processor_path), formatter=formatter)
    model = load_inference_model(str(model_or_checkpoint), device=str(args.device))

    results: List[Dict[str, Any]] = []
    agg_metrics: Dict[str, List[float]] = {}
    started_at = time.time()

    for idx, sample in enumerate(selected, start=1):
        sample_id = str(sample["id"])
        rel_image = str(sample["images"][0])
        image_path = (dataset_root / rel_image).resolve()
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")

        pred_text, sample_score = generate_prediction(
            sample=sample,
            image=image,
            image_path=image_path,
            processor=processor,
            model=model,
            formatter=formatter,
            max_new_tokens=int(args.max_new_tokens),
        )
        cleaned_text = formatter.normalize_generated_text(pred_text)
        pred_lines = sanitize_lines(formatter.text_to_lines(cleaned_text))
        gt_lines = sanitize_lines(extract_gt_lines(sample))
        parse_ok = bool(cleaned_text)

        metrics = sample_metrics(pred_lines=pred_lines, gt_lines=gt_lines, thresholds=[2.0, 4.0, 8.0])
        for key, value in metrics.items():
            agg_metrics.setdefault(key, []).append(float(value))

        if not args.skip_viz:
            gt_panel = render_panel(image=image, lines=gt_lines, title=f"{sample_id} | GT", line_color=(0, 120, 255))
            pred_panel = render_panel(image=image, lines=pred_lines, title=f"{sample_id} | Pred", line_color=(255, 60, 60))
            stack_panels(gt_panel, pred_panel).save(viz_dir / f"{sample_id}.png")

        result_row = {
            "id": sample_id,
            "image": rel_image,
            "image_size": [int(image.width), int(image.height)],
            "parse_ok": parse_ok,
            "sample_score": sample_score,
            "pred_text": pred_text,
            "pred_token_text": cleaned_text,
            "pred_lines": pred_lines,
            "gt_lines": gt_lines,
            "metrics": metrics,
        }
        results.append(result_row)
        append_jsonl(predictions_jsonl_path, result_row)

        if idx == 1 or idx == len(selected) or (int(args.progress_every) > 0 and idx % int(args.progress_every) == 0):
            elapsed_sec = max(0.0, time.time() - started_at)
            avg_sec = elapsed_sec / max(1, idx)
            remaining_sec = avg_sec * max(0, len(selected) - idx)
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "processed": idx,
                        "total": len(selected),
                        "sample_id": sample_id,
                        "parse_ok": parse_ok,
                        "elapsed_sec": round(elapsed_sec, 1),
                        "eta_sec": round(remaining_sec, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    paper_categories = [item.strip() for item in str(args.paper_categories).split(",") if item.strip()]
    summary = {
        "num_samples": len(results),
        "num_parse_ok": sum(1 for item in results if item["parse_ok"]),
        "mean_metrics_note": {
            "description": "Approximate engineering metrics in pixel units; do not confuse them with paper-style meter-based metrics.",
            "pixel_thresholds": [2.0, 4.0, 8.0],
        },
        "mean_metrics": {key: sum(values) / max(1, len(values)) for key, values in agg_metrics.items()},
        "paper_metrics_note": {
            "description": "Paper-style metrics. Chamfer thresholds are in meters, converted using meter_per_pixel.",
            "meter_per_pixel": float(args.meter_per_pixel),
            "chamfer_thresholds_m": [0.9, 1.5, 3.0, 4.5],
            "equivalent_pixel_thresholds": [
                round(0.9 / float(args.meter_per_pixel), 4),
                round(1.5 / float(args.meter_per_pixel), 4),
                round(3.0 / float(args.meter_per_pixel), 4),
                round(4.5 / float(args.meter_per_pixel), 4),
            ],
        },
        "paper_metrics": evaluate_prediction_items(
            items=results,
            meter_per_pixel=float(args.meter_per_pixel),
            line_width_px=int(args.line_width_px),
            categories=paper_categories or None,
            default_image_size=int(args.image_size),
        ),
    }

    (output_dir / "predictions.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
