#!/usr/bin/env python3
"""Compare fixed-1100 patch metrics from multiple checkpoint runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BUCKETS = ("all_selected", "easy", "medium", "hard", "very_hard")
LINE_FIELDS = (
    "instance_pre",
    "instance_recall",
    "instance_f1",
    "length_pre",
    "length_recall",
    "length_f1",
)
INTERSECTION_FIELDS = (
    "mean_matched_iou",
    "mean_sample_union_iou",
    "micro_area_iou",
    "gt_polygon_num",
    "pred_polygon_num",
    "matched_polygon_num",
)


def parse_checkpoint(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--checkpoint must use LABEL=/path/to/by_difficulty")
    return label.strip(), Path(raw_path).expanduser().resolve()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def metric_value(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def compact_eval(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for section in ("centerline_eval", "intersection_eval", "lane_intersection_eval"):
        source = payload.get(section)
        if not isinstance(source, dict):
            continue
        result[section] = {
            field: metric_value(source, field)
            for field in LINE_FIELDS
            if metric_value(source, field) is not None
        }
        if section == "intersection_eval":
            result[section].update(
                {
                    field: metric_value(source, field)
                    for field in INTERSECTION_FIELDS
                    if metric_value(source, field) is not None
                }
            )
    for section in ("lane_type_eval", "intersection_type_eval"):
        source = payload.get(section)
        if isinstance(source, dict):
            result[section] = {
                "status": source.get("status"),
                "matched_type_accuracy": metric_value(source, "matched_type_accuracy"),
                "typed_gt_count": source.get("typed_gt_count"),
                "typed_geometry_matched_count": source.get("typed_geometry_matched_count"),
            }
    return result


def primary_score(compact: dict[str, Any]) -> float | None:
    """Balanced geometry score used for one explicit checkpoint recommendation."""
    lane = compact.get("centerline_eval") if isinstance(compact.get("centerline_eval"), dict) else {}
    intersection = (
        compact.get("intersection_eval") if isinstance(compact.get("intersection_eval"), dict) else {}
    )
    candidates = (
        lane.get("instance_f1"),
        lane.get("length_f1"),
        intersection.get("instance_f1"),
        intersection.get("micro_area_iou", intersection.get("length_f1")),
    )
    values = [float(value) for value in candidates if isinstance(value, (int, float))]
    return sum(values) / len(values) if values else None


def compare(checkpoints: list[tuple[str, Path]]) -> dict[str, Any]:
    labels = [label for label, _ in checkpoints]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Checkpoint labels must be unique: {labels}")
    runs: dict[str, Any] = {}
    for label, root in checkpoints:
        buckets = {}
        for bucket in BUCKETS:
            path = root / bucket / "eval.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing fixed-1100 metric for {label}/{bucket}: {path}")
            buckets[bucket] = compact_eval(read_json(path))
        runs[label] = {"metrics_root": str(root), "buckets": buckets}

    winners: dict[str, Any] = {}
    for bucket in BUCKETS:
        bucket_winners: dict[str, Any] = {}
        for section in ("centerline_eval", "intersection_eval", "lane_intersection_eval"):
            section_winners = {}
            fields = LINE_FIELDS + (INTERSECTION_FIELDS if section == "intersection_eval" else ())
            for field in fields:
                values = {
                    label: runs[label]["buckets"][bucket].get(section, {}).get(field)
                    for label in labels
                }
                numeric = {label: value for label, value in values.items() if isinstance(value, (int, float))}
                if numeric:
                    best = max(numeric.values())
                    section_winners[field] = {
                        "best_value": best,
                        "checkpoints": [label for label, value in numeric.items() if value == best],
                        "values": numeric,
                    }
            if section_winners:
                bucket_winners[section] = section_winners
        winners[bucket] = bucket_winners

    primary_scores = {
        label: primary_score(runs[label]["buckets"]["all_selected"])
        for label in labels
    }
    available_scores = {
        label: score for label, score in primary_scores.items() if isinstance(score, (int, float))
    }
    recommended = []
    if available_scores:
        best = max(available_scores.values())
        recommended = [label for label, score in available_scores.items() if score == best]
    return {
        "comparison_policy": {
            "same_fixed_eval_required": True,
            "primary_score": (
                "equal-weight mean of all_selected centerline instance_f1, centerline length_f1, "
                "intersection instance_f1, and intersection micro_area_iou"
            ),
            "note": "Per-metric winners remain authoritative when model selection prioritizes one task.",
        },
        "checkpoints": labels,
        "runs": runs,
        "winners": winners,
        "all_selected_primary_scores": primary_scores,
        "recommended_by_primary_score": recommended,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    labels = payload["checkpoints"]
    rows = [
        "# Fixed-1100 checkpoint comparison",
        "",
        "All checkpoints used the same saved 1100 records and three-image prompt contract.",
        "",
        "## All-selected metrics",
        "",
        "| Metric | " + " | ".join(labels) + " | Winner |",
        "|---|" + "---:|" * len(labels) + "---|",
    ]
    metric_specs = (
        ("Centerline instance F1", "centerline_eval", "instance_f1"),
        ("Centerline length F1", "centerline_eval", "length_f1"),
        ("Intersection instance F1", "intersection_eval", "instance_f1"),
        ("Intersection micro area IoU", "intersection_eval", "micro_area_iou"),
        ("Intersection mean matched IoU", "intersection_eval", "mean_matched_iou"),
        ("Lane + intersection instance F1", "lane_intersection_eval", "instance_f1"),
        ("Lane + intersection length F1", "lane_intersection_eval", "length_f1"),
    )
    all_winners = payload["winners"]["all_selected"]
    for title, section, field in metric_specs:
        values = [
            payload["runs"][label]["buckets"]["all_selected"].get(section, {}).get(field)
            for label in labels
        ]
        rendered = [f"{value:.4f}" if isinstance(value, (int, float)) else "N/A" for value in values]
        winner = ", ".join(all_winners.get(section, {}).get(field, {}).get("checkpoints", [])) or "N/A"
        rows.append(f"| {title} | " + " | ".join(rendered) + f" | {winner} |")

    rows.extend(["", "## Matched semantic type accuracy", ""])
    rows.append("| Type metric | " + " | ".join(labels) + " |")
    rows.append("|---|" + "---:|" * len(labels))
    for title, section in (
        ("Lane type", "lane_type_eval"),
        ("Intersection type", "intersection_type_eval"),
    ):
        values = [
            payload["runs"][label]["buckets"]["all_selected"].get(section, {}).get("matched_type_accuracy")
            for label in labels
        ]
        rendered = [f"{value:.4f}" if isinstance(value, (int, float)) else "N/A" for value in values]
        rows.append(f"| {title} | " + " | ".join(rendered) + " |")

    rows.extend(["", "## Primary balanced geometry score", ""])
    for label in labels:
        value = payload["all_selected_primary_scores"].get(label)
        rows.append(f"- `{label}`: {value:.4f}" if isinstance(value, (int, float)) else f"- `{label}`: N/A")
    recommended = payload.get("recommended_by_primary_score") or []
    rows.append("")
    rows.append("Recommended by the declared balanced-score policy: " + (", ".join(recommended) or "N/A"))
    rows.extend([
        "",
        "This recommendation is not a hidden aggregate. Select a different checkpoint when "
        "centerline recall, intersection precision, or one difficulty bucket is the actual release target.",
        "",
    ])
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    payload = compare(args.checkpoint)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "output_json": str(args.output_json),
        "output_markdown": str(args.output_markdown),
        "recommended_by_primary_score": payload["recommended_by_primary_score"],
        "all_selected_primary_scores": payload["all_selected_primary_scores"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
