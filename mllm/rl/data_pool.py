from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from mllm.coord_utils import record_coord_config
from mllm.rl.schemas import PoolBucket


@dataclass
class HardPoolConfig:
    low_f1_threshold: float = 0.30
    medium_f1_threshold: float = 0.75
    cut_threshold: float = 0.80
    random_keep_ratio: float = 0.15
    include_parse_fail_in_train: bool = False
    allow_summary_prompt_fallback: bool = False
    max_per_bucket: int | None = None
    seed: int = 42
    map_task: str = "lane"
    meter_per_pixel: float = 0.2
    buffer_size: float = 1.0
    match_threshold: float = 0.33


def _load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in {path}")
        return [item for item in data if isinstance(item, dict)]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _record_id(record: dict[str, Any]) -> str:
    return str(record.get("record_id") or record.get("id") or _stable_id(json.dumps(record, sort_keys=True, ensure_ascii=False)))


def _image_key(image: str | None) -> str | None:
    if not image:
        return None
    return Path(str(image)).stem


def _source_index(source_records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_image: dict[str, dict[str, Any]] = {}
    for row in source_records:
        by_id[_record_id(row)] = row
        key = _image_key(row.get("image"))
        if key:
            by_image.setdefault(key, row)
    return by_id, by_image


def _source_for_summary(
    summary: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    by_image: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    rid = _record_id(summary)
    if rid in by_id:
        return by_id[rid]
    key = _image_key(summary.get("image"))
    if key and key in by_image:
        return by_image[key]
    return None


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _line_scores(record: dict[str, Any]) -> dict[str, float]:
    ev = record.get("centerline_eval") if isinstance(record.get("centerline_eval"), dict) else {}
    instance_f1 = ev.get("instance_f1")
    if instance_f1 is None:
        precision = _safe_div(ev.get("matched_line_num", 0), ev.get("pred_line_num", 0))
        recall = _safe_div(ev.get("matched_line_num", 0), ev.get("gt_line_num", 0))
        instance_f1 = _safe_div(2 * precision * recall, precision + recall)
    length_f1 = ev.get("length_f1")
    if length_f1 is None:
        precision = _safe_div(ev.get("matched_line_length_sum", 0.0), ev.get("pred_line_length_sum", 0.0))
        recall = _safe_div(ev.get("matched_line_length_sum", 0.0), ev.get("gt_line_length_sum", 0.0))
        length_f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "instance_f1": float(instance_f1 or 0.0),
        "length_f1": float(length_f1 or 0.0),
        "mean_f1": (float(instance_f1 or 0.0) + float(length_f1 or 0.0)) / 2.0,
        "matched_line_num": float(ev.get("matched_line_num", 0) or 0),
        "gt_line_num": float(ev.get("gt_line_num", 0) or 0),
        "pred_line_num": float(ev.get("pred_line_num", 0) or 0),
    }


def _coord_config(summary: dict[str, Any], source: dict[str, Any] | None) -> dict[str, Any]:
    base = dict(source or {})
    if "meta" not in base and isinstance(summary.get("meta"), dict):
        base["meta"] = summary["meta"]
    for key in ("coord_mode", "coord_range", "patch_size", "patch_width", "patch_height"):
        if key in summary:
            base[key] = summary[key]
    return record_coord_config(base)


def _reward_components(summary: dict[str, Any], source: dict[str, Any] | None, config: HardPoolConfig) -> dict[str, Any]:
    try:
        from mllm.reward import MapRewardConfig, compute_map_reward
    except Exception as exc:
        return {"reward_parse_ok": False, "components": {}, "reward_error": f"reward backend unavailable: {exc}"}

    prediction = summary.get("prediction_json") or summary.get("prediction") or summary.get("raw_prediction") or ""
    ground_truth = _ground_truth(summary, source)
    if not prediction or not ground_truth:
        return {"reward_parse_ok": False, "components": {}, "reward_error": "missing prediction or ground truth"}
    coord = _coord_config(summary, source)
    reward_config = MapRewardConfig(
        map_task=config.map_task,
        patch_size=int(coord["patch_size"]),
        coord_mode=str(coord["coord_mode"]),
        coord_range=int(coord["coord_range"]),
        meter_per_pixel=config.meter_per_pixel,
        buffer_size=config.buffer_size,
        match_threshold=config.match_threshold,
    )
    result = compute_map_reward(str(prediction), str(ground_truth), reward_config)
    return {
        "reward": result.get("reward"),
        "reward_parse_ok": result.get("parse_ok"),
        "reward_error": result.get("parse_error"),
        "components": result.get("components") or {},
    }


def _bucket_for(summary: dict[str, Any], source: dict[str, Any] | None, config: HardPoolConfig) -> tuple[PoolBucket, dict[str, Any]]:
    scores = _line_scores(summary)
    parse_ok = bool(summary.get("parse_ok", True))
    ev = summary.get("centerline_eval") if isinstance(summary.get("centerline_eval"), dict) else {}
    valid_string = bool(ev.get("valid_string_format", parse_ok))
    reward = _reward_components(summary, source, config) if parse_ok and valid_string else {"components": {}}
    components = reward.get("components") or {}
    cut_score = min(float(components.get("cut_type", 1.0) or 0.0), float(components.get("cut_continuity", 1.0) or 0.0))

    reason: PoolBucket
    if not parse_ok or not valid_string:
        reason = PoolBucket.HARD_PARSE_FAIL
    elif scores["gt_line_num"] > 0 and scores["matched_line_num"] <= 0:
        reason = PoolBucket.HARD_ZERO_MATCH
    elif scores["mean_f1"] < config.low_f1_threshold:
        reason = PoolBucket.HARD_LOW_F1
    elif cut_score < config.cut_threshold:
        reason = PoolBucket.HARD_CUT_ERROR
    elif scores["mean_f1"] < config.medium_f1_threshold:
        reason = PoolBucket.MEDIUM
    else:
        reason = PoolBucket.RANDOM_KEEP

    details = {
        **scores,
        "parse_ok": parse_ok,
        "valid_string_format": valid_string,
        "parse_error": summary.get("parse_error"),
        "reward": reward.get("reward"),
        "reward_parse_ok": reward.get("reward_parse_ok"),
        "reward_error": reward.get("reward_error"),
        "reward_components": components,
        "cut_score": cut_score,
    }
    return reason, details


def _ground_truth(summary: dict[str, Any], source: dict[str, Any] | None) -> str:
    if source:
        conversations = source.get("conversations") or []
        if len(conversations) >= 2 and isinstance(conversations[1], dict):
            return str(conversations[1].get("value") or "")
    return str(summary.get("ground_truth") or summary.get("ground_truth_pixel") or "")


def _prompt(summary: dict[str, Any], source: dict[str, Any] | None) -> str:
    if source:
        conversations = source.get("conversations") or []
        if conversations and isinstance(conversations[0], dict):
            return str(conversations[0].get("value") or "")
    return str(summary.get("prompt") or "")


def _make_pool_record(
    summary: dict[str, Any],
    source: dict[str, Any] | None,
    bucket: PoolBucket,
    details: dict[str, Any],
    source_summary: str,
    config: HardPoolConfig,
    source_jsonl: str | Path | None,
) -> dict[str, Any]:
    if source_jsonl and source is None and not config.allow_summary_prompt_fallback:
        raise ValueError("source record not found; refusing to use formatted inference prompt as training prompt")
    base = dict(source or {})
    sample_id = _record_id(source or summary)
    prompt = _prompt(summary, source)
    ground_truth = _ground_truth(summary, source)
    if not prompt or not ground_truth:
        raise ValueError(f"Cannot build pool record for {sample_id}: missing prompt or ground truth")
    meta = dict(base.get("meta") or summary.get("meta") or {})
    meta["rl_pool"] = {
        "bucket": bucket.value,
        "source_summary": source_summary,
        "source_record_id": _record_id(summary),
        "scores": details,
        "prediction_preview": str(summary.get("prediction") or summary.get("raw_prediction") or "")[:500],
    }
    return {
        "id": sample_id,
        "image": base.get("image") or summary.get("image"),
        "meta": meta,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": ground_truth},
        ],
    }


def _limit_bucket(rows: list[dict[str, Any]], limit: int | None, rng: random.Random) -> list[dict[str, Any]]:
    if limit is None or len(rows) <= limit:
        return rows
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:limit]


def build_hard_pool(
    summary_path: str | Path,
    output_dir: str | Path,
    source_jsonl: str | Path | None = None,
    config: HardPoolConfig | None = None,
) -> dict[str, Any]:
    config = config or HardPoolConfig()
    rng = random.Random(config.seed)
    summary_records = _load_json_or_jsonl(summary_path)
    source_records = _load_json_or_jsonl(source_jsonl) if source_jsonl else []
    by_id, by_image = _source_index(source_records)

    buckets: dict[PoolBucket, list[dict[str, Any]]] = {bucket: [] for bucket in PoolBucket}
    skipped: list[dict[str, Any]] = []
    for summary in summary_records:
        source = _source_for_summary(summary, by_id, by_image) if source_records else None
        try:
            bucket, details = _bucket_for(summary, source, config)
            record = _make_pool_record(summary, source, bucket, details, str(summary_path), config, source_jsonl)
            buckets[bucket].append(record)
        except Exception as exc:
            skipped.append({"record_id": _record_id(summary), "error": str(exc)})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    limited: dict[PoolBucket, list[dict[str, Any]]] = {}
    for bucket, rows in buckets.items():
        selected = _limit_bucket(rows, config.max_per_bucket, rng)
        limited[bucket] = selected
        _write_jsonl(output_dir / f"{bucket.value}.jsonl", selected)

    hard_rows = (
        limited[PoolBucket.HARD_ZERO_MATCH]
        + limited[PoolBucket.HARD_LOW_F1]
        + limited[PoolBucket.HARD_CUT_ERROR]
    )
    if config.include_parse_fail_in_train:
        hard_rows = limited[PoolBucket.HARD_PARSE_FAIL] + hard_rows
    medium_rows = limited[PoolBucket.MEDIUM]
    random_keep = list(limited[PoolBucket.RANDOM_KEEP])
    keep_count = int(round((len(hard_rows) + len(medium_rows)) * max(config.random_keep_ratio, 0.0)))
    rng.shuffle(random_keep)
    combined = hard_rows + medium_rows + random_keep[:keep_count]
    rng.shuffle(combined)
    _write_jsonl(output_dir / "train.jsonl", combined)
    if skipped:
        _write_jsonl(output_dir / "skipped.jsonl", skipped)

    manifest = {
        "format": "mllm_rl_hard_pool",
        "summary_path": str(summary_path),
        "source_jsonl": str(source_jsonl) if source_jsonl else None,
        "config": asdict(config),
        "summary_count": len(summary_records),
        "source_count": len(source_records),
        "bucket_counts": {bucket.value: len(rows) for bucket, rows in buckets.items()},
        "selected_bucket_counts": {bucket.value: len(rows) for bucket, rows in limited.items()},
        "combined_train_count": len(combined),
        "skipped_count": len(skipped),
        "outputs": {
            "train": str(output_dir / "train.jsonl"),
            **{bucket.value: str(output_dir / f"{bucket.value}.jsonl") for bucket in PoolBucket},
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an RL hard-sample pool from inference summaries.")
    parser.add_argument("--summary", required=True, help="Inference summary JSON/JSONL from scripts/infer_centerline_checkpoint.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for bucket JSONL files and combined train.jsonl.")
    parser.add_argument("--source-jsonl", default=None, help="Optional original SFT JSONL to preserve relative image paths and prompts.")
    parser.add_argument("--map-task", default="lane", choices=["lane", "lane_intersection", "intersection", "all"])
    parser.add_argument("--low-f1-threshold", type=float, default=0.30)
    parser.add_argument("--medium-f1-threshold", type=float, default=0.75)
    parser.add_argument("--cut-threshold", type=float, default=0.80)
    parser.add_argument("--random-keep-ratio", type=float, default=0.15)
    parser.add_argument(
        "--include-parse-fail-in-train",
        action="store_true",
        help="Include parse-failure samples in train.jsonl. Default keeps them only as an audit bucket.",
    )
    parser.add_argument(
        "--allow-summary-prompt-fallback",
        action="store_true",
        help="When --source-jsonl is provided but a source row is missing, allow using the inference summary prompt.",
    )
    parser.add_argument("--max-per-bucket", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = HardPoolConfig(
        low_f1_threshold=args.low_f1_threshold,
        medium_f1_threshold=args.medium_f1_threshold,
        cut_threshold=args.cut_threshold,
        random_keep_ratio=args.random_keep_ratio,
        include_parse_fail_in_train=args.include_parse_fail_in_train,
        allow_summary_prompt_fallback=args.allow_summary_prompt_fallback,
        max_per_bucket=args.max_per_bucket,
        seed=args.seed,
        map_task=args.map_task,
    )
    manifest = build_hard_pool(args.summary, args.output_dir, source_jsonl=args.source_jsonl, config=config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
