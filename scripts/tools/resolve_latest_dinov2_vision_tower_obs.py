#!/usr/bin/env python3
"""Resolve the latest completed private DINOv2 vision tower on OBS.

Each segmentation run is considered publishable only after its success marker
and all required HuggingFace vision-tower artifacts are visible. The success
marker is uploaded last by the formal DI segmentation launcher.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_OBS_ROOT = (
    "obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/model/"
    "dinov2_private_seg_dinov3style_lora"
)
SUCCESS_MARKER = "DI_TRAIN_SUCCESS.json"
REQUIRED_RUN_ARTIFACTS = (
    SUCCESS_MARKER,
    "train_summary.json",
    "best/metrics.json",
    "best/vision_tower/config.json",
    "best/vision_tower/preprocessor_config.json",
    "best/vision_tower/model.safetensors",
    "best/vision_tower_verify.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-root", default=DEFAULT_OBS_ROOT)
    parser.add_argument(
        "--report",
        default="",
        help="Optional local JSON file receiving the selected run and rejected candidates.",
    )
    parser.add_argument(
        "--allow-run-root",
        action="store_true",
        help="Also accept --obs-root itself when it is one completed run directory.",
    )
    return parser.parse_args()


def obs_join(root: str, child: str) -> str:
    return root.rstrip("/") + "/" + child.strip("/")


def normalize_list_item(parent: str, item: str) -> str:
    item = str(item).strip().rstrip("/")
    if item.startswith("obs://"):
        return item
    return obs_join(parent, item)


def _safe_exists(mox: Any, path: str) -> bool:
    try:
        return bool(mox.file.exists(path))
    except Exception:
        return False


def _load_obs_json(mox: Any, path: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="dinov2_obs_resolve_") as temp_dir:
        local_path = Path(temp_dir) / Path(path).name
        mox.file.copy(path, str(local_path))
        payload = json.loads(local_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _fallback_run_timestamp(run_root: str) -> float:
    basename = run_root.rstrip("/").rsplit("/", 1)[-1]
    matches = re.findall(r"(20\d{6})[_-]?(\d{6})", basename)
    if not matches:
        return 0.0
    date_part, time_part = matches[-1]
    try:
        from datetime import datetime, timezone

        return datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        ).timestamp()
    except ValueError:
        return 0.0


def inspect_run(mox: Any, run_root: str) -> dict[str, Any]:
    run_root = run_root.rstrip("/")
    missing = [
        relative
        for relative in REQUIRED_RUN_ARTIFACTS
        if not _safe_exists(mox, obs_join(run_root, relative))
    ]
    result: dict[str, Any] = {
        "run_root": run_root,
        "vision_tower": obs_join(run_root, "best/vision_tower"),
        "complete": not missing,
        "missing": missing,
    }
    if missing:
        return result

    try:
        marker = _load_obs_json(mox, obs_join(run_root, SUCCESS_MARKER))
    except Exception as exc:
        result.update(complete=False, marker_error=repr(exc))
        return result

    if marker.get("status") != "passed":
        result.update(complete=False, marker_status=marker.get("status"))
        return result

    completed = marker.get("completed_unix_time")
    try:
        completed_value = float(completed)
    except (TypeError, ValueError):
        completed_value = _fallback_run_timestamp(run_root)
    result.update(
        marker=marker,
        completed_unix_time=completed_value,
        best_metric=marker.get("best_metric"),
        best_metric_value=marker.get("best_metric_value", marker.get("best_mean_iou")),
    )
    return result


def list_candidate_runs(mox: Any, obs_root: str, allow_run_root: bool) -> list[str]:
    obs_root = obs_root.rstrip("/")
    candidates: list[str] = []
    if allow_run_root and _safe_exists(mox, obs_join(obs_root, SUCCESS_MARKER)):
        candidates.append(obs_root)
    try:
        items = mox.file.list_directory(obs_root)
    except Exception as exc:
        raise RuntimeError(f"Unable to list DINOv2 registry root {obs_root}: {exc!r}") from exc
    for item in items or []:
        candidate = normalize_list_item(obs_root, str(item))
        if candidate.endswith((".json", ".txt", ".log", ".safetensors")):
            continue
        if _safe_exists(mox, obs_join(candidate, SUCCESS_MARKER)):
            candidates.append(candidate)
    return sorted(set(candidates))


def resolve_latest(mox: Any, obs_root: str, allow_run_root: bool = False) -> dict[str, Any]:
    candidates = [
        inspect_run(mox, run_root)
        for run_root in list_candidate_runs(mox, obs_root, allow_run_root)
    ]
    complete = [candidate for candidate in candidates if candidate.get("complete")]
    if not complete:
        rejected = [
            {"run_root": item["run_root"], "missing": item.get("missing", [])}
            for item in candidates
        ]
        raise RuntimeError(
            "No completed DINOv2 segmentation run with a verified best/vision_tower "
            f"was found below {obs_root}. Candidates: {rejected}"
        )
    selected = max(
        complete,
        key=lambda item: (float(item.get("completed_unix_time", 0.0)), item["run_root"]),
    )
    return {
        "obs_root": obs_root.rstrip("/"),
        "selection": "latest_completed_run_internal_best",
        "selected": selected,
        "candidates": candidates,
    }


def main() -> None:
    args = parse_args()
    try:
        import moxing as mox  # type: ignore

        report = resolve_latest(mox, args.obs_root, allow_run_root=args.allow_run_root)
    except Exception as exc:
        print(f"[dinov2-resolver] ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc

    selected = report["selected"]
    print(
        "[dinov2-resolver] selected "
        f"run={selected['run_root']} completed={selected.get('completed_unix_time')} "
        f"metric={selected.get('best_metric')} value={selected.get('best_metric_value')}",
        file=sys.stderr,
        flush=True,
    )
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(selected["vision_tower"], flush=True)


if __name__ == "__main__":
    main()
