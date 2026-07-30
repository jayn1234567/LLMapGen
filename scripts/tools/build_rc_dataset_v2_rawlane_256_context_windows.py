#!/usr/bin/env python3
"""Build 550k raw-lane-overlay Dataset V2 variants.

Outputs:
  * rawlane_local256_550k
  * rawlane_context512_roi256_550k

The raw lane is read from ``patch_tif/0_lane.tif`` under each raw sample folder
and rendered as white pixels on top of the BEV input image. Labels, difficulty
sampling, lane/intersection semantic mappings, and train/eval/test split logic
remain the Dataset V2 defaults.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.build_dataset_v2 import allocate_quotas, parse_ratio_spec
from scripts.tools.build_rc_dataset_v2_from_obs import (
    DEFAULT_SOURCE_OBS_ROOTS,
    create_variant_tar,
)


TARGET_SAMPLES = 550_000
DIFFICULTY_RATIOS = "empty=0,easy=0.30,medium=0.33,hard=0.27,very_hard=0.10"
INTERSECTION_RATIO = 0.30
SOURCE_LOCAL = "local256"
SOURCE_CONTEXT = "context512_roi256"
TARGET_LOCAL = "rawlane_local256_550k"
TARGET_CONTEXT = "rawlane_context512_roi256_550k"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", default=r"D:\data\fulldata_rawlane")
    parser.add_argument("--raw-root", default="")
    parser.add_argument("--staging-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--package-root", default="")
    parser.add_argument("--source-obs-root", action="append", default=[])
    parser.add_argument("--obs-backend", choices=["auto", "moxing", "obsutil"], default="auto")
    parser.add_argument("--obsutil-path", default="")
    parser.add_argument("--obsutil-config", default="")
    parser.add_argument("--obsutil-jobs", type=int, default=8)
    parser.add_argument("--archive-workers", type=int, default=16)
    parser.add_argument("--train-stride", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--difficulty-seed", type=int, default=20260713)
    parser.add_argument("--raw-lane-threshold", type=float, default=0.0)
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--image-decode-mode", choices=["sampled", "all", "none"], default="sampled")
    parser.add_argument("--visualize-per-difficulty", type=int, default=0)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-stage", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--keep-raw-source-after-stage", action="store_true")
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-samples", type=int, default=None)
    return parser.parse_args(argv)


def run(command: list) -> None:
    command = [str(item) for item in command]
    print("[rawlane-dataset] command:", shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def expected_counts() -> dict[str, int]:
    return allocate_quotas(TARGET_SAMPLES, parse_ratio_spec(DIFFICULTY_RATIOS))


def relabel_count_keys(counts: dict, source: str, target: str) -> dict:
    prefix = f"{source}:"
    return {
        f"{target}:{key[len(prefix):]}" if str(key).startswith(prefix) else key: value
        for key, value in counts.items()
    }


def relabel_metadata(path: Path, source_variant: str, target_variant: str) -> None:
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dataset_variant"] = target_variant
    payload["active_variant"] = target_variant
    payload["base_view_mode"] = source_variant
    if isinstance(payload.get("variants"), list):
        payload["variants"] = [
            target_variant if item == source_variant else item
            for item in payload["variants"]
        ]
    if isinstance(payload.get("variants"), dict):
        payload["variants"][target_variant] = payload["variants"].pop(source_variant, payload["variants"].get(source_variant, {}))
        if isinstance(payload["variants"].get(target_variant), dict):
            payload["variants"][target_variant]["root"] = str(path.parent)
    if isinstance(payload.get("record_counts"), dict):
        payload["record_counts"] = relabel_count_keys(payload["record_counts"], source_variant, target_variant)
    if isinstance(payload.get("splits"), dict):
        for split_payload in payload["splits"].values():
            if isinstance(split_payload, dict) and isinstance(split_payload.get("record_counts"), dict):
                split_payload["record_counts"] = relabel_count_keys(
                    split_payload["record_counts"],
                    source_variant,
                    target_variant,
                )
            if isinstance(split_payload, dict) and isinstance(split_payload.get("semantic_target_counts"), dict):
                split_payload["semantic_target_counts"] = relabel_count_keys(
                    split_payload["semantic_target_counts"],
                    source_variant,
                    target_variant,
                )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def variant_completion_errors(root: Path, variant: str) -> list[str]:
    errors = []
    info_path = root / "dataset_info.json"
    train_path = root / "phase_a" / "train.jsonl"
    if not info_path.is_file():
        errors.append(f"missing {info_path}")
        return errors
    train_count = count_jsonl(train_path)
    if train_count != TARGET_SAMPLES:
        errors.append(f"train count={train_count}, expected {TARGET_SAMPLES}: {train_path}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"invalid dataset_info.json: {exc}")
        return errors
    overlay = info.get("input_overlay") or {}
    balance = info.get("balance") or {}
    if info.get("dataset_variant") != variant and info.get("active_variant") != variant:
        errors.append(
            f"dataset_variant={info.get('dataset_variant')!r}, active_variant={info.get('active_variant')!r}, "
            f"expected {variant!r}"
        )
    if overlay.get("raw_lane_overlay") is not True:
        errors.append("input_overlay.raw_lane_overlay is not true")
    if overlay.get("raw_lane_overlay_source") != "patch_tif/0_lane.tif":
        errors.append(f"unexpected raw_lane_overlay_source={overlay.get('raw_lane_overlay_source')!r}")
    bucket_counts = balance.get("final_bucket_counts")
    if isinstance(bucket_counts, dict):
        selected_total = sum(int(value) for value in bucket_counts.values())
        if selected_total != TARGET_SAMPLES:
            errors.append(f"final_bucket_counts total={selected_total}, expected {TARGET_SAMPLES}")
    elif balance.get("selected_total") is not None and int(balance["selected_total"]) != TARGET_SAMPLES:
        errors.append(f"selected_total={balance.get('selected_total')}, expected {TARGET_SAMPLES}")
    try:
        actual_intersection_ratio = float(balance.get("actual_intersection_ratio", -1.0))
    except (TypeError, ValueError):
        actual_intersection_ratio = -1.0
    if abs(actual_intersection_ratio - INTERSECTION_RATIO) > 1e-8:
        errors.append(
            f"actual_intersection_ratio={actual_intersection_ratio}, expected {INTERSECTION_RATIO}"
        )
    return errors


def completed_variant(root: Path, variant: str) -> bool:
    return not variant_completion_errors(root, variant)


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    work_root = Path(args.work_root).expanduser().resolve()
    return {
        "work_root": work_root,
        "raw_root": Path(args.raw_root).expanduser().resolve() if args.raw_root else work_root / "raw_sources",
        "staging_root": Path(args.staging_root).expanduser().resolve() if args.staging_root else work_root / "staging_rawlane_256_context",
        "output_root": Path(args.output_root).expanduser().resolve() if args.output_root else work_root / "output_rawlane_256_context",
        "package_root": Path(args.package_root).expanduser().resolve() if args.package_root else work_root / "packages_rawlane",
    }


def run_streaming_builder(paths: dict[str, Path], args: argparse.Namespace) -> None:
    if args.skip_stage:
        return
    command = [
        sys.executable,
        "scripts/tools/build_rc_dataset_v2_streaming_from_obs.py",
        "--work-root", paths["work_root"],
        "--raw-root", paths["raw_root"],
        "--staging-root", paths["staging_root"],
        "--output-root", paths["output_root"],
        "--views", "both",
        "--patch-size", 256,
        "--context-size", 512,
        "--eval-test-stride", 256,
        "--train-target-samples", TARGET_SAMPLES,
        "--train-stride", args.train_stride,
        "--difficulty-ratios", DIFFICULTY_RATIOS,
        "--intersection-target-ratio", INTERSECTION_RATIO,
        "--split-seed", args.split_seed,
        "--difficulty-seed", args.difficulty_seed,
        "--archive-workers", args.archive_workers,
        "--obs-backend", args.obs_backend,
        "--obsutil-jobs", args.obsutil_jobs,
        "--raw-lane-overlay",
        "--require-raw-lane",
        "--raw-lane-threshold", args.raw_lane_threshold,
        "--skip-upload",
    ]
    if args.skip_download:
        command.append("--skip-download")
    if args.keep_raw_source_after_stage:
        command.append("--keep-raw-source-after-stage")
    if args.keep_archives:
        command.append("--keep-archives")
    if args.resume:
        command.append("--resume")
    if args.limit_samples is not None:
        command.extend(["--limit-samples", args.limit_samples])
    if args.obsutil_path:
        command.extend(["--obsutil-path", args.obsutil_path])
    if args.obsutil_config:
        command.extend(["--obsutil-config", args.obsutil_config])
    for source in args.source_obs_root or DEFAULT_SOURCE_OBS_ROOTS:
        command.extend(["--source-obs-root", source])
    run(command)


def rename_variant(output_root: Path, source_variant: str, target_variant: str, resume: bool) -> Path:
    source_root = output_root / source_variant
    target_root = output_root / target_variant
    if resume and completed_variant(target_root, target_variant):
        print(f"[rawlane-dataset] reuse renamed variant: {target_root}", flush=True)
        return target_root
    if target_root.exists() and not completed_variant(target_root, target_variant):
        errors = variant_completion_errors(target_root, target_variant)
        raise ValueError(f"target variant exists but is incomplete: {target_root}; errors={errors}")
    if not target_root.exists():
        if not source_root.is_dir():
            raise FileNotFoundError(f"source variant not found: {source_root}")
        source_root.rename(target_root)
    relabel_metadata(target_root / "dataset_info.json", source_variant, target_variant)
    relabel_metadata(output_root / "build_summary.json", source_variant, target_variant)
    if not completed_variant(target_root, target_variant):
        errors = variant_completion_errors(target_root, target_variant)
        raise ValueError(f"renamed rawlane variant failed metadata checks: {target_root}; errors={errors}")
    return target_root


def validate_variant(root: Path, variant: str, args: argparse.Namespace) -> None:
    if args.skip_validation:
        return
    run([
        sys.executable,
        "scripts/tools/validate_visualize_rc_dataset_v2.py",
        "--dataset-root", root,
        "--variant", variant,
        "--expected-train-samples", TARGET_SAMPLES,
        "--difficulty-ratios", DIFFICULTY_RATIOS,
        "--expected-intersection-ratio", INTERSECTION_RATIO,
        "--output-dir", root.parent / f"{variant}_validation",
        "--visualize-per-difficulty", args.visualize_per_difficulty,
        "--image-decode-mode", args.image_decode_mode,
    ])


def package_variant(root: Path, package_root: Path, resume: bool) -> Path:
    package = package_root / f"{root.name}.tar"
    create_variant_tar(root, package, resume)
    print(f"[rawlane-dataset] package: {package}", flush=True)
    return package


def main(argv=None) -> None:
    args = parse_args(argv)
    paths = resolve_paths(args)
    for key, value in paths.items():
        value.mkdir(parents=True, exist_ok=True)
        print(f"[rawlane-dataset] {key}: {value}", flush=True)
    print(f"[rawlane-dataset] target samples: {TARGET_SAMPLES}", flush=True)
    print(f"[rawlane-dataset] difficulty: {DIFFICULTY_RATIOS}", flush=True)
    print("[rawlane-dataset] overlay: patch_tif/0_lane.tif -> white pixels", flush=True)

    run_streaming_builder(paths, args)

    local_root = rename_variant(paths["output_root"], SOURCE_LOCAL, TARGET_LOCAL, args.resume)
    context_root = rename_variant(paths["output_root"], SOURCE_CONTEXT, TARGET_CONTEXT, args.resume)

    validate_variant(local_root, TARGET_LOCAL, args)
    validate_variant(context_root, TARGET_CONTEXT, args)

    packages = []
    if not args.skip_package:
        packages.append(package_variant(local_root, paths["package_root"], args.resume))
        packages.append(package_variant(context_root, paths["package_root"], args.resume))

    summary = {
        "status": "passed",
        "variants": {
            TARGET_LOCAL: str(local_root),
            TARGET_CONTEXT: str(context_root),
        },
        "packages": [str(path) for path in packages],
        "target_samples": TARGET_SAMPLES,
        "difficulty_ratios": DIFFICULTY_RATIOS,
        "target_quotas": expected_counts(),
        "intersection_ratio": INTERSECTION_RATIO,
        "raw_lane_overlay": {
            "source": "patch_tif/0_lane.tif",
            "style": "white_pixels_on_rgb_channels",
            "threshold": args.raw_lane_threshold,
        },
    }
    summary_path = paths["output_root"] / "rawlane_build_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
