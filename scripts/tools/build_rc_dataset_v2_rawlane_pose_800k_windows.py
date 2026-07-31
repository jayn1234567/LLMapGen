#!/usr/bin/env python3
"""Build paired 800k local/context Dataset V2 views with raw lane and pose.

Each SFT record contains two prompt images:

1. BEV road image with patch_tif/0_lane.tif rendered as a white overlay.
2. A separate black/white historical trajectory image from patch_tif/0_pose.tif.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from itertools import zip_longest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.build_rc_dataset_v2_from_obs import DEFAULT_SOURCE_OBS_ROOTS, create_variant_tar
from scripts.tools.build_rc_dataset_v2_rawlane_256_context_windows import relabel_metadata
from data_process.fixed_source_splits import load_fixed_source_split_manifest


TARGET_SAMPLES = 800_000
DIFFICULTY_RATIOS = "empty=0,easy=0.30,medium=0.33,hard=0.27,very_hard=0.10"
INTERSECTION_RATIO = 0.30
VARIANT_NAMES = {
    "local256": "rawlane_pose_local256_800k",
    "context512_roi256": "rawlane_pose_context512_roi256_800k",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", default=r"D:\data\fulldata_rawlane_pose")
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
    parser.add_argument(
        "--fixed-source-split-manifest",
        default=os.environ.get("RC_FIXED_SOURCE_SPLIT_MANIFEST", ""),
        help="Reusable fixed large-map eval/test manifest.",
    )
    parser.add_argument("--allow-missing-fixed-holdouts", action="store_true")
    parser.add_argument("--difficulty-seed", type=int, default=20260713)
    parser.add_argument("--raw-lane-threshold", type=float, default=0.0)
    parser.add_argument("--pose-threshold", type=float, default=0.0)
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--image-decode-mode", choices=["sampled", "all", "none"], default="sampled")
    parser.add_argument("--visualize-per-difficulty", type=int, default=0)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-stage", action="store_true")
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Complete all source stages, then stop before finalization, validation, and packaging.",
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--keep-raw-source-after-stage", action="store_true")
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-samples", type=int, default=None)
    return parser.parse_args(argv)


def run(command: list) -> None:
    command = [str(item) for item in command]
    print("[rawlane-pose-dataset] command:", shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    work_root = Path(args.work_root).expanduser().resolve()
    return {
        "work_root": work_root,
        "raw_root": Path(args.raw_root).expanduser().resolve() if args.raw_root else work_root / "raw_sources",
        "staging_root": (
            Path(args.staging_root).expanduser().resolve()
            if args.staging_root else work_root / "staging_rawlane_pose_256_context"
        ),
        "output_root": (
            Path(args.output_root).expanduser().resolve()
            if args.output_root else work_root / "output_rawlane_pose_256_context"
        ),
        "package_root": (
            Path(args.package_root).expanduser().resolve()
            if args.package_root else work_root / "packages_rawlane_pose"
        ),
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
        "--copy-mode", args.copy_mode,
        "--raw-lane-overlay",
        "--require-raw-lane",
        "--raw-lane-threshold", args.raw_lane_threshold,
        "--pose-second-image",
        "--pose-threshold", args.pose_threshold,
        "--skip-upload",
    ]
    if args.skip_download:
        command.append("--skip-download")
    if args.stage_only:
        command.append("--skip-finalize")
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
    if args.fixed_source_split_manifest:
        command.extend(["--fixed-source-split-manifest", args.fixed_source_split_manifest])
    if args.allow_missing_fixed_holdouts:
        command.append("--allow-missing-fixed-holdouts")
    for source in args.source_obs_root or DEFAULT_SOURCE_OBS_ROOTS:
        command.extend(["--source-obs-root", source])
    run(command)


def completion_errors(root: Path, expected_fixed_split_sha256: str = "") -> list[str]:
    errors = []
    train_path = root / "phase_a" / "train.jsonl"
    info_path = root / "dataset_info.json"
    train_count = count_jsonl(train_path)
    if train_count != TARGET_SAMPLES:
        errors.append(f"train count={train_count}, expected={TARGET_SAMPLES}")
    if not info_path.is_file():
        errors.append(f"missing {info_path}")
        return errors
    info = json.loads(info_path.read_text(encoding="utf-8"))
    multi = info.get("multi_image_input") or {}
    overlay = info.get("input_overlay") or {}
    if overlay.get("raw_lane_overlay") is not True:
        errors.append("raw_lane_overlay is not true")
    if multi.get("enabled") is not True or int(multi.get("num_images_per_sample", 0)) != 2:
        errors.append(f"invalid multi_image_input={multi!r}")
    if multi.get("pose_image_source") != "patch_tif/0_pose.tif":
        errors.append(f"invalid pose_image_source={multi.get('pose_image_source')!r}")
    balance = info.get("balance") or {}
    if abs(float(balance.get("actual_intersection_ratio", -1.0)) - INTERSECTION_RATIO) > 1e-8:
        errors.append(f"invalid intersection ratio={balance.get('actual_intersection_ratio')}")
    fixed = info.get("fixed_source_split") or {}
    actual_fixed_sha = str(fixed.get("file_sha256") or "")
    if actual_fixed_sha != expected_fixed_split_sha256:
        errors.append(
            f"fixed split sha256={actual_fixed_sha!r}, expected={expected_fixed_split_sha256!r}"
        )
    return errors


def rename_variant(
    output_root: Path,
    source_variant: str,
    target_variant: str,
    resume: bool,
    expected_fixed_split_sha256: str,
) -> Path:
    source_root = output_root / source_variant
    target_root = output_root / target_variant
    if resume and target_root.is_dir() and not completion_errors(
        target_root, expected_fixed_split_sha256
    ):
        print(f"[rawlane-pose-dataset] reuse variant: {target_root}", flush=True)
        return target_root
    if target_root.exists():
        raise ValueError(
            f"target variant exists but is incompatible: {target_root}; "
            f"{completion_errors(target_root, expected_fixed_split_sha256)}. "
            "Use a new --work-root/output-root for a new benchmark split."
        )
    if not source_root.is_dir():
        raise FileNotFoundError(f"source variant not found: {source_root}")
    source_root.rename(target_root)
    relabel_metadata(target_root / "dataset_info.json", source_variant, target_variant)
    relabel_metadata(output_root / "build_summary.json", source_variant, target_variant)
    errors = completion_errors(target_root, expected_fixed_split_sha256)
    if errors:
        raise ValueError(f"renamed pose variant failed checks: {target_root}; errors={errors}")
    return target_root


def validate_two_image_records(root: Path) -> None:
    counts = {}
    for split in ("train", "eval", "test"):
        jsonl_path = root / "phase_a" / f"{split}.jsonl"
        count = 0
        with jsonl_path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                images = record.get("images")
                if not isinstance(images, list) or len(images) != 2:
                    raise ValueError(f"{jsonl_path}:{line_number} expected two images, got {images!r}")
                if record.get("image") != images[0]:
                    raise ValueError(f"{jsonl_path}:{line_number} primary image mismatch")
                if not str(images[0]).startswith(f"images/{split}/"):
                    raise ValueError(f"{jsonl_path}:{line_number} invalid BEV path: {images[0]!r}")
                if not str(images[1]).startswith(f"pose_images/{split}/"):
                    raise ValueError(f"{jsonl_path}:{line_number} invalid pose path: {images[1]!r}")
                prompt = str(record.get("conversations", [{}])[0].get("value", ""))
                if prompt.count("<image>") != 2 or "historical vehicle-trajectory image" not in prompt:
                    raise ValueError(f"{jsonl_path}:{line_number} invalid two-image prompt")
                for relative in images:
                    if not (root / relative).is_file():
                        raise FileNotFoundError(f"{jsonl_path}:{line_number} missing image: {root / relative}")
                count += 1
                if count % 100_000 == 0:
                    print(f"[rawlane-pose-dataset] validated {split}: {count}", flush=True)
        counts[split] = count
    if counts.get("train") != TARGET_SAMPLES:
        raise ValueError(f"two-image validation train count mismatch: {counts}")
    print(f"[rawlane-pose-dataset] two-image records passed: {counts}", flush=True)


def validate_variant_pairing(variant_roots: dict[str, Path]) -> None:
    local_root = variant_roots[VARIANT_NAMES["local256"]]
    context_root = variant_roots[VARIANT_NAMES["context512_roi256"]]
    for split in ("train", "eval", "test"):
        local_path = local_root / "phase_a" / f"{split}.jsonl"
        context_path = context_root / "phase_a" / f"{split}.jsonl"
        with (
            local_path.open("r", encoding="utf-8-sig") as local_handle,
            context_path.open("r", encoding="utf-8-sig") as context_handle,
        ):
            for line_number, (local_line, context_line) in enumerate(
                zip_longest(local_handle, context_handle),
                start=1,
            ):
                if local_line is None or context_line is None:
                    raise ValueError(f"variant length mismatch for {split} at line {line_number}")
                local_id = json.loads(local_line)["id"]
                context_id = json.loads(context_line)["id"]
                if local_id != context_id:
                    raise ValueError(
                        f"variant id mismatch for {split} line {line_number}: "
                        f"{local_id!r} != {context_id!r}"
                    )
        print(f"[rawlane-pose-dataset] paired ids passed: {split}", flush=True)


def validate_variant(root: Path, variant: str, args: argparse.Namespace) -> None:
    if args.skip_validation:
        return
    validate_two_image_records(root)
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
        "--skip-distribution-check",
    ])


def main(argv=None) -> None:
    args = parse_args(argv)
    fixed_manifest = (
        load_fixed_source_split_manifest(args.fixed_source_split_manifest)
        if args.fixed_source_split_manifest else None
    )
    if fixed_manifest is not None:
        args.fixed_source_split_manifest = str(fixed_manifest["path"])
    fixed_split_sha256 = str((fixed_manifest or {}).get("file_sha256") or "")
    paths = resolve_paths(args)
    for key, value in paths.items():
        value.mkdir(parents=True, exist_ok=True)
        print(f"[rawlane-pose-dataset] {key}: {value}", flush=True)
    print(f"[rawlane-pose-dataset] target samples: {TARGET_SAMPLES}", flush=True)
    print(f"[rawlane-pose-dataset] difficulty: {DIFFICULTY_RATIOS}", flush=True)
    print("[rawlane-pose-dataset] image 1: BEV + patch_tif/0_lane.tif", flush=True)
    print("[rawlane-pose-dataset] image 2: patch_tif/0_pose.tif", flush=True)
    print(
        f"[rawlane-pose-dataset] fixed split: {args.fixed_source_split_manifest or '<disabled>'}",
        flush=True,
    )

    run_streaming_builder(paths, args)
    if args.stage_only:
        print(
            "[rawlane-pose-dataset] all available sources staged; "
            "create the fixed source split manifest next.",
            flush=True,
        )
        return
    variant_roots = {
        target: rename_variant(
            paths["output_root"], source, target, args.resume, fixed_split_sha256
        )
        for source, target in VARIANT_NAMES.items()
    }
    if not args.skip_validation:
        validate_variant_pairing(variant_roots)
    for variant, variant_root in variant_roots.items():
        validate_variant(variant_root, variant, args)

    packages = []
    if not args.skip_package:
        for variant, variant_root in variant_roots.items():
            package = paths["package_root"] / f"{variant}.tar"
            create_variant_tar(variant_root, package, args.resume)
            packages.append(str(package))

    summary = {
        "status": "passed",
        "variants": {name: str(root) for name, root in variant_roots.items()},
        "packages": packages,
        "target_samples": TARGET_SAMPLES,
        "difficulty_ratios": DIFFICULTY_RATIOS,
        "intersection_ratio": INTERSECTION_RATIO,
        "images_per_sample": 2,
        "image_roles": ["bev_road_structure", "historical_vehicle_trajectory"],
        "raw_lane_source": "patch_tif/0_lane.tif",
        "pose_source": "patch_tif/0_pose.tif",
        "fixed_source_split_manifest": args.fixed_source_split_manifest or None,
    }
    summary_path = paths["output_root"] / "rawlane_pose_build_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
