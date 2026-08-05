#!/usr/bin/env python3
"""Download RC sources once and build paired clean-BEV/raw-lane/pose 800k datasets.

The builder is disk bounded: it downloads one OBS source, stages both views,
verifies the source shard, and removes the downloaded raw source before moving
to the next one. Raw lane and pose are always separate model inputs; neither is
painted onto the BEV image.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.build_dataset_v2 import allocate_quotas, parse_ratio_spec
from scripts.tools.build_rc_dataset_v2_from_obs import DEFAULT_SOURCE_OBS_ROOTS
from scripts.tools.build_rc_dataset_v2_rawlane_256_context_windows import relabel_metadata


TARGET_SAMPLES = 800_000
DIFFICULTY_RATIOS = "empty=0.05,easy=0.25,medium=0.33,hard=0.27,very_hard=0.10"
INTERSECTION_RATIO = 0.30
VARIANT_NAMES = {
    "local256": "local256_rawlane_pose_800k",
    "context512_roi256": "context512_roi256_rawlane_pose_800k",
}
IMAGE_ROLES = [
    "bev_road_structure",
    "pv_camera_raw_lane",
    "historical_vehicle_trajectory",
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-root",
        default=r"D:\data\rc_dataset_v2_three_image_800k_from_obs",
    )
    parser.add_argument("--source-obs-root", action="append", default=[])
    parser.add_argument("--obsutil-path", default="")
    parser.add_argument("--obsutil-config", default="")
    parser.add_argument("--obsutil-jobs", type=int, default=8)
    parser.add_argument(
        "--fixed-source-split-manifest",
        default=r"D:\data\fixed_splits\rc_fixed_large_maps_v1.json",
    )
    parser.add_argument("--archive-workers", type=int, default=16)
    parser.add_argument("--validation-sample-limit", type=int, default=10_000)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--keep-raw-source-after-stage", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def run(command: list[object]) -> None:
    normalized = [str(item) for item in command]
    print("[three-image-obs] command:", shlex.join(normalized), flush=True)
    subprocess.run(normalized, cwd=REPO_ROOT, check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def expected_difficulty_counts() -> dict[str, int]:
    return allocate_quotas(TARGET_SAMPLES, parse_ratio_spec(DIFFICULTY_RATIOS))


def variant_is_complete(root: Path, source_variant: str) -> bool:
    info_path = root / "dataset_info.json"
    train_path = root / "phase_a" / "train.jsonl"
    if not info_path.is_file() or not train_path.is_file():
        return False
    try:
        info = read_json(info_path)
        balance = info.get("balance") or {}
        multi = info.get("multi_image_input") or {}
        overlay = info.get("input_overlay") or {}
        return (
            count_jsonl(train_path) == TARGET_SAMPLES
            and bool(balance.get("strict_difficulty_quotas", False))
            and balance.get("final_bucket_counts") == expected_difficulty_counts()
            and abs(float(balance.get("actual_intersection_ratio", -1)) - INTERSECTION_RATIO)
            <= 1e-8
            and overlay.get("raw_lane_overlay") is False
            and overlay.get("raw_lane_separate_image") is True
            and int(multi.get("num_images_per_sample", 0)) == 3
            and list(multi.get("image_roles") or []) == IMAGE_ROLES
            and str(info.get("base_view_mode") or source_variant) == source_variant
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def streaming_command(args: argparse.Namespace, paths: dict[str, Path]) -> list[object]:
    command: list[object] = [
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
        "--train-stride", 128,
        "--difficulty-ratios", DIFFICULTY_RATIOS,
        "--intersection-target-ratio", INTERSECTION_RATIO,
        "--strict-difficulty-quotas",
        "--fixed-source-split-manifest", args.fixed_source_split_manifest,
        "--archive-workers", args.archive_workers,
        "--obs-backend", "obsutil",
        "--obsutil-jobs", args.obsutil_jobs,
        "--save-raw-lane-image",
        "--raw-lane-separate-image",
        "--require-raw-lane",
        "--raw-lane-threshold", 0,
        "--pose-second-image",
        "--pose-threshold", 0,
        "--skip-upload",
    ]
    if args.obsutil_path:
        command.extend(["--obsutil-path", args.obsutil_path])
    if args.obsutil_config:
        command.extend(["--obsutil-config", args.obsutil_config])
    if args.skip_download:
        command.append("--skip-download")
    if args.keep_raw_source_after_stage:
        command.append("--keep-raw-source-after-stage")
    if args.resume:
        command.append("--resume")
    for source in args.source_obs_root or DEFAULT_SOURCE_OBS_ROOTS:
        command.extend(["--source-obs-root", source])
    return command


def rename_variant(output_root: Path, source: str, target: str, resume: bool) -> Path:
    source_root = output_root / source
    target_root = output_root / target
    if resume and variant_is_complete(target_root, source):
        print(f"[three-image-obs] reuse completed variant: {target_root}", flush=True)
        return target_root
    if target_root.exists():
        raise ValueError(f"target variant exists but is incomplete: {target_root}")
    if not source_root.is_dir():
        raise FileNotFoundError(f"source variant not found: {source_root}")
    source_root.rename(target_root)
    relabel_metadata(target_root / "dataset_info.json", source, target)
    return target_root


def validate_variant(
    root: Path,
    source_variant: str,
    sample_limit: int,
) -> dict:
    info = read_json(root / "dataset_info.json")
    balance = info.get("balance") or {}
    multi = info.get("multi_image_input") or {}
    overlay = info.get("input_overlay") or {}
    expected_counts = expected_difficulty_counts()
    errors = []
    if overlay.get("raw_lane_overlay") is not False:
        errors.append("input_overlay.raw_lane_overlay must be false")
    if overlay.get("raw_lane_separate_image") is not True:
        errors.append("input_overlay.raw_lane_separate_image must be true")
    if int(multi.get("num_images_per_sample", 0)) != 3:
        errors.append(f"num_images_per_sample={multi.get('num_images_per_sample')}, expected 3")
    if list(multi.get("image_roles") or []) != IMAGE_ROLES:
        errors.append(f"image_roles={multi.get('image_roles')!r}")
    if not bool(balance.get("strict_difficulty_quotas", False)):
        errors.append("strict_difficulty_quotas is false")
    if balance.get("final_bucket_counts") != expected_counts:
        errors.append(
            f"difficulty counts={balance.get('final_bucket_counts')}, expected={expected_counts}"
        )
    if abs(float(balance.get("actual_intersection_ratio", -1)) - INTERSECTION_RATIO) > 1e-8:
        errors.append(
            f"intersection ratio={balance.get('actual_intersection_ratio')}, "
            f"expected={INTERSECTION_RATIO}"
        )

    validated_assets = 0
    record_counts = Counter()
    expected_size = 256 if source_variant == "local256" else 512
    for split in ("train", "eval", "test"):
        jsonl = root / "phase_a" / f"{split}.jsonl"
        with jsonl.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record_counts[split] += 1
                if sample_limit > 0 and validated_assets >= sample_limit:
                    continue
                record = json.loads(line)
                images = record.get("images")
                expected_prefixes = (
                    f"images/{split}/",
                    f"raw_lane_images/{split}/",
                    f"pose_images/{split}/",
                )
                if not isinstance(images, list) or len(images) != 3:
                    raise ValueError(f"{jsonl}:{line_number} expected three images: {images!r}")
                for relative, prefix in zip(images, expected_prefixes):
                    if not str(relative).startswith(prefix):
                        raise ValueError(
                            f"{jsonl}:{line_number} invalid three-image order: {images!r}"
                        )
                    image_path = root / str(relative)
                    if not image_path.is_file():
                        raise FileNotFoundError(f"missing staged image: {image_path}")
                    with Image.open(image_path) as image:
                        if image.size != (expected_size, expected_size):
                            raise ValueError(
                                f"invalid image size={image.size}, expected={expected_size}: "
                                f"{image_path}"
                            )
                prompt = str((record.get("conversations") or [{}])[0].get("value", ""))
                if prompt.count("<image>") != 3:
                    raise ValueError(f"{jsonl}:{line_number} prompt does not contain 3 images")
                validated_assets += 1
    if record_counts["train"] != TARGET_SAMPLES:
        errors.append(f"train records={record_counts['train']}, expected={TARGET_SAMPLES}")
    if errors:
        raise ValueError(f"three-image dataset validation failed for {root}: {errors}")
    report = {
        "status": "passed",
        "dataset_root": str(root),
        "source_variant": source_variant,
        "record_counts": dict(record_counts),
        "validated_three_image_samples": validated_assets,
        "expected_image_size": expected_size,
        "difficulty_counts": expected_counts,
        "intersection_ratio": INTERSECTION_RATIO,
        "image_roles": IMAGE_ROLES,
        "raw_lane_overlay": False,
    }
    write_json(root / "three_image_validation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def create_named_tar(root: Path, package: Path, archive_root_name: str, resume: bool) -> None:
    latest_input = max(
        (root / "dataset_info.json").stat().st_mtime_ns,
        (root / "phase_a" / "train.jsonl").stat().st_mtime_ns,
    )
    if (
        resume
        and package.is_file()
        and package.stat().st_size > 0
        and package.stat().st_mtime_ns >= latest_input
    ):
        print(f"[three-image-obs] reuse package: {package}", flush=True)
        return
    package.parent.mkdir(parents=True, exist_ok=True)
    temporary = package.with_suffix(package.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    print(f"[three-image-obs] package {root} -> {package}", flush=True)
    with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as archive:
        archive.add(root, arcname=archive_root_name, recursive=True)
    temporary.replace(package)


def main(argv=None) -> None:
    args = parse_args(argv)
    work_root = Path(args.work_root).expanduser().resolve()
    manifest = Path(args.fixed_source_split_manifest).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"fixed split manifest not found: {manifest}")
    args.fixed_source_split_manifest = str(manifest)
    paths = {
        "work_root": work_root,
        "raw_root": work_root / "raw_sources",
        "staging_root": work_root / "staging_three_image",
        "output_root": work_root / "output_three_image_800k",
        "package_root": work_root / "packages_three_image_800k",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    target_roots = {
        source: paths["output_root"] / target
        for source, target in VARIANT_NAMES.items()
    }
    completed = args.resume and all(
        variant_is_complete(root, source)
        for source, root in target_roots.items()
    )
    if completed:
        print("[three-image-obs] reuse completed paired datasets", flush=True)
    else:
        incompatible = [root for root in target_roots.values() if root.exists()]
        if incompatible:
            raise ValueError(
                "renamed output exists but the paired build is incomplete; use a new "
                f"--work-root after reviewing it: {incompatible}"
            )
        run(streaming_command(args, paths))
        target_roots = {
            source: rename_variant(
                paths["output_root"],
                source,
                VARIANT_NAMES[source],
                args.resume,
            )
            for source in VARIANT_NAMES
        }

    validation = {}
    if not args.skip_validation:
        for source, root in target_roots.items():
            validation[source] = validate_variant(
                root,
                source,
                args.validation_sample_limit,
            )

    packages = []
    if not args.skip_package:
        for source, root in target_roots.items():
            target_name = VARIANT_NAMES[source]
            package = paths["package_root"] / f"{target_name}.tar"
            create_named_tar(root, package, target_name, args.resume)
            packages.append(str(package))

    summary = {
        "status": "passed",
        "work_root": str(work_root),
        "source_obs_roots": args.source_obs_root or DEFAULT_SOURCE_OBS_ROOTS,
        "datasets": {source: str(root) for source, root in target_roots.items()},
        "packages": packages,
        "target_samples": TARGET_SAMPLES,
        "difficulty_ratios": DIFFICULTY_RATIOS,
        "difficulty_counts": expected_difficulty_counts(),
        "intersection_ratio": INTERSECTION_RATIO,
        "image_roles": IMAGE_ROLES,
        "raw_lane_overlay": False,
        "validation": validation,
    }
    write_json(work_root / "three_image_800k_from_obs_build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
