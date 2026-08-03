#!/usr/bin/env python3
"""Build a fixed-eval 550k context512/ROI256 dataset from existing staging.

This entry never downloads or restages source data. It reuses the raw-lane +
pose bootstrap staging, applies the frozen large-map split manifest, validates
the completed two-image records, and writes a tar package.
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

from data_process.fixed_source_splits import load_fixed_source_split_manifest
from scripts.tools.build_rc_dataset_v2_from_obs import create_variant_tar
from scripts.tools.build_rc_dataset_v2_rawlane_pose_800k_windows import (
    completion_errors,
    rename_variant,
    validate_two_image_records,
)


TARGET_SAMPLES = 550_000
DIFFICULTY_RATIOS = "empty=0.05,easy=0.25,medium=0.33,hard=0.27,very_hard=0.10"
INTERSECTION_RATIO = 0.30
SOURCE_VARIANT = "context512_roi256"
TARGET_VARIANT = "rawlane_pose_context512_roi256_550k"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-root",
        default=r"D:\data\fulldata_rawlane_pose\staging_rawlane_pose_256_context",
        help="Completed bootstrap staging containing the context view.",
    )
    parser.add_argument(
        "--work-root",
        default=r"D:\data\fulldata_rawlane_pose_fixed_v1",
    )
    parser.add_argument("--output-root", default="")
    parser.add_argument("--package-root", default="")
    parser.add_argument(
        "--fixed-source-split-manifest",
        default=r"D:\data\fixed_splits\rc_fixed_large_maps_v1.json",
    )
    parser.add_argument("--difficulty-seed", type=int, default=20260713)
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--image-decode-mode", choices=["sampled", "all", "none"], default="sampled")
    parser.add_argument("--visualize-per-difficulty", type=int, default=0)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    work_root = Path(args.work_root).expanduser().resolve()
    return {
        "work_root": work_root,
        "staging_root": Path(args.staging_root).expanduser().resolve(),
        "output_root": (
            Path(args.output_root).expanduser().resolve()
            if args.output_root
            else work_root / "output_rawlane_pose_context512_roi256_550k"
        ),
        "package_root": (
            Path(args.package_root).expanduser().resolve()
            if args.package_root
            else work_root / "packages_rawlane_pose"
        ),
    }


def run(command: list[object]) -> None:
    normalized = [str(item) for item in command]
    print("[context550k] command:", shlex.join(normalized), flush=True)
    subprocess.run(normalized, cwd=REPO_ROOT, check=True)


def finalize_command(
    args: argparse.Namespace,
    staging_root: Path,
    output_root: Path,
    manifest_path: Path,
) -> list[object]:
    command: list[object] = [
        sys.executable,
        "data_process/build_dataset_v2_staged.py",
        "finalize",
        "--staging-root", staging_root,
        "--output-root", output_root,
        "--views", "context",
        "--patch-size", 256,
        "--context-size", 512,
        "--train-target-samples", TARGET_SAMPLES,
        "--difficulty-ratios", DIFFICULTY_RATIOS,
        "--intersection-target-ratio", INTERSECTION_RATIO,
        "--difficulty-seed", args.difficulty_seed,
        "--duplicate-policy", "last",
        "--copy-mode", args.copy_mode,
        "--fixed-source-split-manifest", manifest_path,
        "--repartition-existing-stages-by-fixed-manifest",
    ]
    if args.resume:
        command.append("--resume")
    return command


def validate_variant(root: Path, output_root: Path, args: argparse.Namespace) -> None:
    if args.skip_validation:
        return
    validate_two_image_records(root, expected_train_samples=TARGET_SAMPLES)
    run([
        sys.executable,
        "scripts/tools/validate_visualize_rc_dataset_v2.py",
        "--dataset-root", root,
        "--variant", SOURCE_VARIANT,
        "--expected-train-samples", TARGET_SAMPLES,
        "--difficulty-ratios", DIFFICULTY_RATIOS,
        "--expected-intersection-ratio", INTERSECTION_RATIO,
        "--output-dir", output_root / f"{TARGET_VARIANT}_validation",
        "--visualize-per-difficulty", args.visualize_per_difficulty,
        "--image-decode-mode", args.image_decode_mode,
        "--skip-distribution-check",
    ])


def main(argv=None) -> None:
    args = parse_args(argv)
    paths = resolve_paths(args)
    staging_root = paths["staging_root"]
    output_root = paths["output_root"]
    package_root = paths["package_root"]
    if not staging_root.is_dir():
        raise FileNotFoundError(f"bootstrap staging root not found: {staging_root}")

    manifest = load_fixed_source_split_manifest(args.fixed_source_split_manifest)
    manifest_path = Path(manifest["path"])
    manifest_sha256 = str(manifest["file_sha256"])
    output_root.mkdir(parents=True, exist_ok=True)
    package_root.mkdir(parents=True, exist_ok=True)
    target_root = output_root / TARGET_VARIANT

    print("============================================================", flush=True)
    print(f"[context550k] staging:    {staging_root}", flush=True)
    print(f"[context550k] manifest:   {manifest_path}", flush=True)
    print(f"[context550k] output:     {target_root}", flush=True)
    print(f"[context550k] train:      {TARGET_SAMPLES}", flush=True)
    print(f"[context550k] difficulty: {DIFFICULTY_RATIOS}", flush=True)
    print("[context550k] OBS download/restaging: disabled", flush=True)
    print("============================================================", flush=True)

    existing_errors = completion_errors(
        target_root,
        manifest_sha256,
        TARGET_SAMPLES,
        DIFFICULTY_RATIOS,
        INTERSECTION_RATIO,
    ) if target_root.is_dir() else ["target does not exist"]
    if args.resume and target_root.is_dir() and not existing_errors:
        print(f"[context550k] reuse completed dataset: {target_root}", flush=True)
    else:
        if target_root.exists():
            raise ValueError(
                f"target dataset exists but is incomplete or incompatible: {target_root}; "
                f"errors={existing_errors}. Use a new --output-root."
            )
        run(finalize_command(args, staging_root, output_root, manifest_path))
        target_root = rename_variant(
            output_root,
            SOURCE_VARIANT,
            TARGET_VARIANT,
            args.resume,
            manifest_sha256,
            target_samples=TARGET_SAMPLES,
            difficulty_ratios=DIFFICULTY_RATIOS,
            intersection_ratio=INTERSECTION_RATIO,
        )

    validate_variant(target_root, output_root, args)
    packages = []
    if not args.skip_package:
        package_path = package_root / f"{TARGET_VARIANT}.tar"
        create_variant_tar(target_root, package_path, args.resume)
        packages.append(str(package_path))

    summary = {
        "status": "passed",
        "dataset": str(target_root),
        "packages": packages,
        "target_samples": TARGET_SAMPLES,
        "difficulty_ratios": DIFFICULTY_RATIOS,
        "intersection_ratio": INTERSECTION_RATIO,
        "view": SOURCE_VARIANT,
        "input_image_size": 512,
        "supervised_roi": [128, 128, 384, 384],
        "images_per_sample": 2,
        "reused_staging_root": str(staging_root),
        "fixed_source_split_manifest": str(manifest_path),
        "fixed_source_split_sha256": manifest_sha256,
        "obs_download_performed": False,
        "source_restaging_performed": False,
    }
    summary_path = output_root / "rawlane_pose_context512_roi256_550k_build_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
