#!/usr/bin/env python3
"""One-command bootstrap and build for fixed-eval raw-lane + pose datasets."""

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
from scripts.tools.build_rc_dataset_v2_from_obs import DEFAULT_SOURCE_OBS_ROOTS


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-work-root",
        default=r"D:\data\fulldata_rawlane_pose",
        help="Temporary hash-split staging used only to select fixed large maps.",
    )
    parser.add_argument(
        "--fixed-work-root",
        default=r"D:\data\fulldata_rawlane_pose_fixed_v1",
        help="Final output root built with the frozen large-map manifest.",
    )
    parser.add_argument(
        "--manifest-path",
        default=r"D:\data\fixed_splits\rc_fixed_large_maps_v1.json",
    )
    parser.add_argument("--eval-count", type=int, default=14)
    parser.add_argument("--test-count", type=int, default=7)
    parser.add_argument("--selection-seed", type=int, default=20260731)
    parser.add_argument("--source-obs-root", action="append", default=[])
    parser.add_argument("--obs-backend", choices=["auto", "moxing", "obsutil"], default="auto")
    parser.add_argument("--obsutil-path", default="")
    parser.add_argument("--obsutil-config", default="")
    parser.add_argument("--obsutil-jobs", type=int, default=8)
    parser.add_argument("--archive-workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--overwrite-manifest",
        action="store_true",
        help="Deliberately replace an existing manifest. Do not use after benchmark adoption.",
    )
    return parser.parse_args(argv)


def run(command: list[object]) -> None:
    command = [str(item) for item in command]
    print("[fixed-eval-build] command:", shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def common_obs_args(args: argparse.Namespace) -> list[object]:
    command: list[object] = [
        "--obs-backend", args.obs_backend,
        "--obsutil-jobs", args.obsutil_jobs,
        "--archive-workers", args.archive_workers,
    ]
    if args.obsutil_path:
        command.extend(["--obsutil-path", args.obsutil_path])
    if args.obsutil_config:
        command.extend(["--obsutil-config", args.obsutil_config])
    for source in args.source_obs_root or DEFAULT_SOURCE_OBS_ROOTS:
        command.extend(["--source-obs-root", source])
    if args.resume:
        command.append("--resume")
    return command


def bootstrap_command(args: argparse.Namespace) -> list[object]:
    return [
        sys.executable,
        "scripts/tools/build_rc_dataset_v2_rawlane_pose_800k_windows.py",
        "--work-root", Path(args.bootstrap_work_root).expanduser().resolve(),
        "--stage-only",
        *common_obs_args(args),
    ]


def create_manifest_command(args: argparse.Namespace, staging_root: Path, manifest_path: Path) -> list[object]:
    command: list[object] = [
        sys.executable,
        "scripts/tools/create_fixed_source_split_manifest.py",
        "--staging-root", staging_root,
        "--output", manifest_path,
        "--eval-count", args.eval_count,
        "--test-count", args.test_count,
        "--seed", args.selection_seed,
    ]
    if args.overwrite_manifest:
        command.append("--overwrite")
    return command


def final_build_command(args: argparse.Namespace, manifest_path: Path) -> list[object]:
    return [
        sys.executable,
        "scripts/tools/build_rc_dataset_v2_rawlane_pose_800k_windows.py",
        "--work-root", Path(args.fixed_work_root).expanduser().resolve(),
        "--fixed-source-split-manifest", manifest_path,
        *common_obs_args(args),
    ]


def main(argv=None) -> None:
    args = parse_args(argv)
    bootstrap_root = Path(args.bootstrap_work_root).expanduser().resolve()
    fixed_root = Path(args.fixed_work_root).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    if bootstrap_root == fixed_root:
        raise ValueError("--bootstrap-work-root and --fixed-work-root must be different")
    if manifest_path.is_file() and not args.overwrite_manifest:
        manifest = load_fixed_source_split_manifest(manifest_path)
        print(
            f"[fixed-eval-build] reuse frozen manifest: {manifest_path} "
            f"id={manifest['manifest_id']}",
            flush=True,
        )
    else:
        run(bootstrap_command(args))
        staging_root = bootstrap_root / "staging_rawlane_pose_256_context"
        run(create_manifest_command(args, staging_root, manifest_path))
        manifest = load_fixed_source_split_manifest(manifest_path)
    run(final_build_command(args, manifest_path))
    summary = {
        "status": "passed",
        "manifest": str(manifest_path),
        "manifest_id": manifest["manifest_id"],
        "bootstrap_work_root": str(bootstrap_root),
        "fixed_work_root": str(fixed_root),
        "datasets": [
            str(fixed_root / "output_rawlane_pose_256_context" / "rawlane_pose_local256_800k"),
            str(
                fixed_root
                / "output_rawlane_pose_256_context"
                / "rawlane_pose_context512_roi256_800k"
            ),
        ],
        "packages": [
            str(fixed_root / "packages_rawlane_pose" / "rawlane_pose_local256_800k.tar"),
            str(
                fixed_root
                / "packages_rawlane_pose"
                / "rawlane_pose_context512_roi256_800k.tar"
            ),
        ],
    }
    summary_path = fixed_root / "fixed_eval_one_command_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
