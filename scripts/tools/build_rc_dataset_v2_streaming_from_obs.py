#!/usr/bin/env python3
"""Build Dataset V2 by downloading, staging, and deleting one source at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.build_rc_dataset_v2_from_obs import (
    DEFAULT_OUTPUT_OBS_ROOT,
    DEFAULT_SOURCE_OBS_ROOTS,
    resolve_obs_backend,
    source_name,
    upload_outputs,
)
from data_process.build_dataset_v2_staged import STAGE_VERSION, selected_variants


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-obs-root", action="append", default=[])
    parser.add_argument("--work-root", default="")
    parser.add_argument("--raw-root", default="")
    parser.add_argument("--staging-root", default="")
    parser.add_argument(
        "--secondary-local256-staging-root",
        default="",
        help="Optionally stage local256 from each downloaded source before deleting it.",
    )
    parser.add_argument("--output-root", default="")
    parser.add_argument("--output-obs-root", default=DEFAULT_OUTPUT_OBS_ROOT)
    parser.add_argument("--views", choices=["local", "context", "both"], default="local")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--eval-test-stride", type=int, default=0, help="Defaults to --patch-size.")
    parser.add_argument("--train-target-samples", type=int, default=550000)
    parser.add_argument("--train-stride", type=int, default=128)
    parser.add_argument("--secondary-local256-train-stride", type=int, default=128)
    parser.add_argument(
        "--raw-lane-overlay",
        action="store_true",
        help="Overlay patch_tif/0_lane.tif as white raw-lane pixels on top of every input image.",
    )
    parser.add_argument(
        "--require-raw-lane",
        action="store_true",
        help="Fail if --raw-lane-overlay is enabled and patch_tif/0_lane.tif is missing.",
    )
    parser.add_argument("--raw-lane-threshold", type=float, default=0.0)
    parser.add_argument(
        "--pose-second-image",
        action="store_true",
        help="Add patch_tif/0_pose.tif as a separate second image for every sample.",
    )
    parser.add_argument("--pose-threshold", type=float, default=0.0)
    parser.add_argument(
        "--difficulty-ratios",
        default="empty=0,easy=0.30,medium=0.33,hard=0.27,very_hard=0.10",
    )
    parser.add_argument("--intersection-target-ratio", type=float, default=0.30)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--fixed-source-split-manifest",
        default=os.environ.get("RC_FIXED_SOURCE_SPLIT_MANIFEST", ""),
        help="Explicit raw_sample_id eval/test manifest shared by every dataset variant.",
    )
    parser.add_argument("--allow-missing-fixed-holdouts", action="store_true")
    parser.add_argument("--difficulty-seed", type=int, default=20260713)
    parser.add_argument("--duplicate-policy", choices=["last", "first", "error"], default="last")
    parser.add_argument("--archive-workers", type=int, default=1)
    parser.add_argument("--obs-backend", choices=["auto", "moxing", "obsutil"], default="auto")
    parser.add_argument("--obsutil-path", default="")
    parser.add_argument("--obsutil-config", default="")
    parser.add_argument("--obsutil-jobs", type=int, default=8)
    parser.add_argument("--upload-mode", choices=["tar", "directory"], default="tar")
    parser.add_argument("--remove-package-after-upload", action="store_true")
    parser.add_argument("--keep-raw-source-after-stage", action="store_true")
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument(
        "--train-candidate-jsonl",
        default="",
        help="Optional completed train JSONL whose ids limit train candidates rendered by each source stage.",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-finalize", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-samples", type=int, default=None)
    return parser.parse_args(argv)


def run(command):
    print("[dataset-v2-stream] command:", shlex.join([str(item) for item in command]), flush=True)
    subprocess.run([str(item) for item in command], cwd=REPO_ROOT, check=True)


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def completed_stage(
    stage_root: Path,
    expected_candidate_sha256: str = "",
    expected_variants: list[str] | None = None,
    expected_patch_size: int | None = None,
    expected_train_stride: int | None = None,
    expected_eval_test_stride: int | None = None,
    expected_raw_lane_overlay: bool | None = None,
    expected_require_raw_lane: bool | None = None,
    expected_raw_lane_threshold: float | None = None,
    expected_pose_second_image: bool | None = None,
    expected_pose_threshold: float | None = None,
    expected_fixed_split_sha256: str | None = None,
) -> bool:
    marker = stage_root / "stage_complete.json"
    if not marker.is_file():
        return False
    payload = json.loads(marker.read_text(encoding="utf-8"))
    complete = (
        payload.get("stage_version") == STAGE_VERSION
        and payload.get("semantic_validation_passed") is True
        and int(payload.get("raw_sample_count", 0)) > 0
        and sum(payload.get("split_record_counts", {}).values()) > 0
    )
    if complete:
        candidate_filter = payload.get("train_candidate_filter") or {}
        complete = str(candidate_filter.get("sha256") or "") == expected_candidate_sha256
    if complete and expected_variants is not None:
        complete = all(name in payload.get("variants", []) for name in expected_variants)
    if complete and expected_patch_size is not None:
        marker_patch_size = payload.get("target_patch_size")
        complete = (
            int(marker_patch_size) == expected_patch_size
            if marker_patch_size is not None
            else expected_patch_size == 256
        )
    if complete and expected_train_stride is not None:
        marker_train_stride = payload.get("train_stride")
        complete = marker_train_stride is not None and int(marker_train_stride) == expected_train_stride
    if complete and expected_eval_test_stride is not None:
        marker_eval_test_stride = payload.get("eval_test_stride")
        complete = marker_eval_test_stride is not None and int(marker_eval_test_stride) == expected_eval_test_stride
    if complete and expected_raw_lane_overlay is not None:
        complete = bool(payload.get("raw_lane_overlay", False)) == bool(expected_raw_lane_overlay)
    if complete and expected_require_raw_lane is not None:
        complete = bool(payload.get("require_raw_lane", False)) == bool(expected_require_raw_lane)
    if complete and expected_raw_lane_threshold is not None:
        complete = abs(float(payload.get("raw_lane_threshold", 0.0)) - float(expected_raw_lane_threshold)) <= 1e-12
    if complete and expected_pose_second_image is not None:
        complete = bool(payload.get("pose_second_image", False)) == bool(expected_pose_second_image)
    if complete and expected_pose_threshold is not None:
        complete = abs(float(payload.get("pose_threshold", 0.0)) - float(expected_pose_threshold)) <= 1e-12
    if complete and expected_fixed_split_sha256 is not None:
        fixed = payload.get("fixed_source_split") or {}
        complete = str(fixed.get("file_sha256") or "") == expected_fixed_split_sha256
    return complete


def remove_stale_stage(stage_root: Path, staging_root: Path) -> None:
    resolved_stage = stage_root.resolve()
    resolved_staging = staging_root.resolve()
    if resolved_stage == resolved_staging:
        raise ValueError(f"refusing to remove staging root itself: {resolved_stage}")
    try:
        resolved_stage.relative_to(resolved_staging)
    except ValueError as exc:
        raise ValueError(f"stale stage is outside staging root: {resolved_stage}") from exc
    shutil.rmtree(resolved_stage)
    print(f"[dataset-v2-stream] removed stale generated stage: {resolved_stage}", flush=True)


def download_one_source(source, local_root: Path, resume: bool, backend):
    marker = local_root / ".obs_download_complete.json"
    if resume and marker.is_file():
        print(f"[dataset-v2-stream] reuse completed download: {local_root}", flush=True)
        return
    local_root.mkdir(parents=True, exist_ok=True)
    print(f"[dataset-v2-stream] download one source: {source} -> {local_root}", flush=True)
    backend.download_tree(source, local_root)
    marker.write_text(
        json.dumps({"source": source, "local_root": str(local_root)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_stage_command(
    args,
    local_root: Path,
    stage_root: Path,
    raw_root: Path,
    source_index: int,
    source: str,
    patch_size: int,
    context_size: int,
    eval_test_stride: int,
    train_stride: int,
    candidate_jsonl: Path | None,
    delete_input: bool,
    views: str | None = None,
) -> list:
    command = [
        sys.executable,
        "data_process/build_dataset_v2_staged.py",
        "stage",
        "--input-root", local_root,
        "--stage-root", stage_root,
        "--source-index", source_index,
        "--source-uri", source,
        "--views", views or args.views,
        "--patch-size", patch_size,
        "--context-size", context_size,
        "--stride", eval_test_stride,
        "--split-seed", args.split_seed,
        "--train-stride", train_stride,
        "--archive-workers", args.archive_workers,
        "--selective-archive-extract",
    ]
    if args.raw_lane_overlay:
        command.append("--raw-lane-overlay")
    if args.require_raw_lane:
        command.append("--require-raw-lane")
    command.extend(["--raw-lane-threshold", args.raw_lane_threshold])
    if args.pose_second_image:
        command.append("--pose-second-image")
    command.extend(["--pose-threshold", args.pose_threshold])
    fixed_manifest = str(getattr(args, "fixed_source_split_manifest", "") or "").strip()
    if fixed_manifest:
        command.extend(["--fixed-source-split-manifest", fixed_manifest])
    if args.resume:
        command.append("--resume")
    if args.keep_archives:
        command.append("--keep-archives")
    if candidate_jsonl:
        command.extend(["--train-candidate-jsonl", candidate_jsonl])
    if args.limit_samples is not None:
        command.extend(["--limit-samples", args.limit_samples])
    if delete_input:
        command.extend([
            "--delete-input-root-after-stage",
            "--delete-root-parent", raw_root,
        ])
    return command


def main(argv=None):
    args = parse_args(argv)
    sources = args.source_obs_root or list(DEFAULT_SOURCE_OBS_ROOTS)
    if not args.work_root:
        if os.name == "nt":
            raise ValueError("--work-root is required on Windows, for example D:\\data\\fulldata")
        args.work_root = "/cache/jn/rc_dataset_v2_streaming"
    work_root = Path(args.work_root)
    raw_root = Path(args.raw_root) if args.raw_root else work_root / "raw_sources"
    staging_root = Path(args.staging_root) if args.staging_root else work_root / "staging"
    secondary_staging_root = (
        Path(args.secondary_local256_staging_root)
        if args.secondary_local256_staging_root
        else None
    )
    output_root = Path(args.output_root) if args.output_root else work_root / "output"
    candidate_jsonl = Path(args.train_candidate_jsonl) if args.train_candidate_jsonl else None
    candidate_filter_sha256 = file_sha256(candidate_jsonl) if candidate_jsonl else ""
    fixed_manifest = (
        Path(args.fixed_source_split_manifest).expanduser().resolve()
        if args.fixed_source_split_manifest else None
    )
    if fixed_manifest is not None and not fixed_manifest.is_file():
        raise FileNotFoundError(f"fixed source split manifest not found: {fixed_manifest}")
    if fixed_manifest is not None:
        args.fixed_source_split_manifest = str(fixed_manifest)
    fixed_split_sha256 = file_sha256(fixed_manifest) if fixed_manifest else ""
    eval_test_stride = args.eval_test_stride or args.patch_size
    expected_variants = selected_variants(args.views, args.patch_size, args.context_size)
    paths = [work_root, raw_root, staging_root, output_root]
    if secondary_staging_root is not None:
        paths.append(secondary_staging_root)
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

    needs_backend = not args.skip_download or (not args.skip_upload and bool(args.output_obs_root))
    backend = resolve_obs_backend(args) if needs_backend else None
    print("============================================================", flush=True)
    print(f"[dataset-v2-stream] work root:      {work_root}", flush=True)
    print(f"[dataset-v2-stream] raw root:       {raw_root}", flush=True)
    print(f"[dataset-v2-stream] staging root:   {staging_root}", flush=True)
    print(f"[dataset-v2-stream] output root:    {output_root}", flush=True)
    print(f"[dataset-v2-stream] sources:        {len(sources)}", flush=True)
    print(f"[dataset-v2-stream] views:          {args.views}", flush=True)
    print(f"[dataset-v2-stream] target patch:   {args.patch_size}", flush=True)
    print(
        f"[dataset-v2-stream] raw lane:       overlay={args.raw_lane_overlay} "
        f"require={args.require_raw_lane} threshold={args.raw_lane_threshold}",
        flush=True,
    )
    print(
        f"[dataset-v2-stream] pose image:     second={args.pose_second_image} "
        f"threshold={args.pose_threshold}",
        flush=True,
    )
    if secondary_staging_root is not None:
        print(f"[dataset-v2-stream] local256 stage: {secondary_staging_root}", flush=True)
        print(
            f"[dataset-v2-stream] local256 stride:{args.secondary_local256_train_stride}",
            flush=True,
        )
    print(f"[dataset-v2-stream] target records: {args.train_target_samples}", flush=True)
    print(
        f"[dataset-v2-stream] fixed split:     {fixed_manifest or '<disabled>'}",
        flush=True,
    )
    print(f"[dataset-v2-stream] free disk:      {shutil.disk_usage(work_root).free / (1024 ** 3):.1f} GiB", flush=True)
    print("============================================================", flush=True)

    for source_index, source in enumerate(sources):
        name = source_name(source)
        local_root = raw_root / f"{source_index:02d}_{name}"
        stage_root = staging_root / f"{source_index:02d}_{name}"
        marker = stage_root / "stage_complete.json"
        secondary_stage_root = (
            secondary_staging_root / f"{source_index:02d}_{name}"
            if secondary_staging_root is not None
            else None
        )
        secondary_marker = (
            secondary_stage_root / "stage_complete.json"
            if secondary_stage_root is not None
            else None
        )
        if args.resume and marker.is_file() and not completed_stage(
            stage_root,
            candidate_filter_sha256,
            expected_variants,
            args.patch_size,
            args.train_stride,
            eval_test_stride,
            args.raw_lane_overlay,
            args.require_raw_lane,
            args.raw_lane_threshold,
            args.pose_second_image,
            args.pose_threshold,
            fixed_split_sha256,
        ):
            print(
                f"[dataset-v2-stream] stale stage lacks {STAGE_VERSION}; it must be rebuilt: {stage_root}",
                flush=True,
            )
            remove_stale_stage(stage_root, staging_root)
        if (
            args.resume
            and secondary_marker is not None
            and secondary_marker.is_file()
            and not completed_stage(
                secondary_stage_root,
                candidate_filter_sha256,
                ["local256"],
                256,
                args.secondary_local256_train_stride,
                256,
                args.raw_lane_overlay,
                args.require_raw_lane,
                args.raw_lane_threshold,
                args.pose_second_image,
                args.pose_threshold,
                fixed_split_sha256,
            )
        ):
            print(
                f"[dataset-v2-stream] stale secondary local256 stage must be rebuilt: "
                f"{secondary_stage_root}",
                flush=True,
            )
            remove_stale_stage(secondary_stage_root, secondary_staging_root)

        primary_complete = args.resume and completed_stage(
            stage_root,
            candidate_filter_sha256,
            expected_variants,
            args.patch_size,
            args.train_stride,
            eval_test_stride,
            args.raw_lane_overlay,
            args.require_raw_lane,
            args.raw_lane_threshold,
            args.pose_second_image,
            args.pose_threshold,
            fixed_split_sha256,
        )
        secondary_complete = secondary_stage_root is None or (
            args.resume
            and completed_stage(
                secondary_stage_root,
                candidate_filter_sha256,
                ["local256"],
                256,
                args.secondary_local256_train_stride,
                256,
                args.raw_lane_overlay,
                args.require_raw_lane,
                args.raw_lane_threshold,
                args.pose_second_image,
                args.pose_threshold,
                fixed_split_sha256,
            )
        )
        if primary_complete and secondary_complete:
            print(f"[dataset-v2-stream] reuse completed primary stage: {stage_root}", flush=True)
            if secondary_stage_root is not None:
                print(
                    f"[dataset-v2-stream] reuse completed local256 stage: {secondary_stage_root}",
                    flush=True,
                )
            if local_root.exists() and not args.keep_raw_source_after_stage:
                run(build_stage_command(
                    args,
                    local_root,
                    stage_root,
                    raw_root,
                    source_index,
                    source,
                    args.patch_size,
                    args.context_size,
                    eval_test_stride,
                    args.train_stride,
                    candidate_jsonl,
                    True,
                ))
            continue

        if args.skip_download:
            if not local_root.exists():
                raise FileNotFoundError(f"local source root is missing: {local_root}")
        else:
            download_one_source(source, local_root, args.resume, backend)

        if not primary_complete:
            delete_after_primary = (
                not args.keep_raw_source_after_stage and secondary_complete
            )
            run(build_stage_command(
                args,
                local_root,
                stage_root,
                raw_root,
                source_index,
                source,
                args.patch_size,
                args.context_size,
                eval_test_stride,
                args.train_stride,
                candidate_jsonl,
                delete_after_primary,
            ))
            primary_complete = completed_stage(
                stage_root,
                candidate_filter_sha256,
                expected_variants,
                args.patch_size,
                args.train_stride,
                eval_test_stride,
                args.raw_lane_overlay,
                args.require_raw_lane,
                args.raw_lane_threshold,
                args.pose_second_image,
                args.pose_threshold,
                fixed_split_sha256,
            )
            if not primary_complete:
                raise RuntimeError(f"source stage validation failed: {stage_root}")

        if secondary_stage_root is not None and not secondary_complete:
            if not local_root.exists():
                raise FileNotFoundError(
                    f"raw source was removed before local256 staging completed: {local_root}"
                )
            run(build_stage_command(
                args,
                local_root,
                secondary_stage_root,
                raw_root,
                source_index,
                source,
                256,
                256,
                256,
                args.secondary_local256_train_stride,
                candidate_jsonl,
                not args.keep_raw_source_after_stage,
                "local",
            ))
            secondary_complete = completed_stage(
                secondary_stage_root,
                candidate_filter_sha256,
                ["local256"],
                256,
                args.secondary_local256_train_stride,
                256,
                args.raw_lane_overlay,
                args.require_raw_lane,
                args.raw_lane_threshold,
                args.pose_second_image,
                args.pose_threshold,
                fixed_split_sha256,
            )
            if not secondary_complete:
                raise RuntimeError(
                    f"secondary local256 source stage validation failed: {secondary_stage_root}"
                )
        print(
            f"[dataset-v2-stream] free disk after source {source_index}: "
            f"{shutil.disk_usage(work_root).free / (1024 ** 3):.1f} GiB",
            flush=True,
        )

    if not args.skip_finalize:
        finalize_command = [
            sys.executable,
            "data_process/build_dataset_v2_staged.py",
            "finalize",
            "--staging-root", staging_root,
            "--output-root", output_root,
            "--views", args.views,
            "--patch-size", args.patch_size,
            "--context-size", args.context_size,
            "--train-target-samples", args.train_target_samples,
            "--difficulty-ratios", args.difficulty_ratios,
            "--intersection-target-ratio", args.intersection_target_ratio,
            "--difficulty-seed", args.difficulty_seed,
            "--duplicate-policy", args.duplicate_policy,
            "--copy-mode", args.copy_mode,
        ]
        if args.resume:
            finalize_command.append("--resume")
        if fixed_manifest is not None:
            finalize_command.extend(["--fixed-source-split-manifest", fixed_manifest])
        if args.allow_missing_fixed_holdouts:
            finalize_command.append("--allow-missing-fixed-holdouts")
        run(finalize_command)

    if not args.skip_upload:
        if not args.output_obs_root:
            print("[dataset-v2-stream] empty output OBS root; skip upload", flush=True)
        else:
            upload_outputs(
                output_root,
                args.output_obs_root,
                args.views,
                args.upload_mode,
                args.resume,
                backend,
                args.remove_package_after_upload,
            )
    print("[dataset-v2-stream] done", flush=True)


if __name__ == "__main__":
    main()
