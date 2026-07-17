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
from data_process.build_dataset_v2_staged import STAGE_VERSION


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-obs-root", action="append", default=[])
    parser.add_argument("--work-root", default="")
    parser.add_argument("--raw-root", default="")
    parser.add_argument("--staging-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--output-obs-root", default=DEFAULT_OUTPUT_OBS_ROOT)
    parser.add_argument("--views", choices=["local", "context", "both"], default="local")
    parser.add_argument("--train-target-samples", type=int, default=550000)
    parser.add_argument("--train-stride", type=int, default=128)
    parser.add_argument(
        "--difficulty-ratios",
        default="empty=0,easy=0.30,medium=0.33,hard=0.27,very_hard=0.10",
    )
    parser.add_argument("--intersection-target-ratio", type=float, default=0.30)
    parser.add_argument("--split-seed", type=int, default=42)
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


def completed_stage(stage_root: Path, expected_candidate_sha256: str = "") -> bool:
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
    output_root = Path(args.output_root) if args.output_root else work_root / "output"
    candidate_jsonl = Path(args.train_candidate_jsonl) if args.train_candidate_jsonl else None
    candidate_filter_sha256 = file_sha256(candidate_jsonl) if candidate_jsonl else ""
    for path in (work_root, raw_root, staging_root, output_root):
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
    print(f"[dataset-v2-stream] target records: {args.train_target_samples}", flush=True)
    print(f"[dataset-v2-stream] free disk:      {shutil.disk_usage(work_root).free / (1024 ** 3):.1f} GiB", flush=True)
    print("============================================================", flush=True)

    for source_index, source in enumerate(sources):
        name = source_name(source)
        local_root = raw_root / f"{source_index:02d}_{name}"
        stage_root = staging_root / f"{source_index:02d}_{name}"
        marker = stage_root / "stage_complete.json"
        if args.resume and marker.is_file() and not completed_stage(stage_root, candidate_filter_sha256):
            print(
                f"[dataset-v2-stream] stale stage lacks {STAGE_VERSION}; it must be rebuilt: {stage_root}",
                flush=True,
            )
            remove_stale_stage(stage_root, staging_root)
        if args.resume and completed_stage(stage_root, candidate_filter_sha256):
            print(f"[dataset-v2-stream] reuse completed stage: {stage_root}", flush=True)
            if local_root.exists() and not args.keep_raw_source_after_stage:
                cleanup_command = [
                    sys.executable,
                    "data_process/build_dataset_v2_staged.py",
                    "stage",
                    "--input-root", local_root,
                    "--stage-root", stage_root,
                    "--source-index", source_index,
                    "--source-uri", source,
                    "--views", args.views,
                    "--resume",
                    "--delete-input-root-after-stage",
                    "--delete-root-parent", raw_root,
                ]
                if candidate_jsonl:
                    cleanup_command.extend(["--train-candidate-jsonl", candidate_jsonl])
                run(cleanup_command)
            continue

        if args.skip_download:
            if not local_root.exists():
                raise FileNotFoundError(f"local source root is missing: {local_root}")
        else:
            download_one_source(source, local_root, args.resume, backend)

        stage_command = [
            sys.executable,
            "data_process/build_dataset_v2_staged.py",
            "stage",
            "--input-root", local_root,
            "--stage-root", stage_root,
            "--source-index", source_index,
            "--source-uri", source,
            "--views", args.views,
            "--split-seed", args.split_seed,
            "--train-stride", args.train_stride,
            "--archive-workers", args.archive_workers,
            "--selective-archive-extract",
        ]
        if args.resume:
            stage_command.append("--resume")
        if args.keep_archives:
            stage_command.append("--keep-archives")
        if candidate_jsonl:
            stage_command.extend(["--train-candidate-jsonl", candidate_jsonl])
        if args.limit_samples is not None:
            stage_command.extend(["--limit-samples", args.limit_samples])
        if not args.keep_raw_source_after_stage:
            stage_command.extend([
                "--delete-input-root-after-stage",
                "--delete-root-parent", raw_root,
            ])
        run(stage_command)
        if not completed_stage(stage_root, candidate_filter_sha256):
            raise RuntimeError(f"source stage validation failed: {stage_root}")
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
            "--train-target-samples", args.train_target_samples,
            "--difficulty-ratios", args.difficulty_ratios,
            "--intersection-target-ratio", args.intersection_target_ratio,
            "--difficulty-seed", args.difficulty_seed,
            "--duplicate-policy", args.duplicate_policy,
        ]
        if args.resume:
            finalize_command.append("--resume")
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
