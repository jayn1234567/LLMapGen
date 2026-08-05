#!/usr/bin/env python3
"""Download the pre-cut context512 triplet dataset and build Dataset V2.

This Windows-oriented entrypoint performs one resumable pipeline:

1. download the pre-cut BEV/Pose/Raw-Lane image tree from OBS;
2. download the per-group GT JSON tree from OBS;
3. convert the embedded GT schema to three-image Stage A Dataset V2;
4. validate the output and create a TAR package.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.build_rc_dataset_v2_from_obs import resolve_obs_backend


DEFAULT_SOURCE_OBS_ROOT = (
    "obs://yw-ncasd-result-gy1/data/RCDataset/BaseModelTrain/"
    "sjn_context_512_roi_256/"
)
DEFAULT_GT_OBS_ROOT = DEFAULT_SOURCE_OBS_ROOT + "GT_json/"
DEFAULT_WINDOWS_WORK_ROOT = r"D:\data\sjn_context512_roi256_three_image_dataset_v2"
PIPELINE_VERSION = "sjn_context512_roi256_triplet_obs_pipeline_v1"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-obs-root", default=DEFAULT_SOURCE_OBS_ROOT)
    parser.add_argument("--gt-obs-root", default=DEFAULT_GT_OBS_ROOT)
    parser.add_argument(
        "--work-root",
        default=DEFAULT_WINDOWS_WORK_ROOT if os.name == "nt" else "/cache/sjn_context512_roi256_v2",
    )
    parser.add_argument("--download-root", default="")
    parser.add_argument("--source-local-root", default="")
    parser.add_argument("--gt-local-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--package-path", default="")
    parser.add_argument("--obs-backend", choices=("auto", "obsutil", "moxing"), default="auto")
    parser.add_argument("--obsutil-path", default="")
    parser.add_argument("--obsutil-config", default="")
    parser.add_argument("--obsutil-jobs", type=int, default=16)
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument(
        "--image-check-mode",
        choices=("sampled", "all", "none"),
        default="sampled",
    )
    parser.add_argument("--image-check-limit", type=int, default=10_000)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--default-split", choices=("train", "eval", "test"), default="train")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalized_obs_uri(value: str) -> str:
    text = str(value or "").strip()
    if not text.startswith("obs://"):
        raise ValueError(f"expected an OBS URI, got: {value!r}")
    return text.rstrip("/") + "/"


def read_download_marker(marker: Path, expected_source: str) -> bool:
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        marker_source = normalized_obs_uri(payload.get("source_obs_root", ""))
    except ValueError:
        return False
    return (
        payload.get("pipeline_version") == PIPELINE_VERSION
        and marker_source == expected_source
        and bool(payload.get("local_files_present"))
    )


def contains_local_files(root: Path) -> bool:
    return any(
        path.is_file() and not path.name.endswith(".partial")
        for path in root.rglob("*")
    )


def download_tree(
    backend,
    source_obs_root: str,
    destination: Path,
    marker: Path,
    resume: bool,
) -> dict:
    source_obs_root = normalized_obs_uri(source_obs_root)
    if resume and read_download_marker(marker, source_obs_root):
        print(f"[context512-triplet-obs] reuse completed download: {destination}", flush=True)
        return json.loads(marker.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    print(
        f"[context512-triplet-obs] download {source_obs_root} -> {destination}",
        flush=True,
    )
    backend.download_tree(source_obs_root, destination)
    local_files_present = contains_local_files(destination)
    if not local_files_present:
        raise FileNotFoundError(f"OBS download produced no files: {destination}")
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "source_obs_root": source_obs_root,
        "destination": str(destination),
        "obs_backend": backend.name,
        "local_files_present": local_files_present,
    }
    write_json(marker, payload)
    return payload


def find_gt_json_count(root: Path) -> int:
    count = 0
    for path in root.rglob("*.json"):
        if path.name.startswith(".obs_download_complete"):
            continue
        count += 1
    return count


def run_converter(args: argparse.Namespace, source_root: Path, gt_root: Path, output_root: Path, package: Path) -> None:
    converter = REPO_ROOT / "scripts" / "tools" / "convert_context512_roi_triplet_gt_to_dataset_v2.py"
    command = [
        sys.executable,
        str(converter),
        "--input-root", str(source_root),
        "--annotation-root", str(gt_root),
        "--output-root", str(output_root),
        "--default-split", args.default_split,
        "--copy-mode", args.copy_mode,
        "--image-check-mode", args.image_check_mode,
        "--image-check-limit", str(args.image_check_limit),
        "--progress-every", str(args.progress_every),
        "--package",
        "--package-path", str(package),
    ]
    if args.max_samples > 0:
        command.extend(("--max-samples", str(args.max_samples)))
    if args.resume:
        command.append("--resume")
    print(
        "[context512-triplet-obs] converter command:",
        shlex.join([str(item) for item in command]),
        flush=True,
    )
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def validate_completed_build(output_root: Path, package: Path) -> dict:
    required = (
        output_root / "dataset_info.json",
        output_root / "split_manifest.json",
        output_root / "build_summary.json",
        output_root / "conversion_validation.json",
        output_root / "phase_a" / "train.jsonl",
        output_root / "phase_a" / "eval.jsonl",
        output_root / "phase_a" / "test.jsonl",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"converted dataset is incomplete; missing={missing}")
    validation = json.loads(
        (output_root / "conversion_validation.json").read_text(encoding="utf-8")
    )
    if validation.get("status") != "passed":
        raise ValueError(f"conversion validation did not pass: {validation}")
    if not package.is_file() or package.stat().st_size <= 0:
        raise FileNotFoundError(f"TAR package was not created: {package}")
    return validation


def main(argv=None) -> None:
    args = parse_args(argv)
    args.source_obs_root = normalized_obs_uri(args.source_obs_root)
    args.gt_obs_root = normalized_obs_uri(args.gt_obs_root)
    work_root = Path(args.work_root).expanduser().resolve()
    download_root = (
        Path(args.download_root).expanduser().resolve()
        if args.download_root
        else work_root / "download"
    )
    source_root = (
        Path(args.source_local_root).expanduser().resolve()
        if args.source_local_root
        else download_root / "source"
    )
    gt_root = (
        Path(args.gt_local_root).expanduser().resolve()
        if args.gt_local_root
        else download_root / "GT_json"
    )
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else work_root / "output" / "context512_roi256_three_image_full"
    )
    package = (
        Path(args.package_path).expanduser().resolve()
        if args.package_path
        else work_root / "packages" / f"{output_root.name}.tar"
    )
    if output_root == source_root or output_root == gt_root:
        raise ValueError("output root must differ from the downloaded source and GT roots")
    work_root.mkdir(parents=True, exist_ok=True)

    download_reports = {}
    if not args.skip_download:
        backend = resolve_obs_backend(args)
        download_reports["source"] = download_tree(
            backend,
            args.source_obs_root,
            source_root,
            download_root / ".source_obs_download_complete.json",
            args.resume,
        )
        download_reports["gt"] = download_tree(
            backend,
            args.gt_obs_root,
            gt_root,
            download_root / ".gt_obs_download_complete.json",
            args.resume,
        )
    if not source_root.is_dir():
        raise FileNotFoundError(f"downloaded source root does not exist: {source_root}")
    gt_json_count = find_gt_json_count(gt_root)
    if gt_json_count <= 0:
        raise FileNotFoundError(f"downloaded GT root contains no JSON files: {gt_root}")

    if not args.skip_build:
        run_converter(args, source_root, gt_root, output_root, package)
    validation = validate_completed_build(output_root, package)
    summary = {
        "status": "passed",
        "pipeline_version": PIPELINE_VERSION,
        "source_obs_root": args.source_obs_root,
        "gt_obs_root": args.gt_obs_root,
        "source_local_root": str(source_root),
        "gt_local_root": str(gt_root),
        "gt_json_count": gt_json_count,
        "output_root": str(output_root),
        "package": str(package),
        "download_reports": download_reports,
        "conversion_validation": validation,
    }
    write_json(work_root / "pipeline_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
