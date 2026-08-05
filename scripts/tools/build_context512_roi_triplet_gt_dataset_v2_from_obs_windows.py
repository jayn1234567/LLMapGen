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
import concurrent.futures
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import zipfile
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
DEFAULT_DIFFICULTY_RATIOS = (
    "empty=0.05,easy=0.25,medium=0.33,hard=0.27,very_hard=0.10"
)
DEFAULT_EMPTY_DONOR_CLEAN_STAGING = r"D:\data\fulldata_context512\staging_context512"
DEFAULT_EMPTY_DONOR_AUX_STAGING = (
    r"D:\data\fulldata_rawlane_pose\staging_rawlane_pose_256_context"
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-obs-root", default=DEFAULT_SOURCE_OBS_ROOT)
    parser.add_argument("--gt-obs-root", default=DEFAULT_GT_OBS_ROOT)
    parser.add_argument(
        "--work-root",
        default=DEFAULT_WINDOWS_WORK_ROOT if os.name == "nt" else "/cache/sjn_context512_roi256_v2",
    )
    parser.add_argument("--download-root", default="")
    parser.add_argument("--extract-root", default="")
    parser.add_argument("--source-local-root", default="")
    parser.add_argument("--gt-local-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument(
        "--balanced-output-root",
        default="",
        help="Final exact-ratio Dataset V2 root; the converted full pool remains reusable.",
    )
    parser.add_argument("--package-path", default="")
    parser.add_argument("--obs-backend", choices=("auto", "obsutil", "moxing"), default="auto")
    parser.add_argument("--obsutil-path", default="")
    parser.add_argument("--obsutil-config", default="")
    parser.add_argument("--obsutil-jobs", type=int, default=16)
    parser.add_argument(
        "--archive-workers",
        type=int,
        default=8,
        help="Number of downloaded TAR/ZIP packages extracted concurrently.",
    )
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument(
        "--image-check-mode",
        choices=("sampled", "all", "none"),
        default="sampled",
    )
    parser.add_argument("--image-check-limit", type=int, default=10_000)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--train-target-samples", type=int, default=800_000)
    parser.add_argument("--difficulty-ratios", default=DEFAULT_DIFFICULTY_RATIOS)
    parser.add_argument("--balance-seed", type=int, default=20260713)
    parser.add_argument(
        "--empty-donor-clean-staging-root",
        default=DEFAULT_EMPTY_DONOR_CLEAN_STAGING if os.name == "nt" else "",
    )
    parser.add_argument(
        "--empty-donor-aux-staging-root",
        default=DEFAULT_EMPTY_DONOR_AUX_STAGING if os.name == "nt" else "",
    )
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


def archive_kind(path: Path) -> str | None:
    name = path.name.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        return "tar"
    if name.endswith(".zip"):
        return "zip"
    return None


def discover_archives(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and archive_kind(path) is not None
    )


def archive_target(archive: Path, source_root: Path, extract_root: Path) -> Path:
    relative = archive.relative_to(source_root).as_posix()
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
    label = archive.name
    for suffix in (".tar.gz", ".tgz", ".tar", ".zip"):
        if label.lower().endswith(suffix):
            label = label[:-len(suffix)]
            break
    safe_label = "".join(char if char.isalnum() or char in "-_" else "_" for char in label)
    return extract_root / f"{digest}_{safe_label or 'archive'}"


def remove_generated_tree(path: Path, extract_root: Path) -> None:
    resolved = path.resolve()
    root_resolved = extract_root.resolve()
    if resolved == root_resolved:
        raise ValueError(f"refusing to remove extraction root: {resolved}")
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"generated extraction path is outside extraction root: {resolved}") from exc
    if path.exists():
        shutil.rmtree(path)


def marker_extraction_status(payload: dict) -> str | None:
    status = payload.get("extraction_status")
    if status in {"extracted", "empty"}:
        return status
    # Backward compatibility with markers written before empty archives were
    # explicitly represented. A true file-presence flag meant success.
    if payload.get("extracted_files_present") is True:
        return "extracted"
    return None


def read_matching_extracted_marker(marker: Path, archive: Path) -> dict | None:
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    stat = archive.stat()
    matches = (
        payload.get("pipeline_version") == PIPELINE_VERSION
        and int(payload.get("archive_size", -1)) == stat.st_size
        and int(payload.get("archive_mtime_ns", -1)) == stat.st_mtime_ns
        and marker_extraction_status(payload) is not None
    )
    return payload if matches else None


def safe_extract_zip(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"unsafe ZIP member path in {archive}: {member.filename}") from exc
        handle.extractall(destination)


def extract_one_archive(
    archive: Path,
    source_root: Path,
    extract_root: Path,
    resume: bool,
) -> dict:
    target = archive_target(archive, source_root, extract_root)
    marker = target / ".archive_extract_complete.json"
    marker_payload = read_matching_extracted_marker(marker, archive) if resume else None
    if marker_payload is not None:
        reused_status = (
            "reused_empty"
            if marker_extraction_status(marker_payload) == "empty"
            else "reused"
        )
        return {"archive": str(archive), "target": str(target), "status": reused_status}
    partial = target.with_name(target.name + ".partial")
    remove_generated_tree(partial, extract_root)
    remove_generated_tree(target, extract_root)
    partial.mkdir(parents=True, exist_ok=True)
    print(f"[context512-triplet-obs] extract {archive} -> {target}", flush=True)
    try:
        if archive_kind(archive) == "zip":
            safe_extract_zip(archive, partial)
        else:
            with tarfile.open(archive, "r:*") as handle:
                handle.extractall(partial, filter="data")
        extracted_files_present = contains_local_files(partial)
        partial.replace(target)
        stat = archive.stat()
        extraction_status = "extracted" if extracted_files_present else "empty"
        write_json(target / ".archive_extract_complete.json", {
            "pipeline_version": PIPELINE_VERSION,
            "archive": str(archive),
            "archive_size": stat.st_size,
            "archive_mtime_ns": stat.st_mtime_ns,
            "target": str(target),
            "extracted_files_present": extracted_files_present,
            "extraction_status": extraction_status,
        })
        if not extracted_files_present:
            print(
                f"[context512-triplet-obs] WARNING empty archive skipped: {archive}",
                flush=True,
            )
    except Exception:
        remove_generated_tree(partial, extract_root)
        raise
    return {"archive": str(archive), "target": str(target), "status": extraction_status}


def read_extraction_summary(extract_root: Path) -> dict | None:
    path = extract_root / "extraction_summary.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != "passed" or int(payload.get("archive_count", 0)) <= 0:
        return None
    return payload


def extract_archives(
    source_root: Path,
    extract_root: Path,
    workers: int,
    resume: bool,
) -> dict:
    archives = discover_archives(source_root)
    if not archives:
        previous = read_extraction_summary(extract_root)
        if previous is not None:
            print(f"[context512-triplet-obs] reuse extraction summary: {extract_root}", flush=True)
            return previous
        print(f"[context512-triplet-obs] no archives found under {source_root}", flush=True)
        return {
            "status": "not_required",
            "archive_count": 0,
            "extract_root": str(source_root),
        }
    extract_root.mkdir(parents=True, exist_ok=True)
    worker_count = max(1, min(int(workers), len(archives)))
    print(
        f"[context512-triplet-obs] extract archives={len(archives)} workers={worker_count}",
        flush=True,
    )
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                extract_one_archive,
                archive,
                source_root,
                extract_root,
                resume,
            ): archive
            for archive in archives
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            done = len(results)
            if done % 10 == 0 or done == len(archives):
                print(
                    f"[context512-triplet-obs] extracted/reused archives={done}/{len(archives)}",
                    flush=True,
                )
    summary = {
        "status": "passed",
        "archive_count": len(archives),
        "extracted_count": sum(item["status"] == "extracted" for item in results),
        "reused_count": sum(item["status"].startswith("reused") for item in results),
        "empty_count": sum(item["status"] in {"empty", "reused_empty"} for item in results),
        "extract_root": str(extract_root),
        "archives": sorted(results, key=lambda item: item["archive"]),
    }
    write_json(extract_root / "extraction_summary.json", summary)
    return summary


def run_converter(args: argparse.Namespace, source_root: Path, gt_root: Path, output_root: Path) -> None:
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
        "--non-512-policy", "skip",
        "--progress-every", str(args.progress_every),
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


def run_balancer(
    args: argparse.Namespace,
    pool_root: Path,
    balanced_root: Path,
    package: Path,
) -> None:
    balancer = REPO_ROOT / "scripts" / "tools" / "build_balanced_three_image_dataset_v2.py"
    command = [
        sys.executable,
        str(balancer),
        "--input-root", str(pool_root),
        "--output-root", str(balanced_root),
        "--train-target-samples", str(args.train_target_samples),
        "--difficulty-ratios", str(args.difficulty_ratios),
        "--seed", str(args.balance_seed),
        "--copy-mode", args.copy_mode,
        "--progress-every", str(args.progress_every),
        "--package",
        "--package-path", str(package),
    ]
    donor_clean = str(args.empty_donor_clean_staging_root or "").strip()
    donor_aux = str(args.empty_donor_aux_staging_root or "").strip()
    if donor_clean or donor_aux:
        command.extend((
            "--empty-donor-clean-staging-root", donor_clean,
            "--empty-donor-aux-staging-root", donor_aux,
        ))
    if args.resume:
        command.append("--resume")
    print(
        "[context512-triplet-obs] balancer command:",
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
        output_root / "balance_preflight.json",
        output_root / "balance_report.json",
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
    balance = json.loads((output_root / "balance_report.json").read_text(encoding="utf-8"))
    if balance.get("status") != "passed":
        raise ValueError(f"difficulty balance did not pass: {balance}")
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
    extract_root = (
        Path(args.extract_root).expanduser().resolve()
        if args.extract_root
        else work_root / "extracted_images"
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
    pool_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else work_root / "output" / "context512_roi256_three_image_full"
    )
    balanced_root = (
        Path(args.balanced_output_root).expanduser().resolve()
        if args.balanced_output_root
        else work_root / "output" / (
            f"context512_roi256_three_image_balanced_{args.train_target_samples // 1000}k"
        )
    )
    package = (
        Path(args.package_path).expanduser().resolve()
        if args.package_path
        else work_root / "packages" / f"{balanced_root.name}.tar"
    )
    if pool_root in {source_root, gt_root} or balanced_root in {source_root, gt_root, pool_root}:
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

    extraction = extract_archives(
        source_root,
        extract_root,
        args.archive_workers,
        args.resume,
    )
    converter_source_root = (
        extract_root if int(extraction.get("archive_count", 0)) > 0 else source_root
    )
    if not args.skip_build:
        run_converter(args, converter_source_root, gt_root, pool_root)
        run_balancer(args, pool_root, balanced_root, package)
    validation = validate_completed_build(balanced_root, package)
    summary = {
        "status": "passed",
        "pipeline_version": PIPELINE_VERSION,
        "source_obs_root": args.source_obs_root,
        "gt_obs_root": args.gt_obs_root,
        "source_local_root": str(source_root),
        "converter_source_root": str(converter_source_root),
        "gt_local_root": str(gt_root),
        "gt_json_count": gt_json_count,
        "converted_pool_root": str(pool_root),
        "output_root": str(balanced_root),
        "package": str(package),
        "download_reports": download_reports,
        "extraction": extraction,
        "conversion_validation": validation,
    }
    write_json(work_root / "pipeline_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
