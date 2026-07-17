#!/usr/bin/env python3
"""Build paired 550k/100k context512_roi256 variants from Dataset V2 sources."""

from __future__ import annotations

import argparse
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

from data_process.build_dataset_v2 import DIFFICULTY_ORDER
from scripts.tools.build_rc_dataset_v2_from_obs import create_variant_tar


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", required=True, help="New context build root on a large NTFS disk.")
    parser.add_argument("--local-550-root", required=True, help="Completed 550k local256 variant root.")
    parser.add_argument("--local-100-root", required=True, help="Completed 100k local256 variant root.")
    parser.add_argument("--obsutil-path", required=True)
    parser.add_argument("--obsutil-config", default="")
    parser.add_argument("--source-obs-root", action="append", default=[])
    parser.add_argument("--archive-workers", type=int, default=16)
    parser.add_argument("--visualize-per-difficulty", type=int, default=50)
    parser.add_argument("--image-decode-mode", choices=["sampled", "all", "none"], default="sampled")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-stage", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    return parser.parse_args(argv)


def run(command):
    command = [str(item) for item in command]
    print("[context512-build] command:", shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_compact_id_filter(source_jsonl: Path, destination: Path, resume: bool) -> int:
    marker = destination.with_suffix(destination.suffix + ".meta.json")
    source_stat = source_jsonl.stat()
    expected_source = {
        "path": str(source_jsonl.resolve()),
        "size": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
    }
    if resume and destination.is_file() and marker.is_file():
        metadata = load_json(marker)
        if metadata.get("source") == expected_source and int(metadata.get("unique_ids", 0)) > 0:
            print(f"[context512-build] reuse id filter: {destination}", flush=True)
            return int(metadata["unique_ids"])

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    seen = set()
    with source_jsonl.open("r", encoding="utf-8-sig") as source, temporary.open("w", encoding="utf-8") as output:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            patch_id = str(item.get("id", "")).strip()
            if not patch_id:
                raise ValueError(f"missing id at {source_jsonl}:{line_number}")
            if patch_id in seen:
                raise ValueError(f"duplicate train id in {source_jsonl}: {patch_id}")
            seen.add(patch_id)
            output.write(json.dumps({"id": patch_id}, ensure_ascii=False, separators=(",", ":")) + "\n")
            if len(seen) % 100000 == 0:
                print(f"[context512-build] id filter {source_jsonl.name}: {len(seen)}", flush=True)
    temporary.replace(destination)
    marker.write_text(
        json.dumps({"source": expected_source, "unique_ids": len(seen)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(seen)


def verify_id_pairing(candidate_jsonl: Path, context_jsonl: Path, output_root: Path, expected_total: int) -> dict:
    remaining = set()
    with candidate_jsonl.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            patch_id = str(json.loads(line).get("id", "")).strip()
            if not patch_id or patch_id in remaining:
                raise ValueError(f"invalid or duplicate candidate id at {candidate_jsonl}:{line_number}")
            remaining.add(patch_id)

    actual_count = 0
    unexpected = []
    with context_jsonl.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            patch_id = str(json.loads(line).get("id", "")).strip()
            actual_count += 1
            if not patch_id or patch_id not in remaining:
                if len(unexpected) < 20:
                    unexpected.append({"line": line_number, "id": patch_id})
                continue
            remaining.remove(patch_id)

    exact_pairing = not remaining and not unexpected and actual_count == expected_total
    report = {
        "status": "passed" if exact_pairing else "failed",
        "candidate_jsonl": str(candidate_jsonl),
        "context_jsonl": str(context_jsonl),
        "expected_samples": expected_total,
        "actual_samples": actual_count,
        "missing_id_count": len(remaining),
        "missing_id_examples": sorted(remaining)[:20],
        "unexpected_or_duplicate_examples": unexpected,
        "exact_id_pairing": exact_pairing,
    }
    variant_root = output_root / "context512_roi256"
    for destination in (output_root / "pairing_report.json", variant_root / "pairing_report.json"):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not exact_pairing:
        raise ValueError(f"context/local train id pairing failed: {report}")
    print(f"[context512-build] exact train id pairing passed: {actual_count}", flush=True)
    return report


def subset_spec(local_root: Path, expected_total: int) -> dict:
    train_jsonl = local_root / "phase_a" / "train.jsonl"
    info_path = local_root / "dataset_info.json"
    if not train_jsonl.is_file() or not info_path.is_file():
        raise FileNotFoundError(f"completed Dataset V2 root is missing train JSONL or dataset_info.json: {local_root}")
    info = load_json(info_path)
    balance = info.get("balance") if isinstance(info.get("balance"), dict) else {}
    counts = balance.get("final_bucket_counts")
    if not isinstance(counts, dict):
        raise ValueError(f"dataset_info has no balance.final_bucket_counts: {info_path}")
    normalized_counts = {}
    for difficulty in DIFFICULTY_ORDER:
        value = counts.get(difficulty)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"invalid final difficulty count for {difficulty}: {value!r}")
        normalized_counts[difficulty] = value
    total = sum(normalized_counts.values())
    if total != expected_total:
        raise ValueError(f"train total in {info_path} is {total}, expected {expected_total}")
    ratios = ",".join(
        f"{difficulty}={normalized_counts[difficulty] / total:.12f}"
        for difficulty in DIFFICULTY_ORDER
    )
    intersection_ratio = float(balance.get("actual_intersection_ratio", -1.0))
    if abs(intersection_ratio - 0.30) > 1e-8:
        raise ValueError(f"expected 30% intersections in {info_path}, got {intersection_ratio}")
    return {
        "local_root": local_root,
        "train_jsonl": train_jsonl,
        "counts": normalized_counts,
        "ratios": ratios,
        "intersection_ratio": intersection_ratio,
        "total": total,
    }


def finalize_context(staging_root: Path, output_root: Path, id_filter: Path, spec: dict, resume: bool):
    command = [
        sys.executable,
        "data_process/build_dataset_v2_staged.py",
        "finalize",
        "--staging-root", staging_root,
        "--output-root", output_root,
        "--views", "context",
        "--train-target-samples", spec["total"],
        "--difficulty-ratios", spec["ratios"],
        "--intersection-target-ratio", spec["intersection_ratio"],
        "--difficulty-seed", 20260713,
        "--duplicate-policy", "last",
        "--copy-mode", "hardlink",
        "--train-candidate-jsonl", id_filter,
    ]
    if resume:
        command.append("--resume")
    run(command)


def validate_context(output_root: Path, expected_total: int, args):
    run([
        sys.executable,
        "scripts/tools/validate_visualize_rc_dataset_v2.py",
        "--dataset-root", output_root / "context512_roi256",
        "--variant", "context512_roi256",
        "--output-dir", output_root / "context512_roi256_validation",
        "--expected-train-samples", expected_total,
        "--visualize-per-difficulty", args.visualize_per_difficulty,
        "--image-decode-mode", args.image_decode_mode,
    ])


def copy_metadata(output_root: Path, package_root: Path, label: str):
    metadata_root = package_root / f"context512_roi256_{label}_metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    for relative in (
        "build_summary.json",
        "semantic_schema_report.json",
        "split_manifest.json",
        "balance_report.json",
        "pairing_report.json",
    ):
        source = output_root / relative
        if source.is_file():
            shutil.copy2(source, metadata_root / Path(relative).name)


def main(argv=None):
    args = parse_args(argv)
    work_root = Path(args.work_root).expanduser().resolve()
    local_550_root = Path(args.local_550_root).expanduser().resolve()
    local_100_root = Path(args.local_100_root).expanduser().resolve()
    raw_root = work_root / "raw_sources"
    staging_root = work_root / "staging_context512"
    output_550_root = work_root / "output_550k"
    output_100_root = work_root / "output_100k"
    package_root = work_root / "packages"
    filters_root = work_root / "filters"
    for path in (work_root, raw_root, staging_root, package_root, filters_root):
        path.mkdir(parents=True, exist_ok=True)

    if os.name == "nt" and work_root.drive.lower() != staging_root.drive.lower():
        raise ValueError("work and staging roots must be on the same NTFS volume for hard links")

    spec_550 = subset_spec(local_550_root, 550000)
    spec_100 = subset_spec(local_100_root, 100000)
    filter_550 = filters_root / "local256_550k_train_ids.jsonl"
    filter_100 = filters_root / "local256_100k_train_ids.jsonl"
    count_550 = build_compact_id_filter(spec_550["train_jsonl"], filter_550, args.resume)
    count_100 = build_compact_id_filter(spec_100["train_jsonl"], filter_100, args.resume)
    if count_550 != 550000 or count_100 != 100000:
        raise ValueError(f"unexpected candidate filter sizes: 550k={count_550}, 100k={count_100}")

    print("============================================================", flush=True)
    print(f"[context512-build] work root:   {work_root}", flush=True)
    print(f"[context512-build] local 550k:  {local_550_root}", flush=True)
    print(f"[context512-build] local 100k:  {local_100_root}", flush=True)
    print(f"[context512-build] free disk:   {shutil.disk_usage(work_root).free / (1024 ** 3):.1f} GiB", flush=True)
    print("============================================================", flush=True)

    if not args.skip_stage:
        stage_command = [
            sys.executable,
            "scripts/tools/build_rc_dataset_v2_streaming_from_obs.py",
            "--work-root", work_root,
            "--raw-root", raw_root,
            "--staging-root", staging_root,
            "--output-root", work_root / "unused_stream_output",
            "--views", "context",
            "--train-target-samples", 550000,
            "--train-stride", 128,
            "--archive-workers", args.archive_workers,
            "--obs-backend", "obsutil",
            "--obsutil-path", args.obsutil_path,
            "--train-candidate-jsonl", filter_550,
            "--skip-finalize",
            "--skip-upload",
        ]
        if args.obsutil_config:
            stage_command.extend(["--obsutil-config", args.obsutil_config])
        for source in args.source_obs_root:
            stage_command.extend(["--source-obs-root", source])
        if args.resume:
            stage_command.append("--resume")
        run(stage_command)

    finalize_context(staging_root, output_550_root, filter_550, spec_550, args.resume)
    finalize_context(staging_root, output_100_root, filter_100, spec_100, args.resume)
    verify_id_pairing(
        filter_550,
        output_550_root / "context512_roi256" / "phase_a" / "train.jsonl",
        output_550_root,
        550000,
    )
    verify_id_pairing(
        filter_100,
        output_100_root / "context512_roi256" / "phase_a" / "train.jsonl",
        output_100_root,
        100000,
    )

    if not args.skip_validation:
        validate_context(output_550_root, 550000, args)
        validate_context(output_100_root, 100000, args)

    if not args.skip_package:
        package_550 = package_root / "context512_roi256_550k.tar"
        package_100 = package_root / "context512_roi256_100k.tar"
        create_variant_tar(output_550_root / "context512_roi256", package_550, args.resume)
        create_variant_tar(output_100_root / "context512_roi256", package_100, args.resume)
        copy_metadata(output_550_root, package_root, "550k")
        copy_metadata(output_100_root, package_root, "100k")
        print(f"[context512-build] package 550k: {package_550}", flush=True)
        print(f"[context512-build] package 100k: {package_100}", flush=True)

    print("[context512-build] done", flush=True)


if __name__ == "__main__":
    main()
