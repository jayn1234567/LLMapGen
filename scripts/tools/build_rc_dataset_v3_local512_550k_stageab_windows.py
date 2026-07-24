#!/usr/bin/env python3
"""Build a 550k local512v3 dataset with both Phase A and Phase B JSONL.

This wrapper assumes the retained ``staging_local512`` artifacts and the V3
difficulty audit already exist.  It finalizes a 550k balanced local512 dataset,
derives Phase B incoming hints from the full staging pool, and packages the
result as one tar archive.
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
from data_process.difficulty_profiles import DIFFICULTY_PROFILE_VERSION
from scripts.tools.build_rc_dataset_v2_from_obs import create_variant_tar
from scripts.tools.build_rc_dataset_v2_local512v2_windows import update_variant_metadata


TARGET_SAMPLES = 550_000
SOURCE_VARIANT = "local512"
DATASET_VARIANT = "local512v3_550k_stageab"
DIFFICULTY_RATIOS = "empty=0,very_easy=0.05,easy=0.20,medium=0.30,hard=0.30,very_hard=0.15"
INTERSECTION_RATIO = 0.30
TRACE_SPACING_PX = 50
TRACE_POINT_COUNT = 3


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-root",
        default=r"D:\data\fulldata_local512",
        help="Root that contains staging_local512 and difficulty_audit_local512_800k.",
    )
    parser.add_argument("--staging-root", default="")
    parser.add_argument("--audit-jsonl", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--package-path", default="")
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--difficulty-seed", type=int, default=20260723)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-finalize", action="store_true")
    parser.add_argument("--skip-phase-b", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--overwrite-phase-b", action="store_true")
    return parser.parse_args(argv)


def run(command: list) -> None:
    command = [str(item) for item in command]
    print("[local512v3-550k-stageab] command:", shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def expected_counts() -> dict[str, int]:
    return allocate_quotas(TARGET_SAMPLES, parse_ratio_spec(DIFFICULTY_RATIOS))


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def phase_a_complete(dataset_root: Path, variant: str) -> bool:
    info_path = dataset_root / "dataset_info.json"
    train_path = dataset_root / "phase_a" / "train.jsonl"
    if not info_path.is_file() or not train_path.is_file():
        return False
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    balance = info.get("balance") or {}
    record_counts = info.get("record_counts") or {}
    return all((
        info.get("dataset_variant") == variant,
        info.get("difficulty_rule_version") == DIFFICULTY_PROFILE_VERSION,
        balance.get("final_bucket_counts") == expected_counts(),
        abs(float(balance.get("actual_intersection_ratio", -1.0)) - INTERSECTION_RATIO) <= 1e-12,
        int(record_counts.get(f"{variant}:train", -1)) == TARGET_SAMPLES,
    ))


def phase_b_complete(dataset_root: Path) -> bool:
    for split in ("train", "eval", "test"):
        phase_a = dataset_root / "phase_a" / f"{split}.jsonl"
        phase_b = dataset_root / "phase_b" / f"{split}.jsonl"
        if not phase_a.is_file() or not phase_b.is_file():
            return False
        if count_jsonl(phase_a) != count_jsonl(phase_b):
            return False
    info_path = dataset_root / "dataset_info.json"
    if not info_path.is_file():
        return False
    info = json.loads(info_path.read_text(encoding="utf-8"))
    generation = info.get("phase_b_generation") or {}
    return all((
        int(generation.get("trace_point_count", -1)) == TRACE_POINT_COUNT,
        int(round(float(generation.get("trace_spacing_px", -1)))) == TRACE_SPACING_PX,
        generation.get("incoming_intersections") == "full_neighbor_polygon",
    ))


def completed_dataset(dataset_root: Path) -> bool:
    return phase_a_complete(dataset_root, DATASET_VARIANT) and phase_b_complete(dataset_root)


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    work_root = Path(args.work_root).expanduser().resolve()
    return {
        "work_root": work_root,
        "staging_root": (
            Path(args.staging_root).expanduser().resolve()
            if str(args.staging_root).strip()
            else work_root / "staging_local512"
        ),
        "audit_jsonl": (
            Path(args.audit_jsonl).expanduser().resolve()
            if str(args.audit_jsonl).strip()
            else work_root / "difficulty_audit_local512_800k" / "sample_metrics.jsonl"
        ),
        "output_root": (
            Path(args.output_root).expanduser().resolve()
            if str(args.output_root).strip()
            else work_root / "output_local512v3_550k_stageab"
        ),
        "package_path": (
            Path(args.package_path).expanduser().resolve()
            if str(args.package_path).strip()
            else work_root / "packages_v3" / f"{DATASET_VARIANT}.tar"
        ),
    }


def finalize_dataset(paths: dict[str, Path], args: argparse.Namespace) -> Path:
    output_root = paths["output_root"]
    target_root = output_root / DATASET_VARIANT
    source_root = output_root / SOURCE_VARIANT
    if args.resume and phase_a_complete(target_root, DATASET_VARIANT):
        print(f"[local512v3-550k-stageab] reuse completed phase_a: {target_root}", flush=True)
        return target_root
    if target_root.exists():
        if phase_a_complete(target_root, DATASET_VARIANT):
            return target_root
        raise ValueError(f"existing dataset is incomplete or incompatible: {target_root}")
    if source_root.exists():
        if not phase_a_complete(source_root, SOURCE_VARIANT):
            raise ValueError(f"existing source variant is incomplete or incompatible: {source_root}")
        source_root.rename(target_root)
        update_variant_metadata(target_root / "dataset_info.json", DATASET_VARIANT, SOURCE_VARIANT)
        update_variant_metadata(output_root / "build_summary.json", DATASET_VARIANT, SOURCE_VARIANT)
        return target_root
    output_root.mkdir(parents=True, exist_ok=True)
    if not paths["audit_jsonl"].is_file():
        raise FileNotFoundError(f"difficulty audit JSONL not found: {paths['audit_jsonl']}")
    run([
        sys.executable,
        "data_process/build_dataset_v2_staged.py",
        "finalize",
        "--staging-root", paths["staging_root"],
        "--output-root", output_root,
        "--views", "local",
        "--patch-size", 512,
        "--context-size", 512,
        "--train-target-samples", TARGET_SAMPLES,
        "--difficulty-ratios", DIFFICULTY_RATIOS,
        "--intersection-target-ratio", INTERSECTION_RATIO,
        "--difficulty-seed", args.difficulty_seed,
        "--duplicate-policy", "last",
        "--copy-mode", args.copy_mode,
        "--difficulty-override-jsonl", paths["audit_jsonl"],
        "--difficulty-rule-version", DIFFICULTY_PROFILE_VERSION,
    ])
    source_root.rename(target_root)
    update_variant_metadata(target_root / "dataset_info.json", DATASET_VARIANT, SOURCE_VARIANT)
    update_variant_metadata(output_root / "build_summary.json", DATASET_VARIANT, SOURCE_VARIANT)
    if not phase_a_complete(target_root, DATASET_VARIANT):
        raise ValueError(f"phase_a metadata check failed: {target_root}")
    return target_root


def derive_phase_b(dataset_root: Path, paths: dict[str, Path], args: argparse.Namespace) -> None:
    if args.resume and phase_b_complete(dataset_root):
        print(f"[local512v3-550k-stageab] reuse completed phase_b: {dataset_root / 'phase_b'}", flush=True)
        return
    command = [
        sys.executable,
        "scripts/tools/derive_stage_b_from_phase_a.py",
        "--dataset-root", dataset_root,
        "--staging-root", paths["staging_root"],
        "--staging-variant", SOURCE_VARIANT,
        "--trace-spacing-px", TRACE_SPACING_PX,
        "--trace-point-count", TRACE_POINT_COUNT,
        "--boundary-tol", 2.0,
        "--max-traces-per-side", 8,
        "--max-intersections-per-side", 8,
    ]
    if args.resume:
        command.append("--resume")
    if args.overwrite_phase_b:
        command.append("--overwrite")
    run(command)
    if not phase_b_complete(dataset_root):
        raise ValueError(f"phase_b metadata check failed: {dataset_root}")


def package_dataset(dataset_root: Path, paths: dict[str, Path], args: argparse.Namespace) -> None:
    create_variant_tar(dataset_root, paths["package_path"], args.resume)
    print(f"[local512v3-550k-stageab] package: {paths['package_path']}", flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    paths = resolve_paths(args)
    for key, value in paths.items():
        print(f"[local512v3-550k-stageab] {key}: {value}", flush=True)

    dataset_root = paths["output_root"] / DATASET_VARIANT
    if not args.skip_finalize:
        dataset_root = finalize_dataset(paths, args)
    elif not phase_a_complete(dataset_root, DATASET_VARIANT):
        raise FileNotFoundError(f"--skip-finalize was set but phase_a is incomplete: {dataset_root}")

    if not args.skip_phase_b:
        derive_phase_b(dataset_root, paths, args)
    elif not phase_b_complete(dataset_root):
        raise FileNotFoundError(f"--skip-phase-b was set but phase_b is incomplete: {dataset_root}")

    if not args.skip_package:
        package_dataset(dataset_root, paths, args)

    print(json.dumps({
        "status": "passed",
        "dataset_root": str(dataset_root),
        "package_path": str(paths["package_path"]) if not args.skip_package else "",
        "target_samples": TARGET_SAMPLES,
        "difficulty_ratios": DIFFICULTY_RATIOS,
        "target_quotas": expected_counts(),
        "phase_b_trace_spacing_px": TRACE_SPACING_PX,
        "phase_b_trace_point_count": TRACE_POINT_COUNT,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
