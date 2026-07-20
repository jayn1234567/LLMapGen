#!/usr/bin/env python3
"""Build 550k/100k true-512 Dataset V2 and intersection-conditioned variants."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.build_rc_dataset_v2_context512_windows import (
    build_compact_id_filter,
    subset_spec,
)
from scripts.tools.build_rc_dataset_v2_from_obs import create_variant_tar


PATCH_SIZE = 512
STANDARD_VARIANT = "local512"
PROMPT_VARIANT = "local512_intersection_prompt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", required=True, help="Output and staging root on a large NTFS disk.")
    parser.add_argument("--obsutil-path", required=True)
    parser.add_argument("--obsutil-config", default="")
    parser.add_argument("--source-obs-root", action="append", default=[])
    parser.add_argument("--archive-workers", type=int, default=16)
    parser.add_argument("--train-stride", type=int, default=256)
    parser.add_argument("--visualize-per-difficulty", type=int, default=50)
    parser.add_argument("--image-decode-mode", choices=["sampled", "all", "none"], default="sampled")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-stage", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    return parser.parse_args(argv)


def run(command):
    command = [str(item) for item in command]
    print("[local512-build] command:", shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def finalize_local512(
    staging_root: Path,
    output_root: Path,
    train_target: int,
    difficulty_ratios: str,
    intersection_ratio: float,
    resume: bool,
    candidate_jsonl: Path | None = None,
):
    command = [
        sys.executable,
        "data_process/build_dataset_v2_staged.py",
        "finalize",
        "--staging-root", staging_root,
        "--output-root", output_root,
        "--views", "local",
        "--patch-size", PATCH_SIZE,
        "--context-size", PATCH_SIZE,
        "--train-target-samples", train_target,
        "--difficulty-ratios", difficulty_ratios,
        "--intersection-target-ratio", intersection_ratio,
        "--difficulty-seed", 20260713,
        "--duplicate-policy", "last",
        "--copy-mode", "hardlink",
    ]
    if candidate_jsonl is not None:
        command.extend(["--train-candidate-jsonl", candidate_jsonl])
    if resume:
        command.append("--resume")
    run(command)


def read_ids(path: Path) -> set[str]:
    ids = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sample_id = str(json.loads(line).get("id", "")).strip()
            if not sample_id or sample_id in ids:
                raise ValueError(f"missing or duplicate id at {path}:{line_number}")
            ids.add(sample_id)
    return ids


def verify_subset(formal_root: Path, quick_root: Path, expected_quick: int, report_path: Path):
    formal_ids = read_ids(formal_root / "phase_a" / "train.jsonl")
    quick_ids = read_ids(quick_root / "phase_a" / "train.jsonl")
    unexpected = quick_ids - formal_ids
    report = {
        "status": "passed" if len(quick_ids) == expected_quick and not unexpected else "failed",
        "formal_train_samples": len(formal_ids),
        "quick_train_samples": len(quick_ids),
        "expected_quick_train_samples": expected_quick,
        "quick_is_strict_subset": quick_ids < formal_ids,
        "unexpected_id_count": len(unexpected),
        "unexpected_id_examples": sorted(unexpected)[:20],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "passed":
        raise ValueError(f"100k subset verification failed: {report}")


def verify_task_pairing(standard_root: Path, prompt_root: Path):
    split_counts = {}
    for split in ("train", "eval", "test"):
        standard_path = standard_root / "phase_a" / f"{split}.jsonl"
        prompt_path = prompt_root / "phase_a" / f"{split}.jsonl"
        count = 0
        with (
            standard_path.open("r", encoding="utf-8-sig") as standard,
            prompt_path.open("r", encoding="utf-8-sig") as prompt,
        ):
            standard_rows = (line for line in standard if line.strip())
            prompt_rows = (line for line in prompt if line.strip())
            while True:
                standard_line = next(standard_rows, None)
                prompt_line = next(prompt_rows, None)
                if standard_line is None or prompt_line is None:
                    if standard_line is not None or prompt_line is not None:
                        raise ValueError(f"task variants have different row counts in split={split}")
                    break
                standard_item = json.loads(standard_line)
                prompt_item = json.loads(prompt_line)
                if standard_item.get("id") != prompt_item.get("id"):
                    raise ValueError(f"task variant id mismatch in split={split} row={count + 1}")
                if standard_item.get("image") != prompt_item.get("image"):
                    raise ValueError(f"task variant image mismatch in split={split} row={count + 1}")
                count += 1
        split_counts[split] = count
    report = {"status": "passed", "exact_id_and_image_pairing": True, "split_counts": split_counts}
    (prompt_root / "task_pairing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def derive_prompt_variant(standard_root: Path, prompt_root: Path, resume: bool):
    command = [
        sys.executable,
        "scripts/tools/derive_intersection_prompt_dataset.py",
        "--input-root", standard_root,
        "--output-root", prompt_root,
        "--copy-mode", "hardlink",
    ]
    if resume:
        command.append("--resume")
    run(command)
    verify_task_pairing(standard_root, prompt_root)


def validate_variant(root: Path, variant: str, expected_total: int, args):
    run([
        sys.executable,
        "scripts/tools/validate_visualize_rc_dataset_v2.py",
        "--dataset-root", root,
        "--variant", variant,
        "--output-dir", root.parent / f"{variant}_validation",
        "--expected-train-samples", expected_total,
        "--visualize-per-difficulty", args.visualize_per_difficulty,
        "--image-decode-mode", args.image_decode_mode,
    ])


def main(argv=None):
    args = parse_args(argv)
    if args.train_stride <= 0 or PATCH_SIZE % args.train_stride:
        raise ValueError(f"--train-stride must be a positive divisor of {PATCH_SIZE}")
    work_root = Path(args.work_root).expanduser().resolve()
    raw_root = work_root / "raw_sources"
    staging_root = work_root / "staging_local512"
    output_550_root = work_root / "output_550k"
    output_100_root = work_root / "output_100k"
    package_root = work_root / "packages"
    filters_root = work_root / "filters"
    for path in (work_root, raw_root, staging_root, output_550_root, output_100_root, package_root, filters_root):
        path.mkdir(parents=True, exist_ok=True)

    print("============================================================", flush=True)
    print(f"[local512-build] work root:    {work_root}", flush=True)
    print(f"[local512-build] train stride: {args.train_stride}", flush=True)
    print(f"[local512-build] free disk:    {shutil.disk_usage(work_root).free / (1024 ** 3):.1f} GiB", flush=True)
    print("============================================================", flush=True)

    if not args.skip_stage:
        stage_command = [
            sys.executable,
            "scripts/tools/build_rc_dataset_v2_streaming_from_obs.py",
            "--work-root", work_root,
            "--raw-root", raw_root,
            "--staging-root", staging_root,
            "--output-root", work_root / "unused_stream_output",
            "--views", "local",
            "--patch-size", PATCH_SIZE,
            "--context-size", PATCH_SIZE,
            "--eval-test-stride", PATCH_SIZE,
            "--train-target-samples", 550000,
            "--train-stride", args.train_stride,
            "--archive-workers", args.archive_workers,
            "--obs-backend", "obsutil",
            "--obsutil-path", args.obsutil_path,
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

    requested_ratios = "empty=0,easy=0.30,medium=0.33,hard=0.27,very_hard=0.10"
    finalize_local512(staging_root, output_550_root, 550000, requested_ratios, 0.30, args.resume)
    standard_550 = output_550_root / STANDARD_VARIANT
    spec_550 = subset_spec(standard_550, 550000)
    formal_filter = filters_root / "local512_550k_train_ids.jsonl"
    if build_compact_id_filter(spec_550["train_jsonl"], formal_filter, args.resume) != 550000:
        raise ValueError("formal local512 train filter does not contain 550000 ids")

    finalize_local512(
        staging_root,
        output_100_root,
        100000,
        spec_550["ratios"],
        spec_550["intersection_ratio"],
        args.resume,
        formal_filter,
    )
    standard_100 = output_100_root / STANDARD_VARIANT
    subset_report = output_100_root / "subset_pairing_report.json"
    verify_subset(standard_550, standard_100, 100000, subset_report)
    shutil.copy2(subset_report, standard_100 / subset_report.name)

    prompt_550 = output_550_root / PROMPT_VARIANT
    prompt_100 = output_100_root / PROMPT_VARIANT
    derive_prompt_variant(standard_550, prompt_550, args.resume)
    derive_prompt_variant(standard_100, prompt_100, args.resume)

    if not args.skip_validation:
        validate_variant(standard_550, STANDARD_VARIANT, 550000, args)
        validate_variant(standard_100, STANDARD_VARIANT, 100000, args)
        validate_variant(prompt_550, PROMPT_VARIANT, 550000, args)
        validate_variant(prompt_100, PROMPT_VARIANT, 100000, args)

    if not args.skip_package:
        packages = {
            package_root / "local512_550k.tar": standard_550,
            package_root / "local512_100k.tar": standard_100,
            package_root / "local512_intersection_prompt_550k.tar": prompt_550,
            package_root / "local512_intersection_prompt_100k.tar": prompt_100,
        }
        for package, variant_root in packages.items():
            create_variant_tar(variant_root, package, args.resume)
            print(f"[local512-build] package: {package}", flush=True)

    print("[local512-build] done", flush=True)


if __name__ == "__main__":
    main()
