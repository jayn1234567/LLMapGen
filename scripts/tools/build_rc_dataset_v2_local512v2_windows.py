#!/usr/bin/env python3
"""Build balanced local512v2 550k/200k datasets from reusable local512 stages."""

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

from data_process.build_dataset_v2 import allocate_quotas, parse_ratio_spec
from data_process.state_update_dataset_common import SEMANTIC_SCHEMA_VERSION
from scripts.tools.build_rc_dataset_v2_context512_windows import (
    build_compact_id_filter,
    subset_spec,
)
from scripts.tools.build_rc_dataset_v2_from_obs import create_variant_tar
from scripts.tools.build_rc_dataset_v2_local512_windows import (
    FORMAL_TRAIN_TARGET,
    finalize_local512,
    sample_count_label,
    verify_subset,
    verify_task_pairing,
)
from scripts.tools.derive_intersection_prompt_dataset import derive_dataset


SOURCE_VARIANT = "local512"
STANDARD_VARIANT = "local512v2"
PROMPT_VARIANT = "local512v2_intersection_prompt"
LOCAL256_SOURCE_VARIANT = "local256"
LOCAL256_STANDARD_VARIANT = "local256v2"
DEFAULT_QUICK_TRAIN_TARGET = 200000
DIFFICULTY_RATIOS = "empty=0,easy=0.20,medium=0.30,hard=0.30,very_hard=0.20"
INTERSECTION_RATIO = 0.30


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", required=True, help="Reuse the previous local512 work root.")
    parser.add_argument("--obsutil-path", required=True)
    parser.add_argument("--obsutil-config", default="")
    parser.add_argument("--source-obs-root", action="append", default=[])
    parser.add_argument("--archive-workers", type=int, default=16)
    parser.add_argument("--train-stride", type=int, default=256)
    parser.add_argument("--local256-train-stride", type=int, default=128)
    parser.add_argument("--quick-train-target-samples", type=int, default=DEFAULT_QUICK_TRAIN_TARGET)
    parser.add_argument("--visualize-per-difficulty", type=int, default=50)
    parser.add_argument("--image-decode-mode", choices=["sampled", "all", "none"], default="sampled")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-stage", action="store_true")
    parser.add_argument("--skip-local256", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    return parser.parse_args(argv)


def run(command):
    command = [str(item) for item in command]
    print("[local512v2-build] command:", shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def expected_difficulty_counts(train_target: int) -> dict[str, int]:
    return allocate_quotas(train_target, parse_ratio_spec(DIFFICULTY_RATIOS))


def relabel_count_keys(counts: dict, source: str, target: str) -> dict:
    prefix = f"{source}:"
    return {
        f"{target}:{key[len(prefix):]}" if str(key).startswith(prefix) else key: value
        for key, value in counts.items()
    }


def update_variant_metadata(path: Path, dataset_variant: str, source_variant: str = SOURCE_VARIANT) -> None:
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dataset_variant"] = dataset_variant
    payload["active_variant"] = dataset_variant
    payload["base_view_mode"] = source_variant
    if isinstance(payload.get("variants"), list):
        payload["variants"] = [
            dataset_variant if item == source_variant else item
            for item in payload["variants"]
        ]
    if isinstance(payload.get("record_counts"), dict):
        payload["record_counts"] = relabel_count_keys(
            payload["record_counts"],
            source_variant,
            dataset_variant,
        )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def completed_named_variant(root: Path, train_target: int, dataset_variant: str = STANDARD_VARIANT) -> bool:
    info_path = root / "dataset_info.json"
    train_path = root / "phase_a" / "train.jsonl"
    if not info_path.is_file() or not train_path.is_file():
        return False
    info = json.loads(info_path.read_text(encoding="utf-8"))
    balance = info.get("balance") or {}
    return (
        info.get("semantic_schema_version") == SEMANTIC_SCHEMA_VERSION
        and info.get("dataset_variant") == dataset_variant
        and balance.get("final_bucket_counts") == expected_difficulty_counts(train_target)
        and abs(float(balance.get("actual_intersection_ratio", -1)) - INTERSECTION_RATIO) <= 1e-12
    )


def finalize_named_variant(
    staging_root: Path,
    output_root: Path,
    train_target: int,
    resume: bool,
    candidate_jsonl: Path | None = None,
    source_variant: str = SOURCE_VARIANT,
    dataset_variant: str = STANDARD_VARIANT,
    patch_size: int = 512,
) -> Path:
    target_root = output_root / dataset_variant
    source_root = output_root / source_variant
    if resume and completed_named_variant(target_root, train_target, dataset_variant):
        print(f"[local512v2-build] reuse completed dataset: {target_root}", flush=True)
        return target_root
    if target_root.exists():
        raise ValueError(
            f"existing generated output is incomplete or incompatible: {target_root}. "
            "Use a new --work-root or remove only this generated output after inspection."
        )
    if source_root.exists():
        raise ValueError(f"unfinished source variant already exists: {source_root}")

    if patch_size == 512 and source_variant == SOURCE_VARIANT:
        finalize_local512(
            staging_root,
            output_root,
            train_target,
            DIFFICULTY_RATIOS,
            INTERSECTION_RATIO,
            resume,
            candidate_jsonl,
        )
    else:
        command = [
            sys.executable,
            "data_process/build_dataset_v2_staged.py",
            "finalize",
            "--staging-root", staging_root,
            "--output-root", output_root,
            "--views", "local",
            "--patch-size", patch_size,
            "--context-size", patch_size,
            "--train-target-samples", train_target,
            "--difficulty-ratios", DIFFICULTY_RATIOS,
            "--intersection-target-ratio", INTERSECTION_RATIO,
            "--difficulty-seed", 20260713,
            "--duplicate-policy", "last",
            "--copy-mode", "hardlink",
        ]
        if candidate_jsonl is not None:
            command.extend(["--train-candidate-jsonl", candidate_jsonl])
        if resume:
            command.append("--resume")
        run(command)
    source_root.rename(target_root)
    update_variant_metadata(target_root / "dataset_info.json", dataset_variant, source_variant)
    update_variant_metadata(output_root / "build_summary.json", dataset_variant, source_variant)
    if not completed_named_variant(target_root, train_target, dataset_variant):
        raise ValueError(f"relabelled dataset failed metadata validation: {target_root}")
    return target_root


def derive_prompt_variant(standard_root: Path, prompt_root: Path, resume: bool) -> None:
    derive_dataset(
        standard_root,
        prompt_root,
        argparse.Namespace(copy_mode="hardlink", resume=resume, progress_every=10000),
    )
    verify_task_pairing(standard_root, prompt_root)


def validate_variant(root: Path, variant: str, expected_total: int, args) -> None:
    run([
        sys.executable,
        "scripts/tools/validate_visualize_rc_dataset_v2.py",
        "--dataset-root", root,
        "--variant", variant,
        "--output-dir", root.parent / f"{variant}_validation",
        "--expected-train-samples", expected_total,
        "--difficulty-ratios", DIFFICULTY_RATIOS,
        "--expected-intersection-ratio", INTERSECTION_RATIO,
        "--visualize-per-difficulty", args.visualize_per_difficulty,
        "--image-decode-mode", args.image_decode_mode,
    ])


def main(argv=None):
    args = parse_args(argv)
    if args.train_stride <= 0 or 512 % args.train_stride:
        raise ValueError("--train-stride must be a positive divisor of 512")
    if args.local256_train_stride <= 0 or 256 % args.local256_train_stride:
        raise ValueError("--local256-train-stride must be a positive divisor of 256")
    if not 0 < args.quick_train_target_samples < FORMAL_TRAIN_TARGET:
        raise ValueError(
            f"--quick-train-target-samples must be positive and smaller than {FORMAL_TRAIN_TARGET}"
        )

    quick_target = args.quick_train_target_samples
    quick_label = sample_count_label(quick_target)
    work_root = Path(args.work_root).expanduser().resolve()
    raw_root = work_root / "raw_sources"
    staging_root = work_root / "staging_local512"
    local256_staging_root = work_root / "staging_local256"
    output_550_root = work_root / "output_local512v2_550k"
    output_quick_root = work_root / f"output_local512v2_{quick_label}"
    local256_output_550_root = work_root / "output_local256v2_550k"
    local256_output_quick_root = work_root / f"output_local256v2_{quick_label}"
    package_root = work_root / "packages"
    filters_root = work_root / "filters"
    for path in (
        work_root,
        raw_root,
        staging_root,
        local256_staging_root,
        output_550_root,
        output_quick_root,
        local256_output_550_root,
        local256_output_quick_root,
        package_root,
        filters_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    print("============================================================", flush=True)
    print(f"[local512v2-build] work root:       {work_root}", flush=True)
    print(f"[local512v2-build] reusable stage:  {staging_root}", flush=True)
    if not args.skip_local256:
        print(f"[local512v2-build] local256 stage:  {local256_staging_root}", flush=True)
    print(f"[local512v2-build] difficulty:      {DIFFICULTY_RATIOS}", flush=True)
    print(f"[local512v2-build] intersection:    {INTERSECTION_RATIO:.2f}", flush=True)
    print(f"[local512v2-build] formal/quick:    {FORMAL_TRAIN_TARGET}/{quick_target}", flush=True)
    print(f"[local512v2-build] free disk:       {shutil.disk_usage(work_root).free / (1024 ** 3):.1f} GiB", flush=True)
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
            "--patch-size", 512,
            "--context-size", 512,
            "--eval-test-stride", 512,
            "--train-target-samples", FORMAL_TRAIN_TARGET,
            "--train-stride", args.train_stride,
            "--archive-workers", args.archive_workers,
            "--obs-backend", "obsutil",
            "--obsutil-path", args.obsutil_path,
            "--skip-finalize",
            "--skip-upload",
        ]
        if not args.skip_local256:
            stage_command.extend([
                "--secondary-local256-staging-root", local256_staging_root,
                "--secondary-local256-train-stride", args.local256_train_stride,
            ])
        if args.obsutil_config:
            stage_command.extend(["--obsutil-config", args.obsutil_config])
        for source in args.source_obs_root:
            stage_command.extend(["--source-obs-root", source])
        if args.resume:
            stage_command.append("--resume")
        run(stage_command)

    standard_550 = finalize_named_variant(
        staging_root,
        output_550_root,
        FORMAL_TRAIN_TARGET,
        args.resume,
    )
    spec_550 = subset_spec(standard_550, FORMAL_TRAIN_TARGET)
    formal_filter = filters_root / "local512v2_550k_train_ids.jsonl"
    if build_compact_id_filter(spec_550["train_jsonl"], formal_filter, args.resume) != FORMAL_TRAIN_TARGET:
        raise ValueError(f"local512v2 formal filter does not contain {FORMAL_TRAIN_TARGET} ids")

    standard_quick = finalize_named_variant(
        staging_root,
        output_quick_root,
        quick_target,
        args.resume,
        formal_filter,
    )
    subset_report = output_quick_root / "subset_pairing_report.json"
    verify_subset(standard_550, standard_quick, quick_target, subset_report)
    shutil.copy2(subset_report, standard_quick / subset_report.name)

    local256_550 = None
    local256_quick = None
    if not args.skip_local256:
        local256_550 = finalize_named_variant(
            local256_staging_root,
            local256_output_550_root,
            FORMAL_TRAIN_TARGET,
            args.resume,
            source_variant=LOCAL256_SOURCE_VARIANT,
            dataset_variant=LOCAL256_STANDARD_VARIANT,
            patch_size=256,
        )
        local256_filter = filters_root / "local256v2_550k_train_ids.jsonl"
        local256_spec = subset_spec(local256_550, FORMAL_TRAIN_TARGET)
        if build_compact_id_filter(
            local256_spec["train_jsonl"],
            local256_filter,
            args.resume,
        ) != FORMAL_TRAIN_TARGET:
            raise ValueError(f"local256v2 formal filter does not contain {FORMAL_TRAIN_TARGET} ids")
        local256_quick = finalize_named_variant(
            local256_staging_root,
            local256_output_quick_root,
            quick_target,
            args.resume,
            local256_filter,
            source_variant=LOCAL256_SOURCE_VARIANT,
            dataset_variant=LOCAL256_STANDARD_VARIANT,
            patch_size=256,
        )
        local256_subset_report = local256_output_quick_root / "subset_pairing_report.json"
        verify_subset(local256_550, local256_quick, quick_target, local256_subset_report)
        shutil.copy2(local256_subset_report, local256_quick / local256_subset_report.name)

    prompt_550 = output_550_root / PROMPT_VARIANT
    prompt_quick = output_quick_root / PROMPT_VARIANT
    derive_prompt_variant(standard_550, prompt_550, args.resume)
    derive_prompt_variant(standard_quick, prompt_quick, args.resume)

    if not args.skip_validation:
        validate_variant(standard_550, STANDARD_VARIANT, FORMAL_TRAIN_TARGET, args)
        validate_variant(standard_quick, STANDARD_VARIANT, quick_target, args)
        validate_variant(prompt_550, PROMPT_VARIANT, FORMAL_TRAIN_TARGET, args)
        validate_variant(prompt_quick, PROMPT_VARIANT, quick_target, args)
        if local256_550 is not None and local256_quick is not None:
            validate_variant(local256_550, LOCAL256_STANDARD_VARIANT, FORMAL_TRAIN_TARGET, args)
            validate_variant(local256_quick, LOCAL256_STANDARD_VARIANT, quick_target, args)

    if not args.skip_package:
        packages = {
            package_root / "local512v2_550k.tar": standard_550,
            package_root / f"local512v2_{quick_label}.tar": standard_quick,
            package_root / "local512v2_intersection_prompt_550k.tar": prompt_550,
            package_root / f"local512v2_intersection_prompt_{quick_label}.tar": prompt_quick,
        }
        if local256_550 is not None and local256_quick is not None:
            packages.update({
                package_root / "local256v2_550k.tar": local256_550,
                package_root / f"local256v2_{quick_label}.tar": local256_quick,
            })
        for package, variant_root in packages.items():
            create_variant_tar(variant_root, package, args.resume)
            print(f"[local512v2-build] package: {package}", flush=True)

    print("[local512v2-build] done", flush=True)


if __name__ == "__main__":
    main()
