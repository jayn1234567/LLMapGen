#!/usr/bin/env python3
"""Build four balanced Dataset V3 variants from retained staging artifacts."""

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
from data_process.build_dataset_v2_staged import (
    build_sample_owners,
    collect_owned_raw_sample_splits,
    discover_stage_roots,
)
from data_process.difficulty_profiles import DIFFICULTY_PROFILE_VERSION
from data_process.state_update_dataset_common import SEMANTIC_SCHEMA_VERSION
from scripts.tools.build_rc_dataset_v2_context512_windows import (
    build_compact_id_filter,
)
from scripts.tools.build_rc_dataset_v2_from_obs import create_variant_tar
from scripts.tools.build_rc_dataset_v2_local512_windows import verify_subset, verify_task_pairing
from scripts.tools.build_rc_dataset_v2_local512v2_windows import update_variant_metadata
from scripts.tools.derive_intersection_prompt_dataset import derive_dataset
from scripts.tools.verify_dataset_v3_eval_sources import verify_dataset_roots


DIFFICULTY_RATIOS = (
    "empty=0,very_easy=0.05,easy=0.20,medium=0.30,hard=0.30,very_hard=0.15"
)
INTERSECTION_RATIO = 0.30
LOCAL_TARGET = 550000
CONTEXT_TARGETS = (550000, 200000)
LOCAL_SOURCE_VARIANT = "local512"
LOCAL_VARIANT = "local512v3"
PROMPT_VARIANT = "local512v3_intersection_prompt"
CONTEXT_SOURCE_VARIANT = "context512_roi256"
CONTEXT_VARIANT = "context512_roi256v3"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--local512-staging-root", default="")
    parser.add_argument("--context-staging-root", default="")
    parser.add_argument("--visualize-per-difficulty", type=int, default=100)
    parser.add_argument("--image-decode-mode", choices=["sampled", "all", "none"], default="sampled")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-difficulty-audit", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    return parser.parse_args(argv)


def run(command: list) -> None:
    command = [str(item) for item in command]
    print("[dataset-v3-build] command:", shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def expected_counts(total: int) -> dict[str, int]:
    return allocate_quotas(total, parse_ratio_spec(DIFFICULTY_RATIOS))


def completed_variant(root: Path, total: int, variant: str) -> bool:
    info_path = root / "dataset_info.json"
    train_path = root / "phase_a" / "train.jsonl"
    if not info_path.is_file() or not train_path.is_file():
        return False
    info = json.loads(info_path.read_text(encoding="utf-8"))
    balance = info.get("balance") or {}
    return all((
        info.get("semantic_schema_version") == SEMANTIC_SCHEMA_VERSION,
        info.get("dataset_variant") == variant,
        info.get("difficulty_rule_version") == DIFFICULTY_PROFILE_VERSION,
        balance.get("final_bucket_counts") == expected_counts(total),
        abs(float(balance.get("actual_intersection_ratio", -1.0)) - INTERSECTION_RATIO) <= 1e-12,
    ))


def ensure_stage_variant(staging_root: Path, variant: str, patch_size: int) -> None:
    roots = discover_stage_roots(staging_root)
    for stage_root in roots:
        marker = json.loads((stage_root / "stage_complete.json").read_text(encoding="utf-8"))
        if variant not in marker.get("variants", []):
            raise ValueError(f"stage does not contain {variant}: {stage_root}")
        if int(marker.get("target_patch_size", -1)) != patch_size:
            raise ValueError(f"stage patch size mismatch for {stage_root}: {marker.get('target_patch_size')}")


def audit_complete(output_dir: Path, profile: str, variant: str) -> bool:
    summary_path = output_dir / "summary.json"
    sidecar_path = output_dir / "sample_metrics.jsonl"
    if not summary_path.is_file() or not sidecar_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return all((
        summary.get("status") == "ok",
        summary.get("variant") == variant,
        (summary.get("difficulty_profile") or {}).get("name") == profile,
        int(summary.get("scanned_nonempty", 0)) > 0,
    ))


def verify_audit_capacity(output_dir: Path, total: int) -> dict[str, int]:
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    available = summary.get("difficulty_counts") or {}
    required = expected_counts(total)
    shortages = {
        bucket: required_count - int(available.get(bucket, 0))
        for bucket, required_count in required.items()
        if required_count > int(available.get(bucket, 0))
    }
    if shortages:
        raise ValueError(
            f"difficulty candidate capacity is insufficient for {total} samples: "
            f"shortages={shortages}, available={available}, required={required}"
        )
    print(
        f"[dataset-v3-build] difficulty capacity passed: total={total} "
        f"available={available} required={required}",
        flush=True,
    )
    return {str(key): int(value) for key, value in available.items()}


def run_difficulty_audit(
    staging_root: Path,
    output_dir: Path,
    variant: str,
    patch_size: int,
    profile: str,
    visualize_per_difficulty: int,
    resume: bool,
) -> Path:
    if resume and audit_complete(output_dir, profile, variant):
        print(f"[dataset-v3-build] reuse difficulty sidecar: {output_dir}", flush=True)
        return output_dir / "sample_metrics.jsonl"
    if output_dir.exists():
        raise ValueError(
            f"difficulty audit output is incomplete or stale: {output_dir}. "
            "Remove only this generated audit directory or use a new --work-root."
        )
    run([
        sys.executable,
        "scripts/tools/audit_staged_512_difficulty.py",
        "--staging-root", staging_root,
        "--output-dir", output_dir,
        "--variant", variant,
        "--patch-size", patch_size,
        "--profile", profile,
        "--visualize-per-difficulty", visualize_per_difficulty,
        "--progress-every", 10000,
    ])
    if not audit_complete(output_dir, profile, variant):
        raise ValueError(f"difficulty audit did not complete: {output_dir}")
    return output_dir / "sample_metrics.jsonl"


def stage_split_sources(staging_root: Path) -> dict[str, list[str]]:
    roots = discover_stage_roots(staging_root)
    owners, _ = build_sample_owners(roots, "last")
    return collect_owned_raw_sample_splits(roots, owners)


def verify_stage_eval_sources(local_staging: Path, context_staging: Path, output_path: Path) -> dict:
    local = stage_split_sources(local_staging)
    context = stage_split_sources(context_staging)
    report = {"status": "passed", "splits": {}}
    for split in ("eval", "test"):
        local_ids = set(local[split])
        context_ids = set(context[split])
        missing = local_ids - context_ids
        extra = context_ids - local_ids
        exact = not missing and not extra
        if not exact:
            report["status"] = "failed"
        report["splits"][split] = {
            "exact_match": exact,
            "local512_raw_images": len(local_ids),
            "context512_roi256_raw_images": len(context_ids),
            "missing_in_context_count": len(missing),
            "extra_in_context_count": len(extra),
            "missing_examples": sorted(missing)[:20],
            "extra_examples": sorted(extra)[:20],
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "passed":
        raise ValueError(f"local/context eval source mismatch: {output_path}")
    print(f"[dataset-v3-build] stage eval/test source check passed: {output_path}", flush=True)
    return report


def finalize_variant(
    staging_root: Path,
    run_root: Path,
    source_variant: str,
    dataset_variant: str,
    views: str,
    patch_size: int,
    context_size: int,
    total: int,
    difficulty_sidecar: Path,
    resume: bool,
    candidate_jsonl: Path | None = None,
) -> Path:
    target_root = run_root / dataset_variant
    source_root = run_root / source_variant
    if resume and completed_variant(target_root, total, dataset_variant):
        print(f"[dataset-v3-build] reuse completed dataset: {target_root}", flush=True)
        return target_root
    if target_root.exists() or source_root.exists():
        raise ValueError(f"generated output is incomplete or incompatible: {target_root} / {source_root}")
    command = [
        sys.executable,
        "data_process/build_dataset_v2_staged.py",
        "finalize",
        "--staging-root", staging_root,
        "--output-root", run_root,
        "--views", views,
        "--patch-size", patch_size,
        "--context-size", context_size,
        "--train-target-samples", total,
        "--difficulty-ratios", DIFFICULTY_RATIOS,
        "--intersection-target-ratio", INTERSECTION_RATIO,
        "--difficulty-seed", 20260723,
        "--duplicate-policy", "last",
        "--copy-mode", "hardlink",
        "--difficulty-override-jsonl", difficulty_sidecar,
        "--difficulty-rule-version", DIFFICULTY_PROFILE_VERSION,
    ]
    if candidate_jsonl is not None:
        command.extend(["--train-candidate-jsonl", candidate_jsonl])
    if resume:
        command.append("--resume")
    run(command)
    source_root.rename(target_root)
    update_variant_metadata(target_root / "dataset_info.json", dataset_variant, source_variant)
    update_variant_metadata(run_root / "build_summary.json", dataset_variant, source_variant)
    if not completed_variant(target_root, total, dataset_variant):
        raise ValueError(f"finalized variant failed metadata checks: {target_root}")
    return target_root


def derive_prompt_variant(standard_root: Path, prompt_root: Path, resume: bool) -> Path:
    derive_dataset(
        standard_root,
        prompt_root,
        argparse.Namespace(copy_mode="hardlink", resume=resume, progress_every=10000),
    )
    verify_task_pairing(standard_root, prompt_root)
    return prompt_root


def validate_variant(root: Path, variant: str, total: int, args: argparse.Namespace) -> None:
    run([
        sys.executable,
        "scripts/tools/validate_visualize_rc_dataset_v2.py",
        "--dataset-root", root,
        "--variant", variant,
        "--output-dir", root.parent / f"{variant}_validation",
        "--expected-train-samples", total,
        "--difficulty-ratios", DIFFICULTY_RATIOS,
        "--expected-intersection-ratio", INTERSECTION_RATIO,
        "--visualize-per-difficulty", 0,
        "--image-decode-mode", args.image_decode_mode,
    ])


def package_variants(packages: dict[Path, Path], resume: bool) -> None:
    for package_path, variant_root in packages.items():
        create_variant_tar(variant_root, package_path, resume)
        print(f"[dataset-v3-build] package: {package_path}", flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    work_root = Path(args.work_root).expanduser().resolve()
    local_staging = (
        Path(args.local512_staging_root).expanduser().resolve()
        if args.local512_staging_root
        else work_root / "staging_local512"
    )
    context_staging = (
        Path(args.context_staging_root).expanduser().resolve()
        if args.context_staging_root
        else work_root / "staging_context512"
    )
    output_root = work_root / "output_dataset_v3"
    audit_root = work_root / "difficulty_audit_v3"
    filters_root = work_root / "filters_v3"
    packages_root = work_root / "packages_v3"
    for path in (output_root, audit_root, filters_root, packages_root):
        path.mkdir(parents=True, exist_ok=True)

    ensure_stage_variant(local_staging, LOCAL_SOURCE_VARIANT, 512)
    ensure_stage_variant(context_staging, CONTEXT_SOURCE_VARIANT, 256)
    verify_stage_eval_sources(
        local_staging,
        context_staging,
        output_root / "stage_eval_source_consistency.json",
    )

    local_audit = audit_root / "local512"
    context_audit = audit_root / "context512_roi256"
    if args.skip_difficulty_audit:
        if not audit_complete(local_audit, "local512_profile_a", LOCAL_SOURCE_VARIANT):
            raise FileNotFoundError(f"completed local512 difficulty audit not found: {local_audit}")
        if not audit_complete(context_audit, "roi256_profile_a", CONTEXT_SOURCE_VARIANT):
            raise FileNotFoundError(f"completed context difficulty audit not found: {context_audit}")
        local_sidecar = local_audit / "sample_metrics.jsonl"
        context_sidecar = context_audit / "sample_metrics.jsonl"
    else:
        local_sidecar = run_difficulty_audit(
            local_staging,
            local_audit,
            LOCAL_SOURCE_VARIANT,
            512,
            "local512_profile_a",
            args.visualize_per_difficulty,
            args.resume,
        )
        context_sidecar = run_difficulty_audit(
            context_staging,
            context_audit,
            CONTEXT_SOURCE_VARIANT,
            256,
            "roi256_profile_a",
            args.visualize_per_difficulty,
            args.resume,
        )

    verify_audit_capacity(local_audit, LOCAL_TARGET)
    verify_audit_capacity(context_audit, CONTEXT_TARGETS[0])

    local_run = output_root / "local512v3_550k"
    context_550_run = output_root / "context512_roi256v3_550k"
    context_200_run = output_root / "context512_roi256v3_200k"
    for path in (local_run, context_550_run, context_200_run):
        path.mkdir(parents=True, exist_ok=True)

    local_root = finalize_variant(
        local_staging,
        local_run,
        LOCAL_SOURCE_VARIANT,
        LOCAL_VARIANT,
        "local",
        512,
        512,
        LOCAL_TARGET,
        local_sidecar,
        args.resume,
    )
    prompt_root = derive_prompt_variant(local_root, local_run / PROMPT_VARIANT, args.resume)

    context_550_root = finalize_variant(
        context_staging,
        context_550_run,
        CONTEXT_SOURCE_VARIANT,
        CONTEXT_VARIANT,
        "context",
        256,
        512,
        CONTEXT_TARGETS[0],
        context_sidecar,
        args.resume,
    )
    context_filter = filters_root / "context512_roi256v3_550k_train_ids.jsonl"
    if build_compact_id_filter(
        context_550_root / "phase_a" / "train.jsonl",
        context_filter,
        args.resume,
    ) != CONTEXT_TARGETS[0]:
        raise ValueError(f"context 550k filter has an unexpected size: {context_filter}")
    context_200_root = finalize_variant(
        context_staging,
        context_200_run,
        CONTEXT_SOURCE_VARIANT,
        CONTEXT_VARIANT,
        "context",
        256,
        512,
        CONTEXT_TARGETS[1],
        context_sidecar,
        args.resume,
        context_filter,
    )
    context_subset_report = context_200_run / "subset_pairing_report.json"
    verify_subset(
        context_550_root,
        context_200_root,
        CONTEXT_TARGETS[1],
        context_subset_report,
    )
    shutil.copy2(context_subset_report, context_200_root / context_subset_report.name)

    split_report = verify_dataset_roots([
        local_root,
        prompt_root,
        context_550_root,
        context_200_root,
    ])
    split_report_path = output_root / "eval_source_consistency.json"
    split_report_path.write_text(json.dumps(split_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if split_report["status"] != "passed":
        raise ValueError(f"final dataset eval/test source mismatch: {split_report_path}")

    if not args.skip_validation:
        validate_variant(local_root, LOCAL_VARIANT, LOCAL_TARGET, args)
        validate_variant(prompt_root, PROMPT_VARIANT, LOCAL_TARGET, args)
        validate_variant(context_550_root, CONTEXT_VARIANT, CONTEXT_TARGETS[0], args)
        validate_variant(context_200_root, CONTEXT_VARIANT, CONTEXT_TARGETS[1], args)

    if not args.skip_package:
        package_variants({
            packages_root / "local512v3_550k.tar": local_root,
            packages_root / "local512v3_intersection_prompt_550k.tar": prompt_root,
            packages_root / "context512_roi256v3_550k.tar": context_550_root,
            packages_root / "context512_roi256v3_200k.tar": context_200_root,
        }, args.resume)

    print("[dataset-v3-build] all four datasets completed", flush=True)
    print(f"[dataset-v3-build] split consistency: {split_report_path}", flush=True)


if __name__ == "__main__":
    main()
