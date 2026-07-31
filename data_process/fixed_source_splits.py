#!/usr/bin/env python3
"""Load and validate fixed raw-source train/eval/test assignments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


FORMAT_VERSION = "rc_fixed_source_split_v1"
SPLIT_POLICY = "explicit_fixed_eval_test_train_complement"
SPLITS = ("train", "eval", "test")


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def assignment_manifest_id(eval_ids: Iterable[str], test_ids: Iterable[str]) -> str:
    canonical = json.dumps(
        {
            "eval": sorted({str(item).strip() for item in eval_ids if str(item).strip()}),
            "test": sorted({str(item).strip() for item in test_ids if str(item).strip()}),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ids_from_payload(payload: dict[str, Any], split: str) -> list[str]:
    by_split = payload.get("raw_sample_ids_by_split") or {}
    values = by_split.get(split)
    if values is None:
        values = payload.get(f"{split}_ids", [])
    if not isinstance(values, list):
        raise ValueError(f"fixed split manifest {split} ids must be a list")
    ids = [str(item).strip() for item in values]
    if any(not item for item in ids):
        raise ValueError(f"fixed split manifest {split} ids contain an empty value")
    if len(ids) != len(set(ids)):
        raise ValueError(f"fixed split manifest {split} ids contain duplicates")
    return ids


def load_fixed_source_split_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"fixed source split manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"fixed source split manifest must contain a JSON object: {manifest_path}")
    format_version = str(payload.get("format_version") or FORMAT_VERSION)
    if format_version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported fixed split format_version={format_version!r}; expected {FORMAT_VERSION!r}"
        )
    eval_ids = _ids_from_payload(payload, "eval")
    test_ids = _ids_from_payload(payload, "test")
    overlap = set(eval_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"fixed eval/test ids overlap: {sorted(overlap)[:10]}")
    manifest_id = assignment_manifest_id(eval_ids, test_ids)
    declared_id = str(payload.get("manifest_id") or "").strip()
    if declared_id and declared_id != manifest_id:
        raise ValueError(
            f"fixed split manifest_id mismatch: declared={declared_id}, computed={manifest_id}"
        )
    return {
        "path": manifest_path,
        "file_sha256": file_sha256(manifest_path),
        "manifest_id": manifest_id,
        "format_version": FORMAT_VERSION,
        "split_policy": SPLIT_POLICY,
        "eval_ids": frozenset(eval_ids),
        "test_ids": frozenset(test_ids),
        "payload": payload,
    }


def fixed_split_descriptor(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    if manifest is None:
        return None
    return {
        "format_version": manifest["format_version"],
        "split_policy": manifest["split_policy"],
        "manifest_id": manifest["manifest_id"],
        "file_sha256": manifest["file_sha256"],
        "path": str(manifest["path"]),
        "eval_raw_sample_count": len(manifest["eval_ids"]),
        "test_raw_sample_count": len(manifest["test_ids"]),
        "unknown_source_policy": "train",
    }


def split_for_raw_sample(sample_id: str, manifest: dict[str, Any] | None) -> str | None:
    if manifest is None:
        return None
    sample_id = str(sample_id)
    if sample_id in manifest["eval_ids"]:
        return "eval"
    if sample_id in manifest["test_ids"]:
        return "test"
    return "train"


def validate_fixed_holdout_coverage(
    raw_sample_ids_by_split: dict[str, Iterable[str]],
    manifest: dict[str, Any],
    allow_missing: bool = False,
) -> dict[str, Any]:
    actual = {
        split: {str(item) for item in raw_sample_ids_by_split.get(split, [])}
        for split in SPLITS
    }
    requested = {
        "eval": set(manifest["eval_ids"]),
        "test": set(manifest["test_ids"]),
    }
    overlap_errors = {
        f"{left}_{right}": sorted(actual[left] & actual[right])
        for left, right in (("train", "eval"), ("train", "test"), ("eval", "test"))
        if actual[left] & actual[right]
    }
    missing = {split: sorted(requested[split] - actual[split]) for split in ("eval", "test")}
    unexpected = {split: sorted(actual[split] - requested[split]) for split in ("eval", "test")}
    leaked_to_train = sorted((requested["eval"] | requested["test"]) & actual["train"])
    wrong_holdout = {
        "eval_in_test": sorted(requested["eval"] & actual["test"]),
        "test_in_eval": sorted(requested["test"] & actual["eval"]),
    }
    errors = []
    if overlap_errors:
        errors.append(f"actual split overlap={overlap_errors}")
    if any(unexpected.values()):
        errors.append(f"unexpected holdout ids={unexpected}")
    if leaked_to_train:
        errors.append(f"fixed holdout leaked into train={leaked_to_train[:20]}")
    if any(wrong_holdout.values()):
        errors.append(f"fixed holdout assigned to wrong split={wrong_holdout}")
    if not allow_missing and any(missing.values()):
        errors.append(f"fixed holdout ids are missing={missing}")
    report = {
        "status": "passed" if not errors else "failed",
        "manifest_id": manifest["manifest_id"],
        "allow_missing_fixed_holdouts": bool(allow_missing),
        "requested_counts": {split: len(requested[split]) for split in ("eval", "test")},
        "actual_counts": {split: len(actual[split]) for split in SPLITS},
        "missing_ids": missing,
        "unexpected_holdout_ids": unexpected,
        "leaked_to_train_ids": leaked_to_train,
        "wrong_holdout_ids": wrong_holdout,
        "errors": errors,
    }
    if errors:
        raise ValueError("fixed source split validation failed: " + "; ".join(errors))
    return report
