#!/usr/bin/env python3
"""Prepare a writable RC E2E run tree with resumable local I/O."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import zipfile
from pathlib import Path


MARKER_NAME = ".e2e_source_prepare_complete.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-root", help="Extracted E2E root, possibly with one wrapper directory.")
    source.add_argument("--archive", help="E2E zip archive used when no extracted source is available.")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--allowed-root", required=True)
    parser.add_argument("--reset", action="store_true", help="Delete destination before preparing it.")
    parser.add_argument("--progress-files", type=int, default=1000)
    return parser.parse_args()


def assert_destination_allowed(destination: Path, allowed_root: Path) -> None:
    if destination != allowed_root and allowed_root not in destination.parents:
        raise ValueError(f"Refusing to prepare E2E data outside {allowed_root}: {destination}")


def resolve_scene_root(root: Path) -> Path:
    candidates = [root, *sorted(path for path in root.iterdir() if path.is_dir())]
    for candidate in candidates:
        if any(
            (child / "rc_one_patch_release").is_dir()
            for child in candidate.iterdir()
            if child.is_dir()
        ):
            return candidate
    raise FileNotFoundError(f"Unable to resolve extracted E2E scene root below {root}")


def marker_matches(marker_path: Path, source_kind: str, source_path: Path) -> bool:
    if not marker_path.is_file():
        return False
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return payload.get("source_kind") == source_kind and payload.get("source_path") == str(source_path)


def same_local_file(source: Path, destination: Path) -> bool:
    if not destination.is_file():
        return False
    source_stat = source.stat()
    destination_stat = destination.stat()
    return (
        source_stat.st_size == destination_stat.st_size
        and source_stat.st_mtime_ns == destination_stat.st_mtime_ns
    )


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def maybe_report(*, scanned: int, copied: int, skipped: int, copied_bytes: int, interval: int) -> None:
    if scanned == 1 or scanned % max(1, interval) == 0:
        print(
            "[original-e2e-data] "
            f"scanned={scanned} copied={copied} skipped={skipped} "
            f"copied_gib={copied_bytes / (1024 ** 3):.2f}",
            flush=True,
        )


def copy_tree_resumable(source: Path, destination: Path, progress_files: int) -> dict:
    scanned = copied = skipped = copied_bytes = 0
    for source_file in source.rglob("*"):
        if not source_file.is_file():
            continue
        relative = source_file.relative_to(source)
        destination_file = destination / relative
        scanned += 1
        if same_local_file(source_file, destination_file):
            skipped += 1
        else:
            size = source_file.stat().st_size
            atomic_copy(source_file, destination_file)
            copied += 1
            copied_bytes += size
        maybe_report(
            scanned=scanned,
            copied=copied,
            skipped=skipped,
            copied_bytes=copied_bytes,
            interval=progress_files,
        )
    return {
        "files_scanned": scanned,
        "files_copied": copied,
        "files_skipped": skipped,
        "bytes_copied": copied_bytes,
    }


def zip_member_destination(destination: Path, member_name: str) -> Path:
    resolved = (destination / member_name).resolve()
    if resolved != destination and destination not in resolved.parents:
        raise ValueError(f"Unsafe zip member path: {member_name}")
    return resolved


def extract_zip_resumable(archive: Path, destination: Path, progress_files: int) -> dict:
    scanned = copied = skipped = copied_bytes = 0
    with zipfile.ZipFile(archive) as handle:
        members = sorted((member for member in handle.infolist() if not member.is_dir()), key=lambda item: item.filename)
        for member in members:
            destination_file = zip_member_destination(destination, member.filename)
            scanned += 1
            if destination_file.is_file() and destination_file.stat().st_size == member.file_size:
                skipped += 1
            else:
                destination_file.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination_file.with_name(f".{destination_file.name}.part-{os.getpid()}")
                try:
                    with handle.open(member) as source_handle, temporary.open("wb") as destination_handle:
                        shutil.copyfileobj(source_handle, destination_handle, length=8 * 1024 * 1024)
                    os.replace(temporary, destination_file)
                finally:
                    temporary.unlink(missing_ok=True)
                copied += 1
                copied_bytes += member.file_size
            maybe_report(
                scanned=scanned,
                copied=copied,
                skipped=skipped,
                copied_bytes=copied_bytes,
                interval=progress_files,
            )
    return {
        "files_scanned": scanned,
        "files_copied": copied,
        "files_skipped": skipped,
        "bytes_copied": copied_bytes,
    }


def prepare(args: argparse.Namespace) -> dict:
    destination = Path(args.destination).resolve()
    allowed_root = Path(args.allowed_root).resolve()
    assert_destination_allowed(destination, allowed_root)
    if args.reset and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    if args.source_root:
        source_kind = "directory"
        source_path = resolve_scene_root(Path(args.source_root).resolve())
    else:
        source_kind = "zip"
        source_path = Path(args.archive).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

    marker_path = destination / MARKER_NAME
    if marker_matches(marker_path, source_kind, source_path):
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
        payload["reused_complete_tree"] = True
        print(f"[original-e2e-data] reuse completed run data: {destination}", flush=True)
        return payload

    print(
        f"[original-e2e-data] resumable prepare: {source_kind} {source_path} -> {destination}",
        flush=True,
    )
    if source_kind == "directory":
        stats = copy_tree_resumable(source_path, destination, args.progress_files)
    else:
        stats = extract_zip_resumable(source_path, destination, args.progress_files)
    payload = {
        "source_kind": source_kind,
        "source_path": str(source_path),
        "destination": str(destination),
        **stats,
        "completed_at_unix": time.time(),
        "reused_complete_tree": False,
    }
    marker_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
