from __future__ import annotations

import json
import os
from types import SimpleNamespace

from scripts.tools.prepare_rc_e2e_original_run_data import MARKER_NAME, prepare


def test_resumable_directory_prepare_skips_existing_files(tmp_path):
    source_root = tmp_path / "raw"
    scene_root = source_root / "scene_1"
    release_root = scene_root / "rc_one_patch_release"
    release_root.mkdir(parents=True)
    first = release_root / "first.bin"
    second = release_root / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    destination = tmp_path / "runs" / "run_1" / "e2e_data"
    existing = destination / "scene_1" / "rc_one_patch_release" / "first.bin"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(first.read_bytes())
    os.utime(existing, ns=(first.stat().st_atime_ns, first.stat().st_mtime_ns))

    args = SimpleNamespace(
        source_root=str(source_root),
        archive=None,
        destination=str(destination),
        allowed_root=str(tmp_path / "runs"),
        reset=False,
        progress_files=1,
    )
    result = prepare(args)

    assert result["files_scanned"] == 2
    assert result["files_skipped"] == 1
    assert result["files_copied"] == 1
    assert (destination / "scene_1" / "rc_one_patch_release" / "second.bin").read_bytes() == b"second"
    marker = json.loads((destination / MARKER_NAME).read_text(encoding="utf-8"))
    assert marker["source_kind"] == "directory"

    reused = prepare(args)
    assert reused["reused_complete_tree"] is True
