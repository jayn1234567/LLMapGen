#!/usr/bin/env python3
"""Train the cleaned DINOv2 -> Qwen RC centerline JSON model."""

from __future__ import annotations

import sys
from pathlib import Path


def _resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "unimapgen").is_dir():
            return parent
    return current.parents[1]


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unimapgen.dinov2_centerline.train import main  # noqa: E402


if __name__ == "__main__":
    main()

