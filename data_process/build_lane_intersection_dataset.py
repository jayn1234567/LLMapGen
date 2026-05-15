#!/usr/bin/env python3
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from state_update_dataset_common import run_cli


if __name__ == "__main__":
    run_cli(
        include_intersections=True,
        description="Build Phase A/B lane + intersection SFT datasets from raw RC TIFF/GeoJSON samples.",
    )
