#!/usr/bin/env python3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.visualize_state_update_global import *  # noqa: F401,F403
from scripts.tools.visualize_state_update_global import main


if __name__ == "__main__":
    main()
