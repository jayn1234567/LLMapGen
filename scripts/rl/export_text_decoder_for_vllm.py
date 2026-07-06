#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mllm.rl.export import export_text_decoder_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a multimodal checkpoint's Qwen text decoder for vLLM prompt-embed rollout.")
    parser.add_argument("--checkpoint", required=True, help="Full no-DeepStack multimodal checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Output text-decoder checkpoint directory for vLLM.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = export_text_decoder_checkpoint(args.checkpoint, args.output_dir, overwrite=args.overwrite)
    print(json.dumps({"vllm_text_model": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
