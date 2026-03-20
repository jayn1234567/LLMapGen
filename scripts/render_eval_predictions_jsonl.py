import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_qwen2_5vl_lora_small_eval import render_panel, sanitize_lines, stack_panels


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-jsonl", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    predictions_jsonl = Path(args.predictions_jsonl)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(predictions_jsonl)
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    for row in rows:
        sample_id = str(row["id"])
        rel_image = str(row["image"])
        image_path = (dataset_root / rel_image).resolve()
        image = Image.open(image_path).convert("RGB")
        gt_lines = sanitize_lines(row.get("gt_lines", []))
        pred_lines = sanitize_lines(row.get("pred_lines", []))
        state_lines = sanitize_lines(row.get("state_lines", []))

        gt_panel = render_panel(
            image=image,
            lines=gt_lines,
            state_lines=state_lines,
            title=f"{sample_id} | GT",
            line_color=(0, 120, 255),
        )
        pred_panel = render_panel(
            image=image,
            lines=pred_lines,
            state_lines=state_lines,
            title=f"{sample_id} | Pred",
            line_color=(255, 60, 60),
        )
        combined = stack_panels(gt_panel, pred_panel)
        combined.save(output_dir / f"{sample_id}.png")

    print(json.dumps({"saved": len(rows), "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
