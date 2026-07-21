import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.tools.remap_fixed_eval_to_dataset import main


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def old_record(sample_id="tile_a_r001_c002"):
    return {
        "id": sample_id,
        "image": f"images/test/tile_a/{sample_id}.png",
        "meta": {"coord_mode": "norm1000", "patch_size": 256},
        "conversations": [
            {"from": "human", "value": "<image>\nold prompt"},
            {"from": "gpt", "value": json.dumps({"lines": []})},
        ],
    }


def new_record(*, split, include_grid_id=True):
    meta = {
        "tile_id": "tile_a",
        "x0": 512,
        "y0": 256,
        "pixel_patch_size": 256,
        "coord_mode": "norm1000",
    }
    if include_grid_id:
        meta["grid_patch_id"] = "tile_a_r001_c002"
    return {
        "id": "tile_a_x00512_y00256",
        "image": f"images/{split}/tile_a/tile_a_x00512_y00256.png",
        "meta": meta,
        "conversations": [
            {"from": "human", "value": "<image>\nnew semantic prompt"},
            {
                "from": "gpt",
                "value": json.dumps(
                    {
                        "lines": [
                            {
                                "category": "centerline",
                                "lane_type": "right_turn",
                                "start_type": "cut",
                                "end_type": "inside",
                                "points": [[0, 500], [500, 500]],
                            }
                        ]
                    }
                ),
            },
        ],
    }


class FixedEvalRemapTest(unittest.TestCase):
    def run_tool(self, root: Path, target_record, split, extra_args=None):
        reference_dir = root / "reference"
        target_root = root / "target"
        output_dir = root / "output"
        write_jsonl(reference_dir / "easy.jsonl", [old_record()])
        write_jsonl(target_root / "phase_a" / f"{split}.jsonl", [target_record])
        argv = [
            "remap_fixed_eval_to_dataset.py",
            "--reference-dir",
            str(reference_dir),
            "--target-dataset-root",
            str(target_root),
            "--output-dir",
            str(output_dir),
            "--reference-difficulties",
            "easy",
            "--scan-target-splits",
            split,
            "--progress-every",
            "0",
        ]
        if extra_args:
            argv.extend(extra_args)
        with patch.object(sys, "argv", argv):
            main()
        return output_dir

    def test_preserved_grid_id_maps_to_new_record_and_new_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.run_tool(Path(directory), new_record(split="test"), "test")

            remapped = json.loads((output / "easy.jsonl").read_text(encoding="utf-8"))
            target = json.loads(remapped["conversations"][1]["value"])
            report = json.loads((output / "mapping_report.json").read_text(encoding="utf-8"))

            self.assertEqual(remapped["id"], "tile_a_x00512_y00256")
            self.assertEqual(target["lines"][0]["lane_type"], "right_turn")
            self.assertEqual(report["selected_count"], 1)
            self.assertEqual(report["match_method_counts"], {"preserved_patch_id": 1})

    def test_coordinate_fallback_matches_without_grid_id(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.run_tool(
                Path(directory),
                new_record(split="eval", include_grid_id=False),
                "eval",
            )
            report = json.loads((output / "mapping_report.json").read_text(encoding="utf-8"))

            self.assertEqual(report["selected_count"], 1)
            self.assertEqual(report["match_method_counts"], {"tile_xy": 1})

    def test_target_train_match_is_reported_but_excluded_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.run_tool(Path(directory), new_record(split="train"), "train")
            report = json.loads((output / "mapping_report.json").read_text(encoding="utf-8"))

            self.assertEqual(report["selected_count"], 0)
            self.assertEqual(report["status_counts"], {"disallowed_target_split": 1})
            self.assertEqual(report["target_split_counts_before_filter"], {"train": 1})
            self.assertTrue(report["fair_holdout_only"])


if __name__ == "__main__":
    unittest.main()
