import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from data_process.state_update_dataset_common import SEMANTIC_SCHEMA_VERSION
from scripts.tools.validate_visualize_rc_dataset_v2 import (
    ErrorCollector,
    audit,
    resolve_expected_difficulty_counts,
    validate_target_lines,
)


PROMPT = """<image>
Please construct the map.
Coordinates use a normalized 0-1000 grid over the original 256x256 image patch.
Every centerline includes lane_type and every intersection includes intersection_type.
"""


def make_record(sample_id, split, tile_id, lines):
    return {
        "id": sample_id,
        "image": f"images/{split}/{tile_id}/{sample_id}.png",
        "meta": {
            "tile_id": tile_id,
            "coord_mode": "norm1000",
            "coord_range": 1000,
            "pixel_patch_size": 256,
            "patch_width": 256,
            "patch_height": 256,
            "target_size": 256,
            "context_image_size": 256,
            "view_mode": "local256",
            "target_roi_in_image": [0, 0, 256, 256],
        },
        "conversations": [
            {"from": "human", "value": PROMPT},
            {"from": "gpt", "value": json.dumps({"lines": lines})},
        ],
    }


class ValidateVisualizeDatasetV2Test(unittest.TestCase):
    def test_small_valid_dataset_passes_and_renders_visuals(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "local256"
            output_dir = root / "audit"
            (dataset_root / "phase_a").mkdir(parents=True)
            (dataset_root / "dataset_info.json").write_text(
                json.dumps({
                    "dataset_version": "rc_dataset_v2_staged_stage_a_semantic_v1",
                    "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
                    "semantic_validation_passed": True,
                }),
                encoding="utf-8",
            )

            centerline = {
                "category": "centerline",
                "lane_type": "common",
                "start_type": "cut",
                "end_type": "cut",
                "points": [[0, 500], [1000, 500]],
            }
            intersection = {
                "category": "intersection",
                "intersection_type": "other",
                "is_cut": False,
                "points": [[100, 100], [900, 100], [900, 900], [100, 100]],
            }
            records = {
                "train": [
                    make_record("train_easy", "train", "tile_train_easy", [centerline]),
                    make_record("train_inter", "train", "tile_train_inter", [centerline, intersection]),
                ],
                "eval": [make_record("eval_easy", "eval", "tile_eval", [centerline])],
                "test": [make_record("test_easy", "test", "tile_test", [centerline])],
            }
            for split, split_records in records.items():
                with (dataset_root / "phase_a" / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
                    for record in split_records:
                        handle.write(json.dumps(record) + "\n")
                        image_path = dataset_root / record["image"]
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        Image.new("RGB", (256, 256), (20, 20, 20)).save(image_path)

            args = SimpleNamespace(
                dataset_root=str(dataset_root),
                output_dir=str(output_dir),
                phase="phase_a",
                expected_train_samples=2,
                difficulty_ratios="empty=0,easy=0.30,medium=0.33,hard=0.27,very_hard=0.10",
                expected_intersection_ratio=-1.0,
                count_tolerance=0,
                skip_distribution_check=True,
                visualize_per_difficulty=1,
                allow_short_visual_buckets=True,
                seed=7,
                progress_every=0,
                image_decode_mode="all",
                image_decode_samples_per_split=1,
                skip_extra_image_scan=False,
                max_error_examples=100,
            )
            report, errors = audit(args)

            self.assertEqual(errors.total, 0)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["splits"]["train"]["samples"], 2)
            self.assertEqual(report["referenced_pngs"], 4)
            self.assertEqual(report["actual_pngs"], 4)
            self.assertTrue((output_dir / "validation_report.json").is_file())
            self.assertTrue((output_dir / "visualization_samples.jsonl").is_file())
            self.assertTrue(any((output_dir / "visualizations").rglob("*.png")))

    def test_missing_semantic_field_is_rejected(self):
        errors = ErrorCollector(max_examples=10)
        validate_target_lines(
            [{
                "category": "centerline",
                "start_type": "cut",
                "end_type": "cut",
                "points": [[0, 0], [1000, 1000]],
            }],
            errors,
            "train",
            1,
            "missing_lane_type",
        )
        self.assertEqual(errors.counts["invalid_lane_type"], 1)

    def test_intersection_subtype_key_variants_are_rejected(self):
        errors = ErrorCollector(max_examples=10)
        validate_target_lines(
            [{
                "category": "intersection",
                "intersection_type": "common",
                "Intersection_SubType": "1",
                "is_cut": False,
                "points": [[0, 0], [1000, 0], [1000, 1000], [0, 0]],
            }],
            errors,
            "train",
            1,
            "stale_subtype",
        )
        self.assertEqual(errors.counts["unexpected_intersection_subtype"], 1)

    def test_final_build_quotas_override_requested_ratios_after_redistribution(self):
        ratios = {
            "empty": 0.0,
            "easy": 0.30,
            "medium": 0.33,
            "hard": 0.27,
            "very_hard": 0.10,
        }
        final_counts = {
            "empty": 0,
            "easy": 165000,
            "medium": 195816,
            "hard": 134184,
            "very_hard": 55000,
        }
        expected, requested, source = resolve_expected_difficulty_counts(
            550000,
            ratios,
            {"balance": {"final_bucket_counts": final_counts}},
        )
        self.assertEqual(expected, final_counts)
        self.assertEqual(requested["medium"], 181500)
        self.assertEqual(requested["hard"], 148500)
        self.assertEqual(source, "dataset_info.balance.final_bucket_counts")


if __name__ == "__main__":
    unittest.main()
