import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from data_process.build_dataset_v2 import dataset_variant_specs
from data_process.build_dataset_v2_staged import selected_variants
from data_process.state_update_dataset_common import IGNORED_LANE_TYPE_CODES, SEMANTIC_SCHEMA_VERSION
from scripts.tools.derive_intersection_prompt_dataset import (
    TASK_MODE,
    derive_dataset,
    extract_prompt_intersections,
    parse_target,
    transform_record,
)
from scripts.tools.build_rc_dataset_v2_local512_windows import parse_args, sample_count_label
from scripts.tools.validate_visualize_rc_dataset_v2 import audit


def source_record(sample_id: str, split: str) -> dict:
    centerline = {
        "category": "centerline",
        "lane_type": "common",
        "start_type": "cut",
        "end_type": "cut",
        "points": [[0, 500], [1000, 500]],
    }
    intersection = {
        "category": "intersection",
        "intersection_type": "t_intersection",
        "is_cut": False,
        "points": [[300, 300], [300, 700], [700, 700], [700, 300], [300, 300]],
    }
    return {
        "id": sample_id,
        "image": f"images/{split}/{sample_id}/{sample_id}.png",
        "meta": {
            "tile_id": sample_id,
            "coord_mode": "norm1000",
            "coord_range": 1000,
            "pixel_patch_size": 512,
            "patch_width": 512,
            "patch_height": 512,
            "target_size": 512,
            "context_image_size": 512,
            "view_mode": "local512",
            "target_roi_in_image": [0, 0, 512, 512],
        },
        "conversations": [
            {"from": "human", "value": "<image>\nstandard source"},
            {"from": "gpt", "value": json.dumps({"lines": [centerline, intersection]})},
        ],
    }


class Local512IntersectionPromptTest(unittest.TestCase):
    def test_local512_builder_defaults_to_200k_quick_subset(self):
        args = parse_args(["--work-root", "work", "--obsutil-path", "obsutil"])
        self.assertEqual(args.quick_train_target_samples, 200000)
        self.assertEqual(sample_count_label(args.quick_train_target_samples), "200k")

    def test_sample_count_label_supports_non_thousand_counts(self):
        self.assertEqual(sample_count_label(12345), "12345")
        with self.assertRaises(ValueError):
            sample_count_label(0)

    def test_true_512_variant_naming(self):
        specs = dataset_variant_specs(Path("output"), "local", 512, 512)
        self.assertEqual(list(specs), ["local512"])
        self.assertEqual(specs["local512"]["context_size"], 512)
        self.assertEqual(selected_variants("local", 512, 512), ["local512"])

    def test_transform_moves_intersections_to_prompt(self):
        transformed, centerlines, intersections = transform_record(source_record("sample", "train"))
        target = parse_target(transformed)
        self.assertEqual(target["lines"], centerlines)
        self.assertTrue(centerlines)
        self.assertFalse(any(item["category"] == "intersection" for item in target["lines"]))
        prompt = transformed["conversations"][0]["value"]
        self.assertEqual(extract_prompt_intersections(prompt), intersections)
        self.assertIn('"waiting_area"', prompt)
        self.assertIn('"bus_lane"', prompt)
        self.assertIn('"main_auxiliary_connector"', prompt)
        self.assertEqual(transformed["meta"]["task_mode"], TASK_MODE)
        self.assertEqual(transformed["meta"]["dataset_variant"], "local512_intersection_prompt")

    def test_derived_dataset_passes_true_512_audit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "local512"
            output_root = root / "local512_intersection_prompt"
            phase_root = source_root / "phase_a"
            phase_root.mkdir(parents=True)
            for split in ("train", "eval", "test"):
                record = source_record(f"{split}_sample", split)
                (phase_root / f"{split}.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
                (phase_root / f"meta_{split}.jsonl").write_text(
                    json.dumps({"id": record["id"], "image": record["image"], "meta": record["meta"]}) + "\n",
                    encoding="utf-8",
                )
                image_path = source_root / Path(record["image"])
                image_path.parent.mkdir(parents=True)
                Image.new("RGB", (512, 512), (20, 30, 40)).save(image_path)
            (source_root / "dataset_info.json").write_text(
                json.dumps({
                    "dataset_version": "test_local512",
                    "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
                    "ignored_source_lane_type_codes": sorted(IGNORED_LANE_TYPE_CODES),
                    "semantic_validation_passed": True,
                    "balance": {
                        "final_bucket_counts": {"empty": 0, "easy": 0, "medium": 1, "hard": 0, "very_hard": 0},
                        "actual_intersection_ratio": 1.0,
                    },
                }),
                encoding="utf-8",
            )
            args = SimpleNamespace(copy_mode="hardlink", resume=False, progress_every=0)
            derive_dataset(source_root, output_root, args)

            report, errors = audit(SimpleNamespace(
                dataset_root=str(output_root),
                output_dir=str(root / "audit"),
                phase="phase_a",
                variant="local512_intersection_prompt",
                expected_train_samples=1,
                difficulty_ratios="empty=0,easy=0,medium=1,hard=0,very_hard=0",
                expected_intersection_ratio=1.0,
                count_tolerance=0,
                skip_distribution_check=True,
                visualize_per_difficulty=0,
                allow_short_visual_buckets=True,
                seed=7,
                progress_every=0,
                image_decode_mode="all",
                image_decode_samples_per_split=10,
                skip_extra_image_scan=False,
                max_error_examples=50,
            ))
            self.assertEqual(errors.total, 0, report["error_examples"])
            self.assertEqual(report["constraints"]["expected_image_size"], [512, 512])
            self.assertEqual(report["splits"]["train"]["intersection_samples"], 1)


if __name__ == "__main__":
    unittest.main()
