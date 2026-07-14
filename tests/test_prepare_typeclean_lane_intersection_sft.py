import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.tools.prepare_typeclean_lane_intersection_sft import (
    STAGE_A_USER_PROMPT,
    convert_dataset,
)


class TypeCleanLaneIntersectionPreparationTest(unittest.TestCase):
    def make_record(self):
        return {
            "id": "sample-1",
            "image": "images/train/group/sample.png",
            "meta": {"coord_mode": "norm1000"},
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nConstruct the complete road map.",
                },
                {
                    "from": "gpt",
                    "value": json.dumps(
                        {
                            "lines": [
                                {
                                    "category": "centerline",
                                    "points": [[0, 0], [1000, 1000]],
                                    "lane_type": 1,
                                },
                                {
                                    "category": "centerline",
                                    "points": [[0, 1], [1000, 999]],
                                    "lane_type": "2",
                                },
                                {
                                    "category": "centerline",
                                    "points": [[0, 2], [1000, 998]],
                                    "lane_type": 25,
                                },
                                {
                                    "category": "centerline",
                                    "points": [[0, 3], [1000, 997]],
                                    "lane_type": 3,
                                },
                                {
                                    "category": "centerline",
                                    "points": [[0, 4], [1000, 996]],
                                },
                                {
                                    "category": "intersection",
                                    "points": [[0, 0], [1, 0], [0, 0]],
                                    "intersection_type": 1,
                                    "intersection_subtype": 1,
                                },
                                {
                                    "category": "intersection",
                                    "points": [[0, 0], [1, 0], [0, 0]],
                                    "intersection_type": 1,
                                    "intersection_subtype": 2,
                                },
                                {
                                    "category": "intersection",
                                    "points": [[0, 0], [1, 0], [0, 0]],
                                    "intersection_type": "1",
                                    "intersection_subtype": "3",
                                },
                                {
                                    "category": "intersection",
                                    "points": [[0, 0], [1, 0], [0, 0]],
                                    "intersection_type": 4,
                                },
                            ]
                        },
                        separators=(",", ":"),
                    ),
                },
            ],
        }

    def test_conversion_normalizes_open_ended_lane_types_and_intersections(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "raw"
            output_root = root / "normalized"
            phase_root = input_root / "phase_a"
            phase_root.mkdir(parents=True)
            record = self.make_record()
            for split in ("train", "eval", "test"):
                (phase_root / f"{split}.jsonl").write_text(
                    json.dumps(record) + "\n", encoding="utf-8"
                )

            report = convert_dataset(
                input_root,
                output_root,
                "phase_a",
                ["train", "eval", "test"],
                True,
                0,
            )
            converted = json.loads(
                (output_root / "phase_a" / "train.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )

        prompt = converted["conversations"][0]["value"]
        target = json.loads(converted["conversations"][1]["value"])
        centerlines = [line for line in target["lines"] if line["category"] == "centerline"]
        intersections = [
            line for line in target["lines"] if line["category"] == "intersection"
        ]
        self.assertEqual(prompt, STAGE_A_USER_PROMPT)
        self.assertIn("lane_type", prompt)
        self.assertNotIn("Incoming traces", prompt)
        self.assertNotIn("Incoming intersections", prompt)
        self.assertEqual(
            [line["lane_type"] for line in centerlines],
            ["common", "right_turn", "other", "other"],
        )
        self.assertEqual(
            [line["intersection_type"] for line in intersections],
            ["common", "t_intersection", "small_untyped", "t_lane_change_area"],
        )
        self.assertTrue(all("intersection_subtype" not in line for line in intersections))
        self.assertEqual(
            report["splits"]["train"]["source_lane_types"]["25"], 1
        )
        self.assertEqual(report["splits"]["train"]["dropped_u_turn_centerlines"], 1)
        self.assertEqual(report["splits"]["train"]["restored_type4_subtype1"], 1)
        self.assertEqual(
            report["splits"]["train"]["target_intersection_types"],
            {
                "common": 1,
                "t_intersection": 1,
                "small_untyped": 1,
                "t_lane_change_area": 1,
            },
        )

    def test_normalized_dataset_passes_strict_inspection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            phase_root = root / "raw" / "phase_a"
            phase_root.mkdir(parents=True)
            record = self.make_record()
            for split in ("train", "eval"):
                (phase_root / f"{split}.jsonl").write_text(
                    json.dumps(record) + "\n", encoding="utf-8"
                )
            convert_dataset(
                root / "raw",
                root / "normalized",
                "phase_a",
                ["train", "eval"],
                True,
                0,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/tools/inspect_lane_intersection_training_dataset.py",
                    "--dataset-root",
                    str(root / "normalized"),
                    "--image-root",
                    str(root / "raw"),
                    "--phase",
                    "phase_a",
                    "--splits",
                    "train",
                    "eval",
                    "--image-checks-per-split",
                    "0",
                    "--forbid-lane-type",
                    "3",
                    "--allowed-centerline-type",
                    "common",
                    "--allowed-centerline-type",
                    "right_turn",
                    "--allowed-centerline-type",
                    "other",
                    "--allowed-intersection-type",
                    "common",
                    "--allowed-intersection-type",
                    "t_intersection",
                    "--allowed-intersection-type",
                    "small_untyped",
                    "--allowed-intersection-type",
                    "t_lane_change_area",
                    "--require-centerline-type-field",
                    "--require-intersection-type-field",
                    "--forbid-intersection-subtype-field",
                    "--require-taxonomy-prompt",
                    "--strict",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unknown_intersection_pair_fails_instead_of_guessing(self):
        record = self.make_record()
        target = json.loads(record["conversations"][1]["value"])
        target["lines"][-1]["intersection_type"] = 9
        record["conversations"][1]["value"] = json.dumps(target)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            phase_root = root / "raw" / "phase_a"
            phase_root.mkdir(parents=True)
            for split in ("train", "eval"):
                (phase_root / f"{split}.jsonl").write_text(
                    json.dumps(record) + "\n", encoding="utf-8"
                )
            with self.assertRaisesRegex(ValueError, "unsupported intersection pair"):
                convert_dataset(
                    root / "raw",
                    root / "normalized",
                    "phase_a",
                    ["train", "eval"],
                    True,
                    0,
                )


if __name__ == "__main__":
    unittest.main()
