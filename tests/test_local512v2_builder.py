import json
import tempfile
import unittest
from pathlib import Path

from data_process.state_update_dataset_common import SEMANTIC_SCHEMA_VERSION
from scripts.tools.build_rc_dataset_v2_local512v2_windows import (
    DIFFICULTY_RATIOS,
    INTERSECTION_RATIO,
    LOCAL256_STANDARD_VARIANT,
    STANDARD_VARIANT,
    completed_named_variant,
    expected_difficulty_counts,
    parse_args,
    update_variant_metadata,
)
from scripts.tools.validate_visualize_rc_dataset_v2 import VARIANT_SPECS


class Local512V2BuilderTest(unittest.TestCase):
    def test_recipe_defaults(self):
        args = parse_args(["--work-root", "work", "--obsutil-path", "obsutil"])
        self.assertEqual(args.quick_train_target_samples, 200000)
        self.assertEqual(DIFFICULTY_RATIOS, "empty=0,easy=0.20,medium=0.30,hard=0.30,very_hard=0.20")
        self.assertEqual(INTERSECTION_RATIO, 0.30)
        self.assertEqual(
            expected_difficulty_counts(200000),
            {
                "empty": 0,
                "very_easy": 0,
                "easy": 40000,
                "medium": 60000,
                "hard": 60000,
                "very_hard": 40000,
            },
        )

    def test_validator_aliases_keep_true_512_geometry(self):
        standard = VARIANT_SPECS["local512v2"]
        prompt = VARIANT_SPECS["local512v2_intersection_prompt"]
        self.assertEqual(standard["image_size"], (512, 512))
        self.assertEqual(standard["view_mode"], "local512")
        self.assertEqual(prompt["view_mode"], "local512")
        self.assertEqual(VARIANT_SPECS[LOCAL256_STANDARD_VARIANT]["image_size"], (256, 256))

    def test_metadata_relabel_and_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / STANDARD_VARIANT
            phase = root / "phase_a"
            phase.mkdir(parents=True)
            (phase / "train.jsonl").write_text("{}\n", encoding="utf-8")
            info = {
                "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
                "variants": ["local512"],
                "record_counts": {"local512:train": 10},
                "balance": {
                    "final_bucket_counts": expected_difficulty_counts(10),
                    "actual_intersection_ratio": INTERSECTION_RATIO,
                },
            }
            path = root / "dataset_info.json"
            path.write_text(json.dumps(info), encoding="utf-8")
            update_variant_metadata(path, STANDARD_VARIANT)
            updated = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(updated["variants"], [STANDARD_VARIANT])
            self.assertEqual(updated["record_counts"], {f"{STANDARD_VARIANT}:train": 10})
            self.assertTrue(completed_named_variant(root, 10))


if __name__ == "__main__":
    unittest.main()
