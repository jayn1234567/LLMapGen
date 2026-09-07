import json
import tempfile
import unittest
from pathlib import Path

from scripts.tools.validate_swift_grpo_dataset import validate_dataset


class ValidateSwiftGrpoDatasetTest(unittest.TestCase):
    def _row(self, image):
        return {
            "sample_id": "sample-1",
            "messages": [{"role": "user", "content": "<image>\nMap it"}],
            "images": [image],
            "solution": "{\"lines\":[]}",
            "coord_config": {"coord_mode": "norm1000", "patch_size": 256, "coord_range": 1000},
            "map_task": "lane_intersection",
        }

    def test_valid_row_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            (root / "images" / "sample.png").write_bytes(b"placeholder")
            dataset = root / "swift.jsonl"
            dataset.write_text(json.dumps(self._row("images/sample.png")) + "\n", encoding="utf-8")
            summary = validate_dataset(dataset, root, expected_images=1, limit=0, strict=False, summary_json=None)
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["records_checked"], 1)

    def test_missing_image_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "swift.jsonl"
            dataset.write_text(json.dumps(self._row("images/missing.png")) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_dataset(dataset, root, expected_images=1, limit=0, strict=False, summary_json=None)

    def test_norm1000_intersection_solution_uses_row_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            (root / "images" / "sample.png").write_bytes(b"placeholder")
            row = self._row("images/sample.png")
            row["solution"] = json.dumps({
                "lines": [{
                    "category": "intersection",
                    "is_cut": False,
                    "points": [[0, 0], [1000, 0], [1000, 1000], [0, 0]],
                }]
            })
            dataset = root / "swift.jsonl"
            dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
            summary = validate_dataset(dataset, root, expected_images=1, limit=0, strict=False, summary_json=None)
            self.assertEqual(summary["status"], "ok")


if __name__ == "__main__":
    unittest.main()
