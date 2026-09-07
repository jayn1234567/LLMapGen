import json
import tempfile
import unittest
from pathlib import Path

from scripts.tools.prepare_swift_grpo_dataset import convert_record


class PrepareSwiftGrpoDatasetTest(unittest.TestCase):
    def test_ground_truth_is_used_and_prediction_is_audit_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "images" / "train" / "sample.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"placeholder")
            ground_truth = {"lines": [{"category": "centerline", "points": [[0, 0], [1, 1]]}]}
            row = {
                "record_id": "sample-1",
                "image": "/old-host-only/sample.png",
                "image_relpath": "images/train/sample.png",
                "conversations": [
                    {"from": "human", "value": "<|im_start|>user\n<image>\nMap it<|im_end|>"},
                    {"from": "gpt", "value": json.dumps(ground_truth)},
                ],
                "prediction": {"lines": []},
                "meta": {"coord_mode": "norm1000", "patch_size": 256},
            }
            converted, audit = convert_record(row, root, strict=True)
            self.assertEqual(converted["sample_id"], "sample-1")
            self.assertEqual(converted["images"], ["images/train/sample.png"])
            self.assertEqual(json.loads(converted["solution"]), ground_truth)
            self.assertNotIn("prediction", converted)
            self.assertTrue(audit["prediction_present_in_source"])


if __name__ == "__main__":
    unittest.main()
