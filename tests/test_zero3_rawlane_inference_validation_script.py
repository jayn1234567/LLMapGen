import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/npu/test/validate_zero3_rawlane_step20_inference_torch240_npu.sh"


class Zero3RawLaneInferenceValidationScriptTest(unittest.TestCase):
    def test_validation_covers_download_merge_and_real_inference(self):
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("CHECKPOINT_NAME=${CHECKPOINT_NAME:-checkpoint-20}", text)
        self.assertIn("EXPECTED_NODES=${EXPECTED_NODES:-4}", text)
        self.assertIn("EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE:-32}", text)
        self.assertIn('f"Incomplete ZeRO checkpoint: optimizer shards={len(optimizer_shards)}', text)
        self.assertIn("merge_zero3_multinode_checkpoint.sh", text)
        self.assertIn("infer_centerline_checkpoint.py", text)
        self.assertIn("--prompt-mode dataset", text)
        self.assertIn("--map-task lane_intersection", text)
        self.assertIn("ZERO-3 CHECKPOINT LOAD + INFERENCE PASSED", text)


if __name__ == "__main__":
    unittest.main()
