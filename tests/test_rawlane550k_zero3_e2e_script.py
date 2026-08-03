import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/npu/test/eval_rawlane550k_zero3_globalstep34376_gt_empty_fresh_obs_original_e2e_npu.sh"


class RawLane550kZero3E2EScriptTest(unittest.TestCase):
    def test_recipe_merges_then_runs_comparable_full_e2e(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("3fce4c245d294c20a99be5699e5269cc", text)
        self.assertIn("CHECKPOINT_NAME=${CHECKPOINT_NAME:-global_step34376}", text)
        self.assertIn("MERGE_ONLY=True", text)
        self.assertIn("merged_${CHECKPOINT_NAME}", text)
        self.assertIn("PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-32}", text)
        self.assertIn("eval_rawlane200k_checkpoint12504_gt_empty_fresh_obs_original_e2e_npu.sh", text)
        self.assertIn("EXPECTED_E2E_SCENES=110", text)
        self.assertIn("GT-empty suppression + all/low/high", text)


if __name__ == "__main__":
    unittest.main()
