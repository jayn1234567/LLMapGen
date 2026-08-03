import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY = REPO_ROOT / "scripts/npu/test/eval_external_local256_predictions_fresh_obs_original_e2e_npu.sh"
PIPELINE = REPO_ROOT / "scripts/npu/test/eval_rc_e2e_context512_roi256_checkpoint12504_original_pipeline_npu.sh"


class ExternalLocal256E2EEvalScriptTest(unittest.TestCase):
    def test_entry_uses_external_results_without_gt_or_model_inference(self):
        text = ENTRY.read_text(encoding="utf-8")
        self.assertIn(
            "/cache/xyk/results/xyk_test_phase_a_lane_intersection_output_256_e2e_20260731_063027/json",
            text,
        )
        self.assertIn("E2E_DATA_SOURCE=auto", text)
        self.assertIn('E2E_RAW_ROOT="${RUN_WORK_ROOT}/unused_raw_e2e_data"', text)
        self.assertIn("PREDICTION_COORD_SCALE=0.256", text)
        self.assertIn("RUN_ALL_EVAL=True", text)
        self.assertIn("RUN_LOW_EVAL=True", text)
        self.assertIn("RUN_HIGH_EVAL=True", text)
        self.assertIn("EXPECTED_E2E_SCENES=\"${EXPECTED_E2E_SCENES}\"", text)
        self.assertIn("RESET_PREPARED_E2E_DATA=${RESET_PREPARED_E2E_DATA:-False}", text)
        self.assertIn("FAIL_ON_INVALID_PREDICTIONS=${FAIL_ON_INVALID_PREDICTIONS:-False}", text)
        self.assertIn('FAIL_ON_INVALID_PREDICTIONS="${FAIL_ON_INVALID_PREDICTIONS}"', text)
        self.assertNotIn("suppress_e2e_predictions_without_patch_gt.py", text)
        self.assertNotIn("infer_centerline", text)

    def test_generic_pipeline_supports_strict_prediction_validation(self):
        text = PIPELINE.read_text(encoding="utf-8")
        self.assertIn("FAIL_ON_INVALID_PREDICTIONS=${FAIL_ON_INVALID_PREDICTIONS:-False}", text)
        self.assertIn('is_true "${FAIL_ON_INVALID_PREDICTIONS}"', text)
        self.assertIn("invalid_predictions.json", text)


if __name__ == "__main__":
    unittest.main()
