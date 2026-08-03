import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MERGE_SCRIPT = REPO_ROOT / "scripts/tools/merge_zero3_multinode_checkpoint.sh"
EVAL_SCRIPT = REPO_ROOT / "scripts/npu/test/test_local_rawlane550k_zero3_globalstep34376_merge_eval_torch240_npu.sh"


class Zero3GlobalStepMergeEvalScriptTest(unittest.TestCase):
    def test_merge_supports_direct_global_step_layout(self):
        text = MERGE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('[[ "${CHECKPOINT_NAME}" == global_step* ]]', text)
        self.assertIn('printf \'%s\\n\' "${CHECKPOINT_NAME}" > "${ASSEMBLED_DIR}/latest"', text)
        self.assertIn('"${ASSEMBLED_DIR}/${CHECKPOINT_NAME}"', text)

    def test_fixed_recipe_downloads_four_nodes_then_evaluates(self):
        text = EVAL_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("3fce4c245d294c20a99be5699e5269cc", text)
        self.assertIn("CHECKPOINT_NAME=${CHECKPOINT_NAME:-global_step34376}", text)
        self.assertIn("EXPECTED_NODES=${EXPECTED_NODES:-4}", text)
        self.assertIn("EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE:-32}", text)
        self.assertIn("merge_zero3_multinode_checkpoint.sh", text)
        self.assertIn("fixed1100_singlepass_torch240_npu.sh", text)
        self.assertIn("VIS_LIMIT=${VIS_LIMIT:-50}", text)
        self.assertIn("MERGE_ONLY=${MERGE_ONLY:-False}", text)
        self.assertIn('if is_true "${MERGE_ONLY}"', text)


if __name__ == "__main__":
    unittest.main()
