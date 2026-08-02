import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / (
    "scripts/npu/test/"
    "compare_rawlane_local256_200k_vs_context512_roi256_200k_"
    "fixed1100_torch240_npu.sh"
)


class RawLane200kSeed42EvalRecipeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = SCRIPT_PATH.read_text(encoding="utf-8")

    def test_uses_the_two_requested_checkpoints(self):
        self.assertIn("2026/07/29/3bf5a8001ec6433ca4ee973564c29976", self.script)
        self.assertIn("ma-job-a782316a-32ec-4958-ae1f-44c69fdedd3f", self.script)
        self.assertIn("2026/07/30/ea77c85a9d54442b825b86cd7f547a26", self.script)
        self.assertIn("ma-job-2e7c82dd-3a05-440b-b686-db5c3bcc2512", self.script)

    def test_builds_independent_seed42_fixed_sets(self):
        expected = (
            "FIXED_EVAL_COUNTS=${FIXED_EVAL_COUNTS:-easy=300,medium=300,hard=300,very_hard=200}",
            "FIXED_EVAL_SEED=${FIXED_EVAL_SEED:-42}",
            "rawlane_local256_200k_fixed1100_e300_m300_h300_vh200_seed42_v1",
            "rawlane_context512_roi256_200k_fixed1100_e300_m300_h300_vh200_seed42_v1",
            '"same_seed": True',
            '"same_sample_ids": False',
            '"same_ground_truth": False',
        )
        for fragment in expected:
            self.assertIn(fragment, self.script)
        self.assertNotIn("--require-all", self.script)
        self.assertNotIn("remap_fixed_eval_to_dataset.py", self.script)

    def test_uses_each_rawlane_dataset_directly(self):
        expected = (
            "local256_200k_rawlane/local256_200k.tar",
            "context512_roi256_200k_rawlane/context512_roi256_200k.tar",
            'EVAL_SOURCE_JSONL="${RAWLANE_LOCAL_DATASET_ROOT}/phase_a/eval.jsonl"',
            'EVAL_SOURCE_JSONL="${RAWLANE_CONTEXT_DATASET_ROOT}/phase_a/eval.jsonl"',
        )
        for fragment in expected:
            self.assertIn(fragment, self.script)

    def test_runs_batched_multidevice_inference_and_writes_summary(self):
        expected = (
            "NPROC_PER_NODE=${NPROC_PER_NODE:-6}",
            "PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-4}",
            "rawlane_local256_200k/checkpoint-12504/by_difficulty/all_selected/eval.json",
            "rawlane_context512_roi256_200k/checkpoint-12504/by_difficulty/all_selected/eval.json",
            "rawlane200k_seed42_independent_eval_summary.json",
        )
        for fragment in expected:
            self.assertIn(fragment, self.script)


if __name__ == "__main__":
    unittest.main()
