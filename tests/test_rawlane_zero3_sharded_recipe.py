import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_rawlane_local256_550k_original_dinov2_caprl4b_nodeepstack_npu.sh"
ZERO3_CONFIG = REPO_ROOT / "scripts/deepspeed_zero3_no_merge.json"
DI_STEP10_SMOKE = REPO_ROOT / "scripts/npu/test/di_smoke_sft_stage_a_lane_intersection_rawlane_local256_550k_zero3_step10_npu.sh"


class RawLaneZero3ShardedRecipeTest(unittest.TestCase):
    def test_recipe_keeps_batch_four_and_never_gathers_during_save(self):
        script = TRAIN_SCRIPT.read_text(encoding="utf-8")
        config = json.loads(ZERO3_CONFIG.read_text(encoding="utf-8"))

        self.assertIn("PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}", script)
        self.assertIn("DEEPSPEED_CONFIG=scripts/deepspeed_zero3_no_merge.json", script)
        self.assertIn("--save_on_each_node True", script)
        self.assertIn("MLLM_NPU_EMPTY_CACHE_BEFORE_CHECKPOINT", script)
        self.assertIn("MLLM_SKIP_DISTRIBUTED_FLOS_ON_SAVE", script)
        self.assertIn("zero_shards/node_${NODE_RANK}", script)
        self.assertFalse(
            config["zero_optimization"]["gather_16bit_weights_on_model_save"]
        )

    def test_di_smoke_saves_at_step_ten_without_changing_formal_defaults(self):
        formal_script = TRAIN_SCRIPT.read_text(encoding="utf-8")
        smoke_script = DI_STEP10_SMOKE.read_text(encoding="utf-8")

        self.assertIn("SAVE_STEPS=${SAVE_STEPS:-1000}", formal_script)
        self.assertIn("export MAX_STEPS=20", smoke_script)
        self.assertIn("export SAVE_STEPS=10", smoke_script)
        self.assertIn("export PER_DEVICE_TRAIN_BATCH_SIZE=4", smoke_script)
        self.assertIn("export TARGET_GLOBAL_BATCH_SIZE=128", smoke_script)
        self.assertIn("exec bash \"${FORMAL_SCRIPT}\"", smoke_script)


if __name__ == "__main__":
    unittest.main()
