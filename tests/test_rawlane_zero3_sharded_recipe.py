import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_rawlane_local256_550k_original_dinov2_caprl4b_nodeepstack_npu.sh"
ZERO3_CONFIG = REPO_ROOT / "scripts/deepspeed_zero3_no_merge.json"


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


if __name__ == "__main__":
    unittest.main()
