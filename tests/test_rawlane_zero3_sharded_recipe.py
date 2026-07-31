import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_rawlane_local256_550k_original_dinov2_caprl4b_nodeepstack_npu.sh"
ZERO3_CONFIG = REPO_ROOT / "scripts/deepspeed_zero3_no_merge.json"
DI_STEP10_SMOKE = REPO_ROOT / "scripts/npu/test/di_smoke_sft_stage_a_lane_intersection_rawlane_local256_550k_zero3_step10_npu.sh"
DI_ORIGINAL_SAVE_STEP10_SMOKE = REPO_ROOT / "scripts/npu/test/di_smoke_sft_stage_a_lane_intersection_rawlane_local256_550k_original_checkpoint_presave_cleanup_step10_npu.sh"
DI_ORIGINAL_SAVE_EVAL_STEP10_SMOKE = REPO_ROOT / "scripts/npu/test/di_smoke_sft_stage_a_lane_intersection_rawlane_local256_550k_original_checkpoint_eval_loss_presave_cleanup_step10_npu.sh"


class RawLaneZero3ShardedRecipeTest(unittest.TestCase):
    def test_recipe_keeps_batch_four_and_never_gathers_during_save(self):
        script = TRAIN_SCRIPT.read_text(encoding="utf-8")
        config = json.loads(ZERO3_CONFIG.read_text(encoding="utf-8"))

        self.assertIn("PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}", script)
        self.assertIn("CHECKPOINT_SAVE_MODE=${CHECKPOINT_SAVE_MODE:-sharded}", script)
        self.assertIn("DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/deepspeed_zero3_no_merge.json}", script)
        self.assertIn('--save_on_each_node "${SAVE_ON_EACH_NODE}"', script)
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

    def test_original_save_smoke_only_adds_presave_npu_cleanup(self):
        formal_script = TRAIN_SCRIPT.read_text(encoding="utf-8")
        smoke_script = DI_ORIGINAL_SAVE_STEP10_SMOKE.read_text(encoding="utf-8")

        self.assertIn("original rank0 consolidated output", formal_script)
        self.assertIn("export MAX_STEPS=20", smoke_script)
        self.assertIn("export SAVE_STEPS=10", smoke_script)
        self.assertIn("export PER_DEVICE_TRAIN_BATCH_SIZE=4", smoke_script)
        self.assertIn("export TARGET_GLOBAL_BATCH_SIZE=128", smoke_script)
        self.assertIn("export CHECKPOINT_SAVE_MODE=original", smoke_script)
        self.assertIn("export DEEPSPEED_CONFIG=scripts/deepspeed_zero3.json", smoke_script)
        self.assertIn("export MLLM_NPU_EMPTY_CACHE_BEFORE_CHECKPOINT=True", smoke_script)
        self.assertIn("export MLLM_SKIP_DISTRIBUTED_FLOS_ON_SAVE=False", smoke_script)
        self.assertIn("zero_shards=False cpu_merge=False", smoke_script)

    def test_eval_loss_smoke_uses_loss_only_eval_before_original_save(self):
        formal_script = TRAIN_SCRIPT.read_text(encoding="utf-8")
        smoke_script = DI_ORIGINAL_SAVE_EVAL_STEP10_SMOKE.read_text(encoding="utf-8")

        self.assertIn("ENABLE_EVAL=${ENABLE_EVAL:-False}", formal_script)
        self.assertIn("SAVE_BEST_EVAL_LOSS=${SAVE_BEST_EVAL_LOSS:-False}", formal_script)
        self.assertIn('--prediction_loss_only True', formal_script)
        self.assertIn('--per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"', formal_script)
        self.assertIn("export ENABLE_EVAL=True", smoke_script)
        self.assertIn("export EVAL_STEPS=10", smoke_script)
        self.assertIn("export EVAL_SAMPLE_LIMIT=${EVAL_SAMPLE_LIMIT:-256}", smoke_script)
        self.assertIn("export PER_DEVICE_EVAL_BATCH_SIZE=1", smoke_script)
        self.assertIn("export SAVE_BEST_EVAL_LOSS=False", smoke_script)
        self.assertIn("export CHECKPOINT_SAVE_MODE=original", smoke_script)
        self.assertIn("export MLLM_NPU_EMPTY_CACHE_BEFORE_CHECKPOINT=True", smoke_script)


if __name__ == "__main__":
    unittest.main()
