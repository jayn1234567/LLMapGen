import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / (
    "scripts/npu/train/"
    "train_sft_stage_a_lane_intersection_datasetv2_rawlane_local256_550k_"
    "original_dinov2_caprl4b_nodeepstack_lora_llm_npu.sh"
)
SMOKE_SCRIPT = REPO_ROOT / (
    "scripts/npu/test/"
    "smoke_sft_stage_a_lane_intersection_datasetv2_rawlane_local256_550k_"
    "original_dinov2_caprl4b_nodeepstack_lora_llm_npu.sh"
)


class RawLaneLoraRecipeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = TRAIN_SCRIPT.read_text(encoding="utf-8")

    def test_recipe_uses_requested_data_topology_batch_and_schedule(self):
        expected = (
            "rawlane_local256_550k/rawlane_local256_550k.tar",
            "EXPECTED_TRAIN_SAMPLES=${EXPECTED_TRAIN_SAMPLES:-550000}",
            "EXPECTED_NNODES=${EXPECTED_NNODES:-4}",
            "EXPECTED_NPROC_PER_NODE=${EXPECTED_NPROC_PER_NODE:-8}",
            "TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-128}",
            "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}",
            "GRADIENT_ACCUMULATION_STEPS=$(( TARGET_GLOBAL_BATCH_SIZE / MICRO_BATCH ))",
            "NUM_EPOCHS=${NUM_EPOCHS:-8}",
        )
        for fragment in expected:
            self.assertIn(fragment, self.script)

    def test_qwen_uses_lora_while_projector_and_dinov2_remain_trainable(self):
        expected = (
            "LR=${LR:-2e-4}",
            "MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-2e-4}",
            "MM_VISION_TOWER_LR=${MM_VISION_TOWER_LR:-2e-5}",
            "LORA_ENABLE=${LORA_ENABLE:-True}",
            "LORA_TARGET_SCOPE=${LORA_TARGET_SCOPE:-llm}",
            "LORA_R=${LORA_R:-8}",
            "LORA_ALPHA=${LORA_ALPHA:-16}",
            "LORA_DROPOUT=${LORA_DROPOUT:-0.05}",
            '--lora_enable "${LORA_ENABLE}"',
            '--lora_target_scope "${LORA_TARGET_SCOPE}"',
            "--unfreeze_mm_vision_tower True",
            "--disable_deepstack True",
        )
        for fragment in expected:
            self.assertIn(fragment, self.script)
        self.assertNotIn("--deepspeed", self.script)

    def test_eval_and_presave_npu_cleanup_are_enabled(self):
        expected = (
            "ENABLE_EVAL=${ENABLE_EVAL:-True}",
            "EVAL_STEPS=${EVAL_STEPS:-2000}",
            "EVAL_SAMPLE_LIMIT=${EVAL_SAMPLE_LIMIT:-10000}",
            "PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-1}",
            '--prediction_loss_only True',
            "SAVE_STEPS=${SAVE_STEPS:-1000}",
            "SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-15}",
            "MLLM_NPU_EMPTY_CACHE_BEFORE_CHECKPOINT=${MLLM_NPU_EMPTY_CACHE_BEFORE_CHECKPOINT:-True}",
            "MLLM_SKIP_DISTRIBUTED_FLOS_ON_SAVE=${MLLM_SKIP_DISTRIBUTED_FLOS_ON_SAVE:-False}",
        )
        for fragment in expected:
            self.assertIn(fragment, self.script)
        self.assertNotIn("zero_shards", self.script)
        self.assertNotIn("CHECKPOINT_SAVE_MODE=${CHECKPOINT_SAVE_MODE", self.script)

    def test_ascend_smoke_exercises_eval_and_checkpoint_at_step_ten(self):
        smoke = SMOKE_SCRIPT.read_text(encoding="utf-8")
        expected = (
            "NNODES=1",
            "NPROC_PER_NODE=8",
            "EXPECTED_NNODES=1",
            "EXPECTED_NPROC_PER_NODE=8",
            "MAX_STEPS=20",
            "SAVE_STEPS=10",
            "EVAL_STEPS=10",
            "EVAL_SAMPLE_LIMIT=256",
            "PER_DEVICE_TRAIN_BATCH_SIZE=4",
            "TARGET_GLOBAL_BATCH_SIZE=128",
            "INSTALL_DEPS=False",
            "ENABLE_MOXING_UPGRADE=False",
            "MLLM_NPU_EMPTY_CACHE_BEFORE_CHECKPOINT=True",
            "Released unused NPU cache before checkpoint save.",
            "non_lora_trainables.bin",
            "adapter_config.json",
        )
        for fragment in expected:
            self.assertIn(fragment, smoke)


if __name__ == "__main__":
    unittest.main()
