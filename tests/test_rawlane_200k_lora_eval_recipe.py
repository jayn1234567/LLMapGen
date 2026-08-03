import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/npu/test/test_local_rawlane_local256_200k_lora_checkpoint12504_fixed1100_torch240_npu.sh"


class RawLane200kLoraEvalRecipeTest(unittest.TestCase):
    def test_recipe_restores_base_non_lora_and_adapter(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("8cfad7c8fd884a8ea34ad63cd92fbda4", text)
        self.assertIn("adapter_config.json", text)
        self.assertIn("adapter_model.safetensors", text)
        self.assertIn("non_lora_trainables.bin", text)
        self.assertIn("ensure_extracted_llm_from_qwen3vl", text)
        self.assertIn('export QWEN_BASE_MODEL_PATH="${QWEN_EXTRACTED_LLM}"', text)
        self.assertIn("rawlane_local256_200k_fixed1100_e300_m300_h300_vh200_seed42_v1", text)
        self.assertIn("fixed1100_singlepass_torch240_npu.sh", text)


if __name__ == "__main__":
    unittest.main()
