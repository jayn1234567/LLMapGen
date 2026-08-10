from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "scripts/npu/test/eval_rawlane_local256_550k_checkpoint34376_gt_empty_fresh_obs_original_e2e_npu.sh"
)
GENERIC_INFER_SCRIPT = (
    REPO_ROOT
    / "scripts/npu/test/run_rc_e2e_context512_roi256_checkpoint12504_npu.sh"
)


def test_rawlane_local256_550k_direct_checkpoint_e2e_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "5e26ea0190634699828ff4b28df4c608" in text
    assert "CHECKPOINT_NAME=${CHECKPOINT_NAME:-checkpoint-34376}" in text
    assert "CHECKPOINT_EXPECTED_KIND=${CHECKPOINT_EXPECTED_KIND:-lora}" in text
    assert 'CHECKPOINT_EXPECTED_KIND="${CHECKPOINT_EXPECTED_KIND}"' in text
    assert "non_lora_trainables.bin" in text
    assert "CapRL-Qwen3VL-4B_llm_extracted" in text
    assert 'export QWEN_BASE_MODEL_PATH="${QWEN_EXTRACTED_LLM}"' in text
    assert "ensure_extracted_llm_from_qwen3vl" in text
    assert "facebook_dinov2-large" in text
    assert 'VISION_TOWER="${VISION_TOWER}"' in text
    assert "E2E_VIEW_MODE=local256" in text
    assert "E2E_TARGET_SIZE=256" in text
    assert "E2E_PROMPT_PROFILE=rawlane_local256_550k_v1" in text
    assert "E2E_INPUT_RASTER=rawlane" in text
    assert "BLACK_RATIO_THRESHOLD=1.0" in text
    assert "MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}" in text
    assert "GT-empty patches" in text
    assert "RUN_ALL_EVAL=True" in text
    assert "RUN_LOW_EVAL=True" in text
    assert "RUN_HIGH_EVAL=True" in text


def test_generic_e2e_inference_can_require_lora_checkpoint_layout() -> None:
    text = GENERIC_INFER_SCRIPT.read_text(encoding="utf-8")

    assert "CHECKPOINT_EXPECTED_KIND=${CHECKPOINT_EXPECTED_KIND:-auto}" in text
    assert '--expected-kind "${CHECKPOINT_EXPECTED_KIND}"' in text
