from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "scripts/npu/test/run_and_eval_rc_e2e_clean_local256_550k_checkpoint34376_npu.sh"
)


def test_clean_local256_dinov2_e2e_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "2260c16d83414dea8b663282962413ba" in text
    assert "checkpoint-34376" in text
    assert "activate_mllm_infer_torch240.sh" in text
    assert "E2E_VIEW_MODE=local256" in text
    assert "E2E_TARGET_SIZE=256" in text
    assert "E2E_CONTEXT_SIZE=256" in text
    assert "E2E_PROMPT_PROFILE=local256_550k_v1" in text
    assert "E2E_INPUT_RASTER=inter" in text
    assert "VALIDATE_RASTER_ALIGNMENT=False" in text
    assert '"raw_lane_overlay": False' in text
    assert "PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-32}" in text
    assert "RUN_ALL_EVAL=True" in text
    assert "RUN_LOW_EVAL=True" in text
    assert "RUN_HIGH_EVAL=True" in text
    assert "build_rc_e2e_patch_gt_presence.py" in text
    assert "suppress_e2e_predictions_without_patch_gt.py" in text
    assert "RUN_INTERSECTION_E2E=${RUN_INTERSECTION_E2E:-True}" in text
    assert "eval_local512_predictions_original_intersection_e2e_npu.sh" in text


def test_clean_local256_script_never_selects_rawlane_input() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "E2E_INPUT_RASTER=rawlane" not in text
    assert "E2E_PROMPT_PROFILE=rawlane" not in text
    assert "E2E_INPUT_RASTER=inter" in text
