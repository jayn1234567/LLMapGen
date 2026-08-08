from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "scripts/npu/test/eval_rawlane_local256_550k_checkpoint34376_gt_empty_fresh_obs_original_e2e_npu.sh"
)


def test_rawlane_local256_550k_direct_checkpoint_e2e_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "5e26ea0190634699828ff4b28df4c608" in text
    assert "CHECKPOINT_NAME=${CHECKPOINT_NAME:-checkpoint-34376}" in text
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
