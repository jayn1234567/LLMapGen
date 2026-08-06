from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/npu/test/run_and_eval_rc_e2e_local512_checkpoint_npu.sh"


def test_generic_local512_entry_runs_only_whole_map_metrics() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "CHECKPOINT_OBS_PATH=${1:-${CHECKPOINT_OBS_PATH:-}}" in text
    assert "E2E_VIEW_MODE=local512" in text
    assert "E2E_TARGET_SIZE=512" in text
    assert "build_rc_e2e_patch_gt_presence.py" in text
    assert "suppress_e2e_predictions_without_patch_gt.py" in text
    assert "ORIGINAL_E2E_LANE_GRID_SIZE=512" in text
    assert "PREDICTION_COORD_SCALE=0.512" in text
    assert "RUN_ALL_EVAL=True" in text
    assert "RUN_LOW_EVAL=True" in text
    assert "RUN_HIGH_EVAL=True" in text
    assert "eval_rc_e2e_context512_roi256_checkpoint12504_patch_metrics.sh" not in text
    assert "gt_empty" not in text.lower()
    assert "oracle" not in text.lower()


def test_generic_local512_entry_uses_original_metric_directory_names() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "${OUTPUT_ROOT}/eval_result_all" in text
    assert "${OUTPUT_ROOT}/eval_result_low" in text
    assert "${OUTPUT_ROOT}/eval_result_high" in text
