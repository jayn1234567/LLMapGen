from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/npu/test/eval_local512_predictions_original_intersection_e2e_npu.sh"
FRESH_OBS_SCRIPT = REPO_ROOT / "scripts/npu/test/eval_local512_predictions_fresh_obs_original_intersection_e2e_npu.sh"
SHARED_EVAL = REPO_ROOT / "scripts/npu/test/eval_rc_e2e_context512_roi256_checkpoint12504_original_pipeline_npu.sh"


def test_intersection_e2e_script_uses_native_local512_geometry_and_one_eval_pass():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "WINDOW_SIZE=${WINDOW_SIZE:-512}" in text
    assert "INTERSECTION_STRIDE=${INTERSECTION_STRIDE:-512}" in text
    assert "ENGINE_EXTRACT_ROOT=${ENGINE_EXTRACT_ROOT:-${RUN_WORK_ROOT}/original_engine_grid512}" in text
    assert "INTER_RESULT_SUBDIR=${INTER_RESULT_SUBDIR:-inter512/tif_512_256}" in text
    assert "QUERY_NAME=${QUERY_NAME:-output_llm_intersection_jn}" in text
    assert "RUN_ALL_EVAL=True" in text
    assert "RUN_LOW_EVAL=False" in text
    assert "RUN_HIGH_EVAL=False" in text
    assert "RUN_FORMAT_STEP=False" in text
    assert "RUN_RULE_STEP=False" in text
    assert "EVAL_SIMPLIFY_PATH=False" in text
    assert "COLLAPSE_INTERSECTION_TYPE_TO_ONE=${COLLAPSE_INTERSECTION_TYPE_TO_ONE:-False}" in text
    assert "--collapse-type-to-one" in text
    assert "PREDICTION_COORD_SCALE=0.512" in text
    assert "ORIGINAL_E2E_LANE_GRID_SIZE=512" in text
    assert "SUPPRESS_PREDICTIONS_WITHOUT_GT_INTERSECTION=${SUPPRESS_PREDICTIONS_WITHOUT_GT_INTERSECTION:-False}" in text
    assert "suppress_rc_e2e_intersections_without_patch_gt.py" in text
    assert 'E2E_USE_RAW_ROOT_DIRECTLY="${E2E_USE_RAW_ROOT_DIRECTLY}"' in text


def test_shared_original_evaluator_accepts_a_dedicated_query_name():
    text = SHARED_EVAL.read_text(encoding="utf-8")
    assert "EVAL_QUERY_NAME=${EVAL_QUERY_NAME:-output_base}" in text
    assert '--query-suffix "${EVAL_QUERY_NAME}"' in text
    assert 'payload["query_name"] = sys.argv[8]' in text


def test_fresh_obs_intersection_entry_reuses_predictions_without_inference():
    text = FRESH_OBS_SCRIPT.read_text(encoding="utf-8")
    assert "e2e_data.zip" in text
    assert "mox.file.copy" in text
    assert "zipfile.is_zipfile" in text
    assert "prepare_rc_e2e_original_run_data.py" in text
    assert "--reset" in text
    assert "eval_local512_predictions_original_intersection_e2e_npu.sh" in text
    assert "SUPPRESS_PREDICTIONS_WITHOUT_GT_INTERSECTION=${SUPPRESS_PREDICTIONS_WITHOUT_GT_INTERSECTION:-True}" in text
    assert "COLLAPSE_INTERSECTION_TYPE_TO_ONE=${COLLAPSE_INTERSECTION_TYPE_TO_ONE:-False}" in text
    assert "EVAL_VIS_FLAG=${EVAL_VIS_FLAG:-False}" in text
    assert "E2E_USE_RAW_ROOT_DIRECTLY=True" in text
    assert "infer_centerline_checkpoint.py" not in text
