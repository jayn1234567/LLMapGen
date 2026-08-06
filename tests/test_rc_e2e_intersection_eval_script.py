from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/npu/test/eval_local512_predictions_original_intersection_e2e_npu.sh"
SHARED_EVAL = REPO_ROOT / "scripts/npu/test/eval_rc_e2e_context512_roi256_checkpoint12504_original_pipeline_npu.sh"


def test_intersection_e2e_script_uses_native_local512_geometry_and_one_eval_pass():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "WINDOW_SIZE=${WINDOW_SIZE:-512}" in text
    assert "INTERSECTION_STRIDE=${INTERSECTION_STRIDE:-512}" in text
    assert "INTER_RESULT_SUBDIR=${INTER_RESULT_SUBDIR:-inter512/tif_512_256}" in text
    assert "QUERY_NAME=${QUERY_NAME:-output_llm_intersection_jn}" in text
    assert "RUN_ALL_EVAL=True" in text
    assert "RUN_LOW_EVAL=False" in text
    assert "RUN_HIGH_EVAL=False" in text
    assert "RUN_FORMAT_STEP=False" in text
    assert "RUN_RULE_STEP=False" in text


def test_shared_original_evaluator_accepts_a_dedicated_query_name():
    text = SHARED_EVAL.read_text(encoding="utf-8")
    assert "EVAL_QUERY_NAME=${EVAL_QUERY_NAME:-output_base}" in text
    assert '--query-suffix "${EVAL_QUERY_NAME}"' in text
    assert 'payload["query_name"] = sys.argv[8]' in text
