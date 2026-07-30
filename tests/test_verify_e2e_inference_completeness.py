import json

from scripts.tools.verify_e2e_inference_completeness import verify


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_verify_accepts_one_to_one_outputs(tmp_path):
    infer_jsonl = tmp_path / "infer.jsonl"
    infer_jsonl.write_text('{"id":"a"}\n{"id":"b"}\n', encoding="utf-8")
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    _write_json(prediction_dir / "rank0_a.json", {"record_id": "a", "prediction_json": "{}"})
    _write_json(prediction_dir / "rank1_b.json", {"record_id": "b", "prediction_json": "{}"})

    summary = verify(infer_jsonl, prediction_dir)

    assert summary["complete"] is True
    assert summary["expected_records"] == 2
    assert summary["prediction_files"] == 2


def test_verify_rejects_duplicate_and_missing_outputs(tmp_path):
    infer_jsonl = tmp_path / "infer.jsonl"
    infer_jsonl.write_text('{"id":"a"}\n{"id":"b"}\n', encoding="utf-8")
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    _write_json(prediction_dir / "rank0_a.json", {"record_id": "a", "prediction_json": "{}"})
    _write_json(prediction_dir / "rank1_a.json", {"record_id": "a", "prediction_json": "{}"})

    summary = verify(infer_jsonl, prediction_dir)

    assert summary["complete"] is False
    assert summary["duplicate_prediction_ids"] == ["a"]
    assert summary["missing_prediction_ids"] == ["b"]
