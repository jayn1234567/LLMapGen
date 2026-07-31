from __future__ import annotations

import json

from scripts.tools.suppress_e2e_predictions_without_patch_gt import suppress_predictions


def test_suppresses_predictions_only_for_gt_empty_patches(tmp_path):
    eval_jsonl = tmp_path / "eval.jsonl"
    records = [
        {
            "record_id": "positive",
            "ground_truth_pixel": '{"lines":[{"category":"centerline","points":[[0,0],[1,1]]}]}',
        },
        {"record_id": "empty", "ground_truth_pixel": '{"lines":[]}'},
    ]
    eval_jsonl.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    for record_id in ("positive", "empty"):
        (prediction_dir / f"{record_id}.json").write_text(
            json.dumps(
                {
                    "record_id": record_id,
                    "image": f"images/1234567890123/0_inter/{record_id}.png",
                    "prediction_json": (
                        '{"lines":[{"category":"centerline","points":[[0,0],[1,1]]}]}'
                    ),
                }
            ),
            encoding="utf-8",
        )

    output_dir = tmp_path / "oracle"
    report = suppress_predictions(
        eval_jsonl,
        prediction_dir,
        output_dir,
        tmp_path / "report.json",
        reset=True,
        strict=True,
    )

    positive = json.loads((output_dir / "positive.json").read_text(encoding="utf-8"))
    empty = json.loads((output_dir / "empty.json").read_text(encoding="utf-8"))
    assert json.loads(positive["prediction_json"])["lines"]
    assert json.loads(empty["prediction_json"])["lines"] == []
    assert empty["gt_oracle_patch_suppressed"] is True
    assert report["kept_gt_positive_patches"] == 1
    assert report["suppressed_gt_empty_patches"] == 1
    assert report["suppressed_centerlines"] == 1
