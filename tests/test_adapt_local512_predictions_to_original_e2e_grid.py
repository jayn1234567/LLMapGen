import json
from pathlib import Path

from scripts.tools.adapt_local512_predictions_to_original_e2e_grid import adapt_directory


def write_prediction(path: Path) -> None:
    payload = {
        "record_id": "scene_0_1_2",
        "image": "/tmp/images/1234567890123/0_inter/1_2.png",
        "row": 1,
        "col": 2,
        "meta": {
            "scene_id": "1234567890123",
            "tif_stem": "0_inter",
            "tif_prefix": "0",
            "row": 1,
            "col": 2,
            "patch_size": 512,
            "coord_mode": "norm1000",
            "coord_range": 1000,
        },
        "parse_ok": True,
        "prediction_json": json.dumps(
            {
                "lines": [
                    {
                        "category": "centerline",
                        "lane_type": "common",
                        "points": [[0, 250], [1000, 250]],
                    },
                    {
                        "category": "intersection",
                        "points": [[0, 0], [10, 0], [0, 10], [0, 0]],
                    },
                ]
            }
        ),
        "prediction_json_pixel": json.dumps(
            {
                "lines": [
                    {
                        "category": "centerline",
                        "lane_type": "common",
                        "points": [[0, 128], [512, 128]],
                    },
                    {
                        "category": "intersection",
                        "points": [[0, 0], [5, 0], [0, 5], [0, 0]],
                    },
                ]
            }
        ),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_by_id(output_dir: Path) -> dict[str, dict]:
    result = {}
    for path in output_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[payload["record_id"]] = payload
    return result


def test_adapt_local512_line_to_original_256_grid(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    write_prediction(input_dir / "source.json")

    report = adapt_directory(
        input_dir,
        output_dir,
        tmp_path / "report.json",
        source_patch_size=512,
        engine_patch_size=256,
        coord_range=1000,
        reset=True,
        strict=True,
    )

    assert report["complete"] is True
    assert report["output_prediction_records"] == 4
    records = load_by_id(output_dir)
    assert set(records) == {
        "1234567890123_0_2_4",
        "1234567890123_0_2_5",
        "1234567890123_0_3_4",
        "1234567890123_0_3_5",
    }

    left = json.loads(records["1234567890123_0_2_4"]["prediction_json"])["lines"]
    right = json.loads(records["1234567890123_0_2_5"]["prediction_json"])["lines"]
    expected = [{"category": "centerline", "lane_type": "common", "points": [[0, 500], [1000, 500]]}]
    assert left == expected
    assert right == expected
    assert json.loads(records["1234567890123_0_3_4"]["prediction_json"])["lines"] == []
    assert json.loads(records["1234567890123_0_3_5"]["prediction_json"])["lines"] == []
    assert records["1234567890123_0_2_5"]["image"].endswith("/0_inter/2_5.png")
    assert records["1234567890123_0_2_5"]["x0"] == 5 * 256
    assert records["1234567890123_0_2_5"]["y0"] == 2 * 256


def test_adapt_uses_norm_payload_when_pixel_payload_is_absent(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    write_prediction(input_dir / "source.json")
    payload = json.loads((input_dir / "source.json").read_text(encoding="utf-8"))
    payload.pop("prediction_json_pixel")
    (input_dir / "source.json").write_text(json.dumps(payload), encoding="utf-8")

    report = adapt_directory(
        input_dir,
        output_dir,
        tmp_path / "report.json",
        source_patch_size=512,
        engine_patch_size=256,
        coord_range=1000,
        reset=True,
        strict=True,
    )

    assert report["stats"]["source_space:norm"] == 1
    records = load_by_id(output_dir)
    left = json.loads(records["1234567890123_0_2_4"]["prediction_json"])["lines"]
    assert left[0]["points"] == [[0, 500], [1000, 500]]


def test_parse_failure_becomes_four_empty_engine_cells(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    write_prediction(input_dir / "source.json")
    payload = json.loads((input_dir / "source.json").read_text(encoding="utf-8"))
    payload["parse_ok"] = False
    payload["prediction_json_pixel"] = ""
    payload["prediction_json"] = "truncated output"
    (input_dir / "source.json").write_text(json.dumps(payload), encoding="utf-8")

    report = adapt_directory(
        input_dir,
        output_dir,
        tmp_path / "report.json",
        source_patch_size=512,
        engine_patch_size=256,
        coord_range=1000,
        reset=True,
        strict=True,
    )

    assert report["stats"]["source_space:parse_failure"] == 1
    records = load_by_id(output_dir)
    assert len(records) == 4
    assert all(json.loads(record["prediction_json"])["lines"] == [] for record in records.values())
    assert all(record["parse_ok"] is False for record in records.values())


def test_local512_recipe_adapts_before_original_engine() -> None:
    root = Path(__file__).resolve().parents[1]
    recipe = (
        root
        / "scripts/npu/test/eval_local512_550k_checkpoint34376_gt_empty_fresh_obs_original_e2e_npu.sh"
    ).read_text(encoding="utf-8")
    original_entry = (
        root
        / "scripts/npu/test/eval_rc_e2e_context512_roi256_checkpoint12504_original_pipeline_npu.sh"
    ).read_text(encoding="utf-8")

    assert "adapt_local512_predictions_to_original_e2e_grid.py" in recipe
    assert 'SOURCE_PREDICTION_DIR="${ORIGINAL_GRID_PREDICTION_DIR}"' in recipe
    assert "PATCH_SIZE=256" in recipe
    assert "PREDICTION_COORD_SCALE=0.256" in recipe
    assert "PREDICTION_COORD_SCALE=0.512" not in recipe
    assert "LaneNNParser is fixed to a 256x256 grid" in original_entry
