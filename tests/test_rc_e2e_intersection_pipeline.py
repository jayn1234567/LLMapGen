from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from scripts.tools.build_rc_e2e_intersection_geojson import build_all
from scripts.tools.format_rc_e2e_intersection_predictions import format_predictions


def write_source_tif(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2048,
        height=2048,
        count=3,
        dtype="uint8",
        crs="EPSG:32650",
        transform=from_origin(500000.0, 3000000.0, 0.2, 0.2),
    ) as dataset:
        dataset.write(np.zeros((3, 2048, 2048), dtype=np.uint8))


def write_prediction(path: Path, *, x0: int = 1024) -> None:
    payload = {
        "record_id": "scene_001_0_1_2",
        "image": "images/scene_001/0_inter/1_2.png",
        "row": 1,
        "col": 2,
        "x0": x0,
        "y0": 512,
        "meta": {"scene_id": "scene_001", "tif_prefix": "0"},
        "prediction_json": json.dumps(
            {
                "lines": [
                    {
                        "category": "intersection",
                        "intersection_type": "common",
                        "points": [[0, 0], [1000, 0], [1000, 1000], [0, 1000], [0, 0]],
                    },
                    {"category": "centerline", "points": [[0, 0], [1000, 1000]]},
                ]
            }
        ),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_format_and_merge_local512_intersections(tmp_path):
    e2e_root = tmp_path / "e2e_data"
    scene = e2e_root / "scene_001"
    write_source_tif(
        scene / "rc_one_patch_release" / "center_line_v2" / "inter_patch_tif" / "0_inter.tif"
    )
    gt_dir = scene / "gt"
    gt_dir.mkdir()
    (gt_dir / "Lane.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature"}]}),
        encoding="utf-8",
    )

    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    write_prediction(prediction_dir / "rank0_sample.json")

    format_report = format_predictions(
        prediction_dir,
        e2e_root,
        tmp_path / "format_report.json",
        window_size=512,
        stride=512,
        coord_range=1000,
        result_subdir=Path("inter512/tif_512_256"),
        reset=True,
        strict=True,
    )

    patch_result = (
        scene
        / "rc_one_patch_release"
        / "center_line_v2"
        / "inter512"
        / "tif_512_256"
        / "0_tif_res"
        / "1_2.json"
    )
    payload = json.loads(patch_result.read_text(encoding="utf-8"))
    assert format_report["complete"] is True
    assert payload["intersection"][0]["label"] == "1_1"
    assert payload["intersection"][0]["coords"][1] == [512.0, 0.0]

    report = build_all(
        e2e_root,
        tmp_path / "geojson_report.json",
        result_subdir=Path("inter512/tif_512_256"),
        query_name="output_llm_intersection_jn",
        stride=512,
        merge_buffer_meters=0.5,
        expected_scenes=1,
        reset_query=True,
    )

    intersection_path = scene / "output_llm_intersection_jn" / "Intersection.geojson"
    intersection = json.loads(intersection_path.read_text(encoding="utf-8"))
    lane = json.loads((intersection_path.parent / "Lane.geojson").read_text(encoding="utf-8"))
    assert report["intersection_features"] == 1
    assert len(intersection["features"]) == 1
    assert intersection["features"][0]["properties"]["IntersectionType"] == 1
    assert intersection["features"][0]["properties"]["IntersectionSubType"] == 1
    assert lane["features"] == []


def test_formatter_rejects_stride_mismatch_in_strict_mode(tmp_path):
    e2e_root = tmp_path / "e2e_data"
    (e2e_root / "scene_001" / "rc_one_patch_release").mkdir(parents=True)
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    write_prediction(prediction_dir / "bad.json", x0=512)

    with pytest.raises(RuntimeError, match="Failed to format"):
        format_predictions(
            prediction_dir,
            e2e_root,
            tmp_path / "format_report.json",
            window_size=512,
            stride=512,
            coord_range=1000,
            result_subdir=Path("inter512/tif_512_256"),
            reset=True,
            strict=True,
        )

    report = json.loads((tmp_path / "format_report.json").read_text(encoding="utf-8"))
    assert "expected col*stride=1024" in report["errors"][0]["error"]
