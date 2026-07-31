import json
from pathlib import Path

from scripts.tools.ensure_rc_e2e_scene_outputs import audit_and_fill


def write_collection(path: Path, features: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"type": "FeatureCollection", "name": "fixture", "features": features}),
        encoding="utf-8",
    )


def make_scene(root: Path, name: str, *, prediction: bool) -> Path:
    scene = root / name
    (scene / "rc_one_patch_release").mkdir(parents=True)
    write_collection(scene / "gt" / "Lane.geojson", [{"type": "Feature"}])
    write_collection(scene / "gt" / "Intersection.geojson", [{"type": "Feature"}])
    if prediction:
        write_collection(scene / "output_base" / "Lane.geojson", [])
        write_collection(scene / "output_base" / "Intersection.geojson", [])
    return scene


def test_missing_prediction_becomes_empty_geojson(tmp_path):
    missing = make_scene(tmp_path, "100", prediction=False)
    make_scene(tmp_path, "200", prediction=True)
    report_path = tmp_path / "report.json"

    report = audit_and_fill(
        tmp_path,
        report_path,
        expected_scenes=2,
        baseline_suffix="gt",
        query_suffix="output_base",
        fill_missing_predictions=True,
    )

    assert report["missing_prediction_before"] == ["100"]
    assert report["created_empty_predictions"] == ["100"]
    assert report["evaluable_scenes_after"] == 2
    assert report["complete"] is True
    lane = json.loads((missing / "output_base" / "Lane.geojson").read_text(encoding="utf-8"))
    intersection = json.loads(
        (missing / "output_base" / "Intersection.geojson").read_text(encoding="utf-8")
    )
    assert lane["features"] == []
    assert intersection["features"] == []
    assert lane["name"] == "fixture"
