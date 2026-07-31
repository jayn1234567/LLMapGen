import json

from scripts.tools.sanitize_rc_e2e_predictions_for_original_formatter import sanitize_prediction


def _sanitize(points):
    text = json.dumps(
        {
            "lines": [
                {
                    "category": "centerline",
                    "points": points,
                }
            ]
        }
    )
    sanitized, stats = sanitize_prediction(text)
    return json.loads(sanitized)["lines"], stats


def test_drops_centerline_fully_outside_roi():
    lines, stats = _sanitize([[1001, 100], [1200, 200]])

    assert lines == []
    assert stats["dropped_outside_roi_centerlines"] == 1


def test_keeps_centerline_with_outside_endpoints_crossing_roi():
    lines, stats = _sanitize([[-10, 500], [1010, 500]])

    assert len(lines) == 1
    assert lines[0]["points"] == [[0.0, 500.0], [1000.0, 500.0]]
    assert stats["dropped_outside_roi_centerlines"] == 0
    assert stats["clipped_centerlines"] == 1


def test_clips_centerline_with_one_endpoint_outside_roi():
    lines, stats = _sanitize([[500, 500], [1200, 500]])

    assert len(lines) == 1
    assert lines[0]["points"] == [[500.0, 500.0], [1000.0, 500.0]]
    assert stats["clipped_centerlines"] == 1


def test_keeps_centerline_on_inclusive_roi_boundary():
    lines, stats = _sanitize([[1000, 100], [1000, 900]])

    assert len(lines) == 1
    assert lines[0]["points"] == [[1000.0, 100.0], [1000.0, 900.0]]
    assert stats["dropped_outside_roi_centerlines"] == 0


def test_splits_polyline_that_leaves_and_reenters_roi():
    lines, stats = _sanitize([[100, 100], [200, 100], [1200, 100], [1200, 900], [200, 900], [100, 900]])

    assert [line["points"] for line in lines] == [
        [[100.0, 100.0], [200.0, 100.0], [1000.0, 100.0]],
        [[1000.0, 900.0], [200.0, 900.0], [100.0, 900.0]],
    ]
    assert stats["output_centerline_fragments"] == 2
