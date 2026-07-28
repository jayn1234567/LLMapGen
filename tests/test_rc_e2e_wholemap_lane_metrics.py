from argparse import Namespace

from shapely.geometry import LineString

from scripts.tools.evaluate_rc_e2e_wholemap_lane_metrics import (
    PredSegment,
    lane_match_score,
    scene_metrics,
    stitch_segments,
    summarize,
)


def test_official_lane_score_checks_overlap_and_direction():
    gt = LineString([(0, 0), (20, 0)])
    same = LineString([(0, 0.5), (20, 0.5)])
    reversed_line = LineString([(20, 0.5), (0, 0.5)])

    assert lane_match_score(gt, same, buffer_size=2.5, direction_threshold=10) == 1.0
    assert lane_match_score(gt, reversed_line, buffer_size=2.5, direction_threshold=10) == 0.0


def test_boundary_stitch_preserves_directed_chain():
    segments = [
        PredSegment(LineString([(0, 0), (10, 0)]), "tile:0:0", False, True),
        PredSegment(LineString([(10.2, 0), (20, 0)]), "tile:0:1", True, False),
    ]
    stitched = stitch_segments(segments, distance_threshold=1.0, direction_threshold=20.0)

    assert len(stitched) == 1
    assert list(stitched[0].coords) == [(0.0, 0.0), (10.0, 0.0), (10.2, 0.0), (20.0, 0.0)]


def test_summary_uses_official_instance_and_length_denominators():
    args = Namespace(
        lane_buffer_size=2.5,
        direction_threshold_deg=10.0,
        lane_overlap_threshold=0.8,
    )
    gt = [LineString([(0, 0), (20, 0)])]
    pred = [LineString([(0, 0), (20, 0)]), LineString([(0, 10), (20, 10)])]

    result = summarize(scene_metrics(gt, pred, args))

    assert result["instance_precision"] == 0.5
    assert result["instance_recall"] == 1.0
    assert result["length_precision"] == 0.5
    assert result["length_recall"] == 1.0
