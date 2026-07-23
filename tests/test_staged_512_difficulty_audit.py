import unittest

from scripts.tools.audit_staged_512_difficulty import (
    DEFAULT_CAPS,
    classify_metrics,
    resolution_aware_score,
    threshold_bucket_counts,
)


def metrics(**overrides):
    result = {
        "difficulty_score": 0.0,
        "difficulty_score_components": {
            "line_instances": 0.0,
            "output_points": 0.0,
            "intersections": 0.0,
        },
        "centerline_count": 2,
        "point_count": 8,
        "intersection_count": 0,
        "fork_node_count": 0,
        "cycle_count": 0,
        "crossing_count": 0,
        "lane_change_like_count": 0,
        "short_fragment_count": 0,
        "total_turn_degrees": 0.0,
        "max_turn_degrees": 0.0,
    }
    result.update(overrides)
    return result


class Staged512DifficultyAuditTest(unittest.TestCase):
    def test_bucket_caps_separate_four_nonempty_levels(self):
        self.assertEqual(classify_metrics(metrics(), DEFAULT_CAPS), "easy")
        self.assertEqual(
            classify_metrics(metrics(difficulty_score=2.0, centerline_count=8, point_count=40), DEFAULT_CAPS),
            "medium",
        )
        self.assertEqual(
            classify_metrics(metrics(difficulty_score=6.0, centerline_count=13, point_count=70), DEFAULT_CAPS),
            "hard",
        )
        self.assertEqual(
            classify_metrics(metrics(difficulty_score=10.0, centerline_count=20, point_count=110), DEFAULT_CAPS),
            "very_hard",
        )

    def test_empty_is_kept_outside_four_training_buckets(self):
        self.assertEqual(
            classify_metrics(metrics(centerline_count=0, point_count=0, intersection_count=0), DEFAULT_CAPS),
            "empty",
        )

    def test_resolution_aware_score_removes_small_512_count_penalty(self):
        item = metrics(centerline_count=6, point_count=32)
        item["difficulty_score_components"]["line_instances"] = 1.5
        item["difficulty_score_components"]["output_points"] = 0.8
        score = resolution_aware_score(item, free_centerlines=6, free_points=32)
        self.assertEqual(score, 0.0)
        self.assertEqual(item["difficulty_score_components"]["line_instances"], 0.0)
        self.assertEqual(item["difficulty_score_components"]["output_points"], 0.0)

    def test_quantile_count_report_preserves_ties(self):
        counts = threshold_bucket_counts([0.0, 0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
        self.assertEqual(counts, {"easy": 2, "medium": 1, "hard": 1, "very_hard": 1})


if __name__ == "__main__":
    unittest.main()
