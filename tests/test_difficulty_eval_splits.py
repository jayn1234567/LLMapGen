import argparse
import json
import unittest

from scripts.tools.tag_hard_map_samples import sample_metrics


def metric_args():
    return argparse.Namespace(
        coord_mode="auto",
        coord_range=1000.0,
        junction_tol=36.0,
        intersection_tol=16.0,
        dense_line_threshold=8,
        dense_point_threshold=34,
        long_total_length_threshold=3600.0,
        many_cut_threshold=6,
    )


def make_record(points, coord_mode, patch_size=256):
    lines = [
        {
            "category": "centerline",
            "start_type": "cut",
            "end_type": "cut",
            "points": item,
        }
        for item in points
    ]
    return {
        "id": f"sample-{coord_mode}",
        "image": "images/test/sample.png",
        "meta": {
            "coord_mode": coord_mode,
            "patch_width": patch_size,
            "patch_height": patch_size,
        },
        "conversations": [
            {"from": "human", "value": "<image>\nConstruct the road map."},
            {"from": "gpt", "value": json.dumps({"lines": lines})},
        ],
    }


class DifficultyMetricNormalizationTest(unittest.TestCase):
    def test_pixel_and_norm1000_geometry_have_same_difficulty(self):
        pixel_lines = [
            [[0, index * 30], [255, index * 30]]
            for index in range(8)
        ]
        norm_lines = [
            [[0, round(index * 30 / 255 * 1000)], [1000, round(index * 30 / 255 * 1000)]]
            for index in range(8)
        ]

        pixel = sample_metrics(make_record(pixel_lines, "pixel"), None, metric_args())
        normalized = sample_metrics(make_record(norm_lines, "norm1000"), None, metric_args())

        self.assertEqual(pixel["difficulty"], normalized["difficulty"])
        self.assertEqual(pixel["tags"], normalized["tags"])
        self.assertAlmostEqual(
            pixel["total_centerline_length"],
            normalized["total_centerline_length"],
            delta=2.0,
        )
        self.assertEqual(pixel["coord_mode"], "pixel")
        self.assertEqual(normalized["coord_mode"], "norm1000")


if __name__ == "__main__":
    unittest.main()
