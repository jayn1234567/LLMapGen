import json
import unittest

from infer_index.line_eval import evaluate_lane_intersection_records
from mllm.coord_utils import convert_payload_text
from mllm.reward.map_schema import parse_map_json


def payload(lane_type="common", intersection_type="t_intersection", intersection_points=None):
    intersection_points = intersection_points or [[20, 20], [40, 20], [40, 40], [20, 40], [20, 20]]
    return json.dumps(
        {
            "lines": [
                {
                    "category": "centerline",
                    "lane_type": lane_type,
                    "start_type": "inside",
                    "end_type": "inside",
                    "points": [[0, 10], [50, 10]],
                },
                {
                    "category": "intersection",
                    "intersection_type": intersection_type,
                    "is_cut": False,
                    "points": intersection_points,
                },
            ]
        }
    )


class MapSchemaSemanticTypeTest(unittest.TestCase):
    def test_parser_preserves_semantic_types(self):
        parsed = parse_map_json(payload(), map_task="lane_intersection", patch_size=256)
        self.assertTrue(parsed.ok, parsed.error)
        self.assertEqual(parsed.items[0]["lane_type"], "common")
        self.assertEqual(parsed.items[1]["intersection_type"], "t_intersection")

    def test_parser_allows_legacy_output_without_semantic_types(self):
        value = json.loads(payload())
        value["lines"][0].pop("lane_type")
        value["lines"][1].pop("intersection_type")
        parsed = parse_map_json(json.dumps(value), map_task="lane_intersection", patch_size=256)
        self.assertTrue(parsed.ok, parsed.error)
        self.assertNotIn("lane_type", parsed.items[0])
        self.assertNotIn("intersection_type", parsed.items[1])


class MapSemanticEvalTest(unittest.TestCase):
    def evaluate(self, prediction, ground_truth=None):
        return evaluate_lane_intersection_records(
            [
                {
                    "id": "sample",
                    "ground_truth_pixel": ground_truth or payload(),
                    "prediction_json_pixel": prediction,
                    "parse_ok": True,
                }
            ],
            meter_per_pixel=0.2,
            buffer_size=1.0,
            match_threshold=0.33,
            intersection_iou_threshold=0.5,
        )

    def test_perfect_geometry_and_types(self):
        result = self.evaluate(payload())
        self.assertEqual(result["intersection"]["instance_f1"], 1.0)
        self.assertEqual(result["intersection"]["micro_area_iou"], 1.0)
        self.assertEqual(result["intersection_type"]["matched_type_accuracy"], 1.0)
        self.assertEqual(result["lane_type"]["matched_type_accuracy"], 1.0)
        self.assertEqual(result["intersection_type"]["per_type"]["t_intersection"]["correct"], 1)

    def test_wrong_types_do_not_change_geometry_score(self):
        result = self.evaluate(payload(lane_type="other", intersection_type="common"))
        self.assertEqual(result["intersection"]["instance_f1"], 1.0)
        self.assertEqual(result["intersection_type"]["matched_type_accuracy"], 0.0)
        self.assertEqual(result["lane_type"]["matched_type_accuracy"], 0.0)
        matrix = result["intersection_type"]["confusion_matrix"]["rows_gt_columns_prediction"]
        self.assertEqual(matrix["t_intersection"]["common"], 1)

    def test_dataset_v2_extended_lane_type_is_scored(self):
        result = self.evaluate(
            payload(lane_type="waiting_area"),
            ground_truth=payload(lane_type="waiting_area"),
        )
        self.assertEqual(result["lane_type"]["matched_type_accuracy"], 1.0)
        self.assertEqual(result["lane_type"]["per_type"]["waiting_area"]["correct"], 1)

    def test_polygon_below_iou_threshold_is_not_an_instance_match(self):
        prediction = payload(
            intersection_points=[[35, 20], [55, 20], [55, 40], [35, 40], [35, 20]]
        )
        result = self.evaluate(prediction)
        self.assertEqual(result["intersection"]["matched_polygon_num"], 0)
        self.assertEqual(result["intersection"]["instance_f1"], 0.0)
        self.assertGreater(result["intersection"]["micro_area_iou"], 0.0)
        self.assertLess(result["intersection"]["micro_area_iou"], 0.5)

    def test_missing_prediction_types_are_reported_as_unknown(self):
        prediction = json.loads(payload())
        prediction["lines"][0].pop("lane_type")
        prediction["lines"][1].pop("intersection_type")
        result = self.evaluate(json.dumps(prediction))
        self.assertEqual(result["intersection_type"]["matched_type_accuracy"], 0.0)
        self.assertEqual(result["intersection_type"]["unknown_prediction_count"], 1)
        self.assertEqual(result["lane_type"]["unknown_prediction_count"], 1)

    def test_norm1000_and_old_normalized_prediction_restore_types(self):
        raw_prediction = payload()
        pixel_prediction = json.loads(
            convert_payload_text(raw_prediction, "norm1000", "pixel", 256, 256, coord_range=1000)
        )
        pixel_prediction["lines"][0].pop("lane_type")
        pixel_prediction["lines"][1].pop("intersection_type")
        result = evaluate_lane_intersection_records(
            [
                {
                    "id": "legacy_summary",
                    "coord_mode": "norm1000",
                    "coord_range": 1000,
                    "patch_width": 256,
                    "patch_height": 256,
                    "ground_truth": payload(),
                    "prediction": raw_prediction,
                    "prediction_json_pixel": json.dumps(pixel_prediction),
                    "parse_ok": True,
                }
            ],
            intersection_iou_threshold=0.5,
        )
        self.assertEqual(result["intersection"]["micro_area_iou"], 1.0)
        self.assertEqual(result["intersection_type"]["matched_type_accuracy"], 1.0)
        self.assertEqual(result["lane_type"]["matched_type_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
