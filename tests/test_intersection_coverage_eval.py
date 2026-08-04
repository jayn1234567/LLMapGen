import json
import unittest

from infer_index.intersection_coverage_eval import evaluate_intersection_coverage_records


def polygon(points, intersection_type="common"):
    return {
        "category": "intersection",
        "intersection_type": intersection_type,
        "points": points,
    }


def payload(*items):
    return json.dumps({"lines": list(items)})


def record(gt, pred, **extra):
    return {
        "id": "sample",
        "ground_truth_pixel": payload(*gt),
        "prediction_json_pixel": payload(*pred),
        "parse_ok": True,
        **extra,
    }


SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]


class IntersectionCoverageEvalTest(unittest.TestCase):
    def test_perfect_prediction(self):
        result = evaluate_intersection_coverage_records(
            [record([polygon(SQUARE)], [polygon(SQUARE)])]
        )["intersection"]
        self.assertEqual(result["instance_precision"], 1.0)
        self.assertEqual(result["instance_recall"], 1.0)
        self.assertEqual(result["area_precision"], 1.0)
        self.assertEqual(result["area_recall"], 1.0)

    def test_large_prediction_is_recalled_but_not_correct(self):
        large = [[-10, -10], [20, -10], [20, 20], [-10, 20], [-10, -10]]
        result = evaluate_intersection_coverage_records(
            [record([polygon(SQUARE)], [polygon(large)])]
        )["intersection"]
        self.assertEqual(result["recalled_num"], 1)
        self.assertEqual(result["correct_num"], 0)
        self.assertEqual(result["instance_precision"], 0.0)
        self.assertEqual(result["instance_recall"], 1.0)
        self.assertAlmostEqual(result["area_precision"], 100 / 900, places=4)

    def test_small_prediction_is_correct_but_not_recalled(self):
        small = [[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]]
        result = evaluate_intersection_coverage_records(
            [record([polygon(SQUARE)], [polygon(small)])]
        )["intersection"]
        self.assertEqual(result["recalled_num"], 0)
        self.assertEqual(result["correct_num"], 1)
        self.assertEqual(result["instance_precision"], 1.0)
        self.assertEqual(result["instance_recall"], 0.0)

    def test_threshold_is_strictly_greater_than_half(self):
        half = [[0, 0], [5, 0], [5, 10], [0, 10], [0, 0]]
        result = evaluate_intersection_coverage_records(
            [record([polygon(SQUARE)], [polygon(half)])]
        )["intersection"]
        self.assertEqual(result["recalled_num"], 0)
        self.assertEqual(result["correct_num"], 1)

    def test_t_intersection_subset_and_empty_denominator_policy(self):
        result = evaluate_intersection_coverage_records(
            [record([polygon(SQUARE, "t_intersection")], [])]
        )
        t_metric = result["t_intersection"]
        self.assertEqual(t_metric["gt_num"], 1)
        self.assertEqual(t_metric["pred_num"], 0)
        self.assertEqual(t_metric["instance_precision"], 1.0)
        self.assertEqual(t_metric["instance_recall"], 0.0)

    def test_union_prevents_duplicate_overlap_area(self):
        duplicate_prediction = [polygon(SQUARE), polygon(SQUARE)]
        result = evaluate_intersection_coverage_records(
            [record([polygon(SQUARE)], duplicate_prediction)]
        )["intersection"]
        self.assertEqual(result["matched_area"], 100.0)
        self.assertEqual(result["pred_total_area"], 100.0)
        self.assertEqual(result["correct_num"], 2)

    def test_malformed_prediction_counts_as_empty_and_invalid_format(self):
        sample = record([polygon(SQUARE)], [])
        sample["prediction_json_pixel"] = "not-json"
        result = evaluate_intersection_coverage_records([sample])["intersection"]
        self.assertEqual(result["pred_num"], 0)
        self.assertEqual(result["valid_string_format"], 0)
        self.assertEqual(result["instance_recall"], 0.0)


if __name__ == "__main__":
    unittest.main()

