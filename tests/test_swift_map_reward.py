import json
import unittest

from mllm.reward.swift_map_reward import score_one


def _line(points):
    return {
        "category": "centerline",
        "start_type": "cut",
        "end_type": "cut",
        "points": points,
    }


def _payload(lines):
    return json.dumps({"lines": lines}, separators=(",", ":"))


class SwiftMapRewardTest(unittest.TestCase):
    def test_perfect_prediction_is_near_one(self):
        target = _payload([_line([[0, 0], [1000, 1000]])])
        result = score_one(target, target)
        self.assertTrue(result["parse_ok"])
        self.assertGreater(result["reward"], 0.99)
        self.assertAlmostEqual(result["format_reward"], 1.0)

    def test_missing_or_extra_lines_are_penalized_by_line_f1(self):
        target = _payload([
            _line([[0, 0], [1000, 1000]]),
            _line([[0, 1000], [1000, 0]]),
        ])
        missing = score_one(_payload([json.loads(target)["lines"][0]]), target)
        extra = score_one(
            _payload(json.loads(target)["lines"] + [_line([[500, 0], [500, 1000]])]),
            target,
        )
        self.assertLess(missing["line_f1"], 0.9)
        self.assertLess(extra["line_f1"], 0.9)

    def test_bad_json_gets_invalid_reward(self):
        target = _payload([_line([[0, 0], [1000, 1000]])])
        result = score_one("not json", target)
        self.assertFalse(result["parse_ok"])
        self.assertEqual(result["reward"], -1.0)

    def test_json_with_prose_loses_only_format_component(self):
        target = _payload([_line([[0, 0], [1000, 1000]])])
        result = score_one("answer: " + target, target)
        self.assertTrue(result["parse_ok"])
        self.assertEqual(result["format_reward"], 0.0)
        self.assertGreater(result["line_f1"], 0.99)


if __name__ == "__main__":
    unittest.main()
