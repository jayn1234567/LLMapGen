import json
import tempfile
import unittest
from pathlib import Path

from data_process.build_dataset_v2 import (
    empty_candidate_pools,
    parse_ratio_spec,
    select_balanced_candidates,
)
from data_process.difficulty_profiles import LOCAL512_PROFILE, ROI256_PROFILE, classify_metrics
from scripts.tools.build_rc_dataset_v3_balanced_windows import (
    DIFFICULTY_RATIOS,
    expected_counts,
    verify_audit_capacity,
)
from scripts.tools.verify_dataset_v3_eval_sources import verify_dataset_roots


def metric(**updates):
    value = {
        "difficulty_score": 0.0,
        "centerline_count": 1,
        "point_count": 4,
        "intersection_count": 0,
        "fork_node_count": 0,
        "cycle_count": 0,
        "crossing_count": 0,
        "lane_change_like_count": 0,
        "non_common_lane_count": 0,
        "short_fragment_count": 0,
        "total_turn_degrees": 0.0,
        "max_turn_degrees": 0.0,
    }
    value.update(updates)
    return value


class DatasetV3BalancedBuilderTest(unittest.TestCase):
    def test_requested_ratios_and_exact_counts(self):
        self.assertEqual(
            DIFFICULTY_RATIOS,
            "empty=0,very_easy=0.05,easy=0.20,medium=0.30,hard=0.30,very_hard=0.15",
        )
        self.assertEqual(expected_counts(550000), {
            "empty": 0,
            "very_easy": 27500,
            "easy": 110000,
            "medium": 165000,
            "hard": 165000,
            "very_hard": 82500,
        })
        self.assertEqual(expected_counts(200000), {
            "empty": 0,
            "very_easy": 10000,
            "easy": 40000,
            "medium": 60000,
            "hard": 60000,
            "very_hard": 30000,
        })

    def test_profiles_keep_trivial_and_nontrivial_easy_separate(self):
        self.assertEqual(classify_metrics(metric(), LOCAL512_PROFILE), "very_easy")
        self.assertEqual(
            classify_metrics(metric(centerline_count=4, point_count=20, difficulty_score=1.0), LOCAL512_PROFILE),
            "easy",
        )
        self.assertEqual(
            classify_metrics(metric(non_common_lane_count=1), LOCAL512_PROFILE),
            "easy",
        )
        self.assertEqual(classify_metrics(metric(), ROI256_PROFILE), "very_easy")

    def test_very_easy_excess_never_fills_other_bucket_shortage(self):
        pools = empty_candidate_pools()
        ratios = parse_ratio_spec(
            "empty=0,very_easy=0.20,easy=0.20,medium=0.20,hard=0.20,very_hard=0.20"
        )
        for index in range(30):
            pools["very_easy"]["plain"].append(f"ve_{index}")
        for index in range(20):
            pools["easy"]["plain"].append(f"e_{index}")
            pools["hard"]["plain"].append(f"h_{index}")
            pools["very_hard"]["plain"].append(f"vh_{index}")
        counts, report = select_balanced_candidates(
            pools,
            empty_candidate_pools(),
            target_total=40,
            ratios=ratios,
            intersection_target_ratio=0.0,
            seed=7,
        )
        self.assertEqual(sum(counts.values()), 40)
        self.assertEqual(report["final_bucket_counts"]["very_easy"], 8)
        self.assertEqual(report["base_grid"]["difficulty_plan"]["shifted_in"]["very_easy"], 0)

    def test_eval_source_verifier_uses_raw_images_not_patch_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = []
            for name in ("local", "prompt", "context550", "context200"):
                root = Path(tmp) / name
                root.mkdir()
                (root / "split_manifest.json").write_text(json.dumps({
                    "raw_sample_ids_by_split": {
                        "train": [f"train_{name}"],
                        "eval": ["scene_eval_a", "scene_eval_b"],
                        "test": ["scene_test_a"],
                    }
                }), encoding="utf-8")
                phase_a = root / "phase_a"
                phase_a.mkdir()
                for split, ids in (
                    ("eval", ["scene_eval_a", "scene_eval_b"]),
                    ("test", ["scene_test_a"]),
                ):
                    with (phase_a / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
                        for index, raw_sample_id in enumerate(ids):
                            handle.write(json.dumps({
                                "id": f"{split}_{index}",
                                "meta": {"raw_sample_id": raw_sample_id},
                            }) + "\n")
                roots.append(root)
            report = verify_dataset_roots(roots)
            self.assertEqual(report["status"], "passed")
            self.assertTrue(all(
                item["eval"]["exact_match"] and item["test"]["exact_match"]
                for item in report["comparisons"].values()
            ))

    def test_capacity_preflight_rejects_a_bucket_shortage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            counts = expected_counts(200000)
            counts["very_hard"] -= 1
            (root / "summary.json").write_text(json.dumps({
                "difficulty_counts": counts,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "very_hard"):
                verify_audit_capacity(root, 200000)


if __name__ == "__main__":
    unittest.main()
