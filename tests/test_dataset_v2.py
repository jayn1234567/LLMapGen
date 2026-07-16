import random
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from data_process import state_update_dataset_common as dataset_common
from data_process.build_dataset_v2 import (
    DEFAULT_DIFFICULTY_RATIOS,
    DIFFICULTY_ORDER,
    allocate_global_intersection_quotas,
    annotate_translation_grid,
    classify_row,
    empty_candidate_pools,
    select_balanced_candidates,
)
from data_process.state_update_dataset_common import (
    build_sft_record,
    centered_target_roi,
    extract_centered_context,
    lane_type_name,
    normalize_lane_type_code,
)
from scripts.tools.build_rc_dataset_v2_from_obs import (
    DEFAULT_OUTPUT_OBS_ROOT,
    ObsutilBackend,
)


class DatasetV2ContextTest(unittest.TestCase):
    def test_parallel_archive_extraction_writes_and_reuses_completion_markers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archives = []
            for index in range(3):
                source = root / f"payload_{index}.txt"
                source.write_text(f"payload {index}", encoding="utf-8")
                archive = root / f"sample_{index}.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    tar.add(source, arcname=source.name)
                source.unlink()
                archives.append(archive)

            dataset_common.extract_archives(root, delete_archive=False, workers=3)

            for index, archive in enumerate(archives):
                target = root / f"sample_{index}"
                self.assertEqual(
                    (target / f"payload_{index}.txt").read_text(encoding="utf-8"),
                    f"payload {index}",
                )
                self.assertTrue(dataset_common.archive_extract_is_complete(archive))

            adopted_archive = archives[0]
            dataset_common.archive_extract_marker_path(adopted_archive).unlink()
            with patch.object(
                dataset_common.tarfile.TarFile,
                "extractall",
                side_effect=AssertionError("a complete legacy extraction must be adopted"),
            ):
                dataset_common.safe_extract_tar_gz(adopted_archive, delete_archive=False)
            self.assertTrue(dataset_common.archive_extract_is_complete(adopted_archive))

            with patch.object(
                dataset_common.tarfile,
                "open",
                side_effect=AssertionError("completed archives must not be opened again"),
            ):
                dataset_common.extract_archives(root, delete_archive=False, workers=3)

    def test_u_turn_reference_lane_type_is_excluded(self):
        self.assertIsNone(lane_type_name(3))
        self.assertIsNone(lane_type_name("3"))
        self.assertIsNone(lane_type_name("3.0"))

    def test_lane_type_metadata_keeps_right_turn_and_common_lines(self):
        self.assertEqual(lane_type_name(2), "right_turn")
        self.assertEqual(lane_type_name("2.0"), "right_turn")
        self.assertEqual(lane_type_name(1), "common")
        self.assertEqual(lane_type_name(None), "common")
        self.assertEqual(normalize_lane_type_code(" 2.0 "), 2)

    def test_line_loader_drops_u_turn_reference_geometry(self):
        class FakeLineString:
            def __init__(self, offset):
                self.coords = [(offset, 0), (offset + 1, 1)]
                self.is_empty = False

        class FakeMultiLineString:
            pass

        class FakeGeoDataFrame:
            def to_crs(self, crs):
                del crs
                return self

            def iterrows(self):
                rows = [
                    SimpleNamespace(LaneType=3, geometry=FakeLineString(0)),
                    SimpleNamespace(LaneType=2, geometry=FakeLineString(10)),
                    SimpleNamespace(LaneType=1, geometry=FakeLineString(20)),
                ]
                return iter(enumerate(rows))

        class FakeGeoPandas:
            @staticmethod
            def read_file(path):
                del path
                return FakeGeoDataFrame()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lane.geojson"
            path.touch()
            with (
                patch.object(dataset_common, "gpd", FakeGeoPandas),
                patch.object(dataset_common, "LineString", FakeLineString),
                patch.object(dataset_common, "MultiLineString", FakeMultiLineString),
            ):
                lines = dataset_common.load_line_geometries(path, "EPSG:4326", None, 0.0)

        self.assertEqual(len(lines), 2)
        self.assertEqual(
            [line["_source_lane_type"] for line in lines],
            ["right_turn", "common"],
        )
        self.assertEqual([line["_source_lane_type_code"] for line in lines], [2, 1])

    def test_centered_context_uses_black_padding(self):
        image = np.arange(1, 17, dtype=np.uint8).reshape(1, 4, 4)
        context = extract_centered_context(image, 0, 0, 2, 4)
        self.assertEqual(context.shape, (1, 4, 4))
        np.testing.assert_array_equal(context[:, 1:3, 1:3], image[:, 0:2, 0:2])
        self.assertTrue(np.all(context[:, 0, :] == 0))
        self.assertTrue(np.all(context[:, :, 0] == 0))

    def test_context_prompt_and_metadata_use_target_roi_coordinates(self):
        row = {
            "id": "sample_r000_c000",
            "image": "images/train/sample/sample_r000_c000.png",
            "incoming_traces": [],
            "incoming_intersections": [],
            "target_lines": [
                {
                    "category": "centerline",
                    "start_type": "cut",
                    "end_type": "inside",
                    "points": [[0, 0], [255, 255]],
                }
            ],
            "meta": {"x0": 0, "y0": 0},
        }
        record = build_sft_record(
            row,
            256,
            True,
            "a",
            context_size=512,
            view_mode="context512_roi256",
        )
        prompt = record["conversations"][0]["value"]
        self.assertIn("central 256x256 target ROI [128,128,384,384)", prompt)
        self.assertIn("relative to the target ROI, not the full context image", prompt)
        self.assertEqual(record["meta"]["target_roi_in_image"], [128, 128, 384, 384])
        self.assertEqual(centered_target_roi(256, 512), [128, 128, 384, 384])


class DatasetV2BalancingTest(unittest.TestCase):
    def test_production_classifier_does_not_score_cut_endpoints(self):
        row = {
            "id": "simple_cut_roads",
            "image": "images/train/simple_cut_roads.png",
            "target_lines": [
                {
                    "category": "centerline",
                    "start_type": "cut",
                    "end_type": "cut",
                    "points": [[0, y], [255, y]],
                }
                for y in (50, 128, 205)
            ],
        }

        metrics = classify_row(row, patch_size=256, coord_range=1000)

        self.assertEqual(metrics["stratum"], "easy")
        self.assertEqual(metrics["cut_endpoint_count"], 6)
        self.assertFalse(metrics["cut_affects_difficulty"])

    def test_production_classifier_requires_explicitly_simple_easy_geometry(self):
        def classify_parallel_lines(count):
            return classify_row(
                {
                    "id": f"parallel_{count}",
                    "image": f"images/train/parallel_{count}.png",
                    "target_lines": [
                        {
                            "category": "centerline",
                            "start_type": "cut",
                            "end_type": "cut",
                            "points": [
                                [0, round((index + 1) * 255 / (count + 1))],
                                [255, round((index + 1) * 255 / (count + 1))],
                            ],
                        }
                        for index in range(count)
                    ],
                },
                patch_size=256,
                coord_range=1000,
            )

        medium = classify_parallel_lines(4)
        hard = classify_parallel_lines(8)

        self.assertEqual(medium["stratum"], "medium")
        self.assertFalse(medium["strict_easy"])
        self.assertEqual(hard["stratum"], "hard")
        self.assertEqual(hard["difficulty_score_components"]["line_instances"], 2.5)

    def test_balancing_hits_requested_distribution_and_intersection_share(self):
        pools = {
            name: {
                "intersection": [f"{name}_i_{idx}" for idx in range(50)],
                "plain": [f"{name}_p_{idx}" for idx in range(50)],
            }
            for name in DIFFICULTY_ORDER
        }
        pools["empty"]["intersection"] = []
        counts, report = select_balanced_candidates(
            pools,
            empty_candidate_pools(),
            100,
            DEFAULT_DIFFICULTY_RATIOS,
            0.30,
            42,
        )
        self.assertEqual(sum(counts.values()), 100)
        self.assertEqual(report["selected_total"], 100)
        self.assertEqual(report["target_quotas"], {
            "empty": 0,
            "easy": 30,
            "medium": 33,
            "hard": 27,
            "very_hard": 10,
        })
        self.assertAlmostEqual(report["actual_intersection_ratio"], 0.30, places=2)
        self.assertEqual(report["intersection_constraint_scope"], "global")
        self.assertEqual(report["exact_repeated_records"], 0)
        self.assertEqual(report["translated_grid_records"], 0)

    def test_global_intersection_target_does_not_force_each_bucket_to_same_ratio(self):
        ratios = {
            "empty": 0.0,
            "easy": 0.5,
            "medium": 0.2,
            "hard": 0.2,
            "very_hard": 0.1,
        }
        pools = {
            "empty": {"intersection": [], "plain": []},
            "easy": {"intersection": [], "plain": [f"easy_p_{idx}" for idx in range(100)]},
            "medium": {
                "intersection": [f"medium_i_{idx}" for idx in range(20)],
                "plain": [f"medium_p_{idx}" for idx in range(80)],
            },
            "hard": {"intersection": [f"hard_i_{idx}" for idx in range(100)], "plain": []},
            "very_hard": {
                "intersection": [f"very_hard_i_{idx}" for idx in range(100)],
                "plain": [],
            },
        }
        counts, report = select_balanced_candidates(
            pools, empty_candidate_pools(), 100, ratios, 0.30, 11
        )
        self.assertEqual(sum(counts.values()), 100)
        buckets = report["base_grid"]["buckets"]
        self.assertEqual(buckets["easy"]["selected_intersection"], 0)
        self.assertEqual(buckets["hard"]["selected_intersection"], 20)
        self.assertEqual(buckets["very_hard"]["selected_intersection"], 10)
        self.assertEqual(buckets["medium"]["selected_intersection"], 0)
        self.assertAlmostEqual(report["actual_intersection_ratio"], 0.30, places=2)

    def test_global_allocator_respects_limited_unique_intersection_records(self):
        pools = {
            "empty": {"intersection": [], "plain": []},
            "easy": {
                "intersection": [f"easy_i_{idx}" for idx in range(5)],
                "plain": [f"easy_p_{idx}" for idx in range(100)],
            },
            "medium": {
                "intersection": [f"medium_i_{idx}" for idx in range(30)],
                "plain": [f"medium_p_{idx}" for idx in range(100)],
            },
            "hard": {"intersection": [], "plain": []},
            "very_hard": {"intersection": [], "plain": []},
        }
        quotas = {"empty": 0, "easy": 20, "medium": 20, "hard": 0, "very_hard": 0}
        plan, report = allocate_global_intersection_quotas(
            pools,
            quotas,
            20,
            random.Random(7),
        )
        self.assertEqual(sum(plan.values()), 20)
        self.assertLessEqual(plan["easy"], 5)
        self.assertGreaterEqual(plan["medium"], 15)
        self.assertEqual(report["constraint_scope"], "global")

    def test_short_bucket_is_redistributed_to_medium_and_hard(self):
        pools = empty_candidate_pools()
        pools["easy"]["plain"] = [f"easy_{idx}" for idx in range(5)]
        pools["medium"]["plain"] = [f"medium_{idx}" for idx in range(30)]
        pools["hard"]["plain"] = [f"hard_{idx}" for idx in range(30)]
        pools["very_hard"]["plain"] = [f"very_hard_{idx}" for idx in range(5)]
        counts, report = select_balanced_candidates(
            pools,
            empty_candidate_pools(),
            40,
            DEFAULT_DIFFICULTY_RATIOS,
            0.0,
            7,
        )
        self.assertEqual(sum(counts.values()), 40)
        self.assertTrue(all(count == 1 for count in counts.values()))
        self.assertEqual(report["translated_grid_records"], 0)
        self.assertGreater(report["final_bucket_counts"]["medium"], 13)
        self.assertGreater(report["final_bucket_counts"]["hard"], 11)

    def test_translation_grid_fills_remaining_unique_shortage(self):
        base = empty_candidate_pools()
        for name in ("easy", "medium", "hard", "very_hard"):
            base[name]["plain"] = [f"base_{name}"]
        translated = empty_candidate_pools()
        translated["medium"]["plain"] = [f"shift_medium_{idx}" for idx in range(10)]
        translated["hard"]["plain"] = [f"shift_hard_{idx}" for idx in range(10)]
        counts, report = select_balanced_candidates(
            base,
            translated,
            12,
            DEFAULT_DIFFICULTY_RATIOS,
            0.0,
            7,
        )
        self.assertEqual(sum(counts.values()), 12)
        self.assertEqual(report["base_grid_records"], 4)
        self.assertEqual(report["translated_grid_records"], 8)
        self.assertEqual(report["exact_repeated_records"], 0)
        self.assertTrue(all(count == 1 for count in counts.values()))

    def test_global_intersection_target_is_coordinated_across_grid_kinds(self):
        base = empty_candidate_pools()
        base["medium"]["plain"] = [f"base_p_{idx}" for idx in range(8)]
        translated = empty_candidate_pools()
        translated["medium"]["intersection"] = ["shift_i_0", "shift_i_1"]
        ratios = {name: 0.0 for name in DIFFICULTY_ORDER}
        ratios["medium"] = 1.0
        counts, report = select_balanced_candidates(
            base,
            translated,
            10,
            ratios,
            0.20,
            17,
        )
        self.assertEqual(sum(counts.values()), 10)
        self.assertEqual(report["base_grid_records"], 8)
        self.assertEqual(report["translated_grid_records"], 2)
        self.assertEqual(report["base_grid"]["intersection_plan"]["planned_records"], 0)
        self.assertEqual(report["translation_grid"]["intersection_plan"]["planned_records"], 2)
        self.assertAlmostEqual(report["actual_intersection_ratio"], 0.20)

    def test_translation_grid_metadata_uses_half_patch_offsets(self):
        row = {
            "id": "sample_r003_c001",
            "image": "images/train/sample/sample_r003_c001.png",
            "tile_id": "sample",
            "meta": {"x0": 384, "y0": 128},
        }
        tagged = annotate_translation_grid(row, 256)
        self.assertEqual(tagged["id"], "sample_x00384_y00128")
        self.assertEqual(tagged["image"], "images/train/sample/sample_x00384_y00128.png")
        self.assertEqual(tagged["meta"]["grid_patch_id"], "sample_r003_c001")
        self.assertEqual(tagged["meta"]["translation_offset"], [128, 128])
        self.assertEqual(tagged["meta"]["grid_kind"], "translated")


class DatasetV2ObsutilTest(unittest.TestCase):
    def test_obsutil_recursive_copy_uses_config_and_parallel_jobs(self):
        backend = ObsutilBackend(
            r"C:\tools\obsutil.exe",
            r"C:\Users\tester\.obsutilconfig",
            jobs=6,
        )
        with patch("scripts.tools.build_rc_dataset_v2_from_obs.subprocess.run") as run:
            backend.download_tree("obs://bucket/source/", r"D:\rcv2\raw")
        command = run.call_args.args[0]
        self.assertEqual(command[:3], [r"C:\tools\obsutil.exe", "cp", "obs://bucket/source/"])
        self.assertIn("-r", command)
        self.assertIn("-f", command)
        self.assertIn("-j=6", command)
        self.assertIn(r"-config=C:\Users\tester\.obsutilconfig", command)
        self.assertTrue(run.call_args.kwargs["check"])

    def test_default_output_is_scoped_under_the_user_data_prefix(self):
        self.assertEqual(
            DEFAULT_OUTPUT_OBS_ROOT,
            "obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/data/"
            "rc_dataset_v2_550k_noempty_i30_shift128/",
        )


if __name__ == "__main__":
    unittest.main()
