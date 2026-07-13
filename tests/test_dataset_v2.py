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
            100,
            DEFAULT_DIFFICULTY_RATIOS,
            0.38,
            42,
            True,
        )
        self.assertEqual(sum(counts.values()), 100)
        self.assertEqual(report["selected_total"], 100)
        self.assertEqual(report["target_quotas"], {
            "empty": 5,
            "easy": 30,
            "medium": 30,
            "hard": 25,
            "very_hard": 10,
        })
        self.assertAlmostEqual(report["actual_intersection_ratio"], 0.38, places=2)

    def test_short_hard_bucket_is_oversampled_without_duplicate_images(self):
        pools = {
            name: {"intersection": [], "plain": [f"{name}_only"]}
            for name in DIFFICULTY_ORDER
        }
        counts, report = select_balanced_candidates(
            pools,
            20,
            DEFAULT_DIFFICULTY_RATIOS,
            0.0,
            7,
            True,
        )
        self.assertEqual(sum(counts.values()), 20)
        self.assertGreater(report["oversampled_records"], 0)
        self.assertEqual(len(counts), len(DIFFICULTY_ORDER))

    def test_short_buckets_remain_short_when_oversampling_is_disabled(self):
        pools = {
            name: {"intersection": [], "plain": [f"{name}_only"]}
            for name in DIFFICULTY_ORDER
        }
        counts, report = select_balanced_candidates(
            pools,
            20,
            DEFAULT_DIFFICULTY_RATIOS,
            0.0,
            7,
            False,
        )
        self.assertLess(sum(counts.values()), 20)
        self.assertFalse(report["allow_oversample"])


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
            "obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/data/rc_dataset_v2/",
        )


if __name__ == "__main__":
    unittest.main()
