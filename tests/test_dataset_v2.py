import json
import random
import shutil
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
from data_process.build_dataset_v2_staged import STAGE_VERSION, finalize_stages, stable_sample_split
from data_process.state_update_dataset_common import (
    build_sft_record,
    centered_target_roi,
    extract_centered_context,
    IGNORED_LANE_TYPE_CODES,
    intersection_type_name,
    lane_type_name,
    normalize_lane_type_code,
    semantic_sft_record_counts,
)
from scripts.tools.build_rc_dataset_v2_from_obs import (
    DEFAULT_OUTPUT_OBS_ROOT,
    ObsutilBackend,
)
from scripts.tools.build_rc_dataset_v2_streaming_from_obs import completed_stage
from scripts.tools.build_rc_dataset_v2_context512_windows import (
    build_compact_id_filter,
    subset_spec,
    verify_id_pairing,
)


class DatasetV2ContextTest(unittest.TestCase):
    def test_lane_type_mapping_and_exclusions(self):
        self.assertEqual(IGNORED_LANE_TYPE_CODES, frozenset({3, 22}))
        self.assertIsNone(lane_type_name(3))
        self.assertIsNone(lane_type_name("22"))
        self.assertEqual(lane_type_name(1), "common")
        self.assertEqual(lane_type_name(2), "right_turn")
        self.assertEqual(lane_type_name(4), "waiting_area")
        self.assertEqual(lane_type_name("18"), "bus_lane")
        self.assertEqual(lane_type_name(25), "main_auxiliary_connector")
        self.assertEqual(lane_type_name(20), "other")

    def test_stable_stage_split_is_order_independent(self):
        first = stable_sample_split("sample_123", 42, 0.9, 0.05)
        second = stable_sample_split("sample_123", 42, 0.9, 0.05)
        self.assertEqual(first, second)
        self.assertIn(first, {"train", "eval", "test"})

    def test_resume_rejects_pre_semantic_stage_markers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stage_root = Path(temp_dir)
            marker = stage_root / "stage_complete.json"
            marker.write_text(
                json.dumps({
                    "stage_version": "rc_dataset_v2_source_stage_v1",
                    "raw_sample_count": 1,
                    "split_record_counts": {"train": 2},
                }),
                encoding="utf-8",
            )
            self.assertFalse(completed_stage(stage_root))
            marker.write_text(
                json.dumps({
                    "stage_version": STAGE_VERSION,
                    "semantic_validation_passed": True,
                    "raw_sample_count": 1,
                    "split_record_counts": {"train": 2},
                    "train_candidate_filter": {"sha256": "candidate-sha"},
                }),
                encoding="utf-8",
            )
            self.assertFalse(completed_stage(stage_root))
            self.assertTrue(completed_stage(stage_root, "candidate-sha"))
            self.assertFalse(completed_stage(stage_root, "different-sha"))

    def test_resume_requires_matching_true512_geometry_and_strides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stage_root = Path(temp_dir)
            marker = stage_root / "stage_complete.json"
            marker.write_text(
                json.dumps({
                    "stage_version": STAGE_VERSION,
                    "semantic_validation_passed": True,
                    "raw_sample_count": 1,
                    "split_record_counts": {"train": 2},
                    "train_candidate_filter": {"sha256": ""},
                    "variants": ["local512"],
                    "target_patch_size": 512,
                    "train_stride": 256,
                    "eval_test_stride": 512,
                }),
                encoding="utf-8",
            )
            self.assertTrue(completed_stage(stage_root, "", ["local512"], 512, 256, 512))
            self.assertFalse(completed_stage(stage_root, "", ["local512"], 512, 128, 512))
            self.assertFalse(completed_stage(stage_root, "", ["local256"], 512, 256, 512))
            self.assertFalse(completed_stage(stage_root, "", ["local512"], 256, 256, 512))

            payload = json.loads(marker.read_text(encoding="utf-8"))
            payload.pop("eval_test_stride")
            marker.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(completed_stage(stage_root, "", ["local512"], 512, 256, 512))

    def test_context_wrapper_builds_filter_and_verifies_exact_pairing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            local_root = root / "local256"
            phase_root = local_root / "phase_a"
            phase_root.mkdir(parents=True)
            source_jsonl = phase_root / "train.jsonl"
            source_jsonl.write_text(
                "".join(json.dumps({"id": f"sample_{index}"}) + "\n" for index in range(10)),
                encoding="utf-8",
            )
            (local_root / "dataset_info.json").write_text(
                json.dumps({
                    "balance": {
                        "final_bucket_counts": {
                            "empty": 0,
                            "easy": 3,
                            "medium": 3,
                            "hard": 3,
                            "very_hard": 1,
                        },
                        "actual_intersection_ratio": 0.3,
                    }
                }),
                encoding="utf-8",
            )
            spec = subset_spec(local_root, 10)
            self.assertEqual(spec["total"], 10)
            self.assertEqual(spec["counts"]["very_hard"], 1)

            compact = root / "ids.jsonl"
            self.assertEqual(build_compact_id_filter(source_jsonl, compact, resume=False), 10)
            context_jsonl = root / "context512_roi256" / "phase_a" / "train.jsonl"
            context_jsonl.parent.mkdir(parents=True)
            shutil.copy2(source_jsonl, context_jsonl)
            report = verify_id_pairing(compact, context_jsonl, root, 10)
            self.assertTrue(report["exact_id_pairing"])
            self.assertTrue((root / "context512_roi256" / "pairing_report.json").is_file())

    def test_finalize_staged_sources_balances_and_materializes_candidates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            staging_root = root / "staging"
            output_root = root / "final"
            rows = [
                ("easy_plain", "easy", False),
                ("medium_inter", "medium", True),
                ("hard_plain", "hard", False),
                ("very_inter", "very_hard", True),
            ]
            for source_index in (0, 1):
                stage_root = staging_root / f"source_{source_index:02d}"
                records_root = stage_root / "records"
                variant_records = records_root / "local256"
                records_root.mkdir(parents=True)
                variant_records.mkdir(parents=True)
                (stage_root / dataset_common.ARCHIVE_EXTRACT_MARKER).unlink(missing_ok=True)
                (stage_root / "stage_complete.json").write_text(
                    json.dumps({
                        "source_index": source_index,
                        "variants": ["local256"],
                        "stage_version": STAGE_VERSION,
                        "semantic_schema_version": dataset_common.SEMANTIC_SCHEMA_VERSION,
                        "semantic_validation_passed": True,
                    }),
                    encoding="utf-8",
                )
                for split in ("train", "eval", "test"):
                    index_path = records_root / f"{split}.index.jsonl"
                    sft_path = variant_records / f"{split}.jsonl"
                    index_lines = []
                    sft_lines = []
                    active_rows = rows[source_index * 2:(source_index + 1) * 2] if split == "train" else []
                    for patch_id, difficulty, has_intersection in active_rows:
                        image = f"images/train/sample_{source_index}/{patch_id}.png"
                        image_path = stage_root / "variants" / "local256" / image
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        image_path.write_bytes(b"png")
                        index_lines.append(json.dumps({
                            "id": patch_id,
                            "raw_sample_id": f"sample_{source_index}_{patch_id}",
                            "source_index": source_index,
                            "split": "train",
                            "stratum": difficulty,
                            "difficulty": difficulty,
                            "difficulty_score": 1.0,
                            "has_intersection": has_intersection,
                            "grid_kind": "base",
                            "image": image,
                        }))
                        target_lines = [{
                            "category": "centerline",
                            "lane_type": "common",
                            "start_type": "cut",
                            "end_type": "cut",
                            "points": [[0, 500], [1000, 500]],
                        }]
                        if has_intersection:
                            target_lines.append({
                                "category": "intersection",
                                "intersection_type": "other",
                                "is_cut": False,
                                "points": [[100, 100], [900, 100], [900, 900], [100, 100]],
                            })
                        sft_lines.append(json.dumps({
                            "id": patch_id,
                            "image": image,
                            "meta": {},
                            "conversations": [
                                {"from": "human", "value": "<image> lane_type intersection_type"},
                                {"from": "gpt", "value": json.dumps({"lines": target_lines})},
                            ],
                        }))
                    index_path.write_text("\n".join(index_lines) + ("\n" if index_lines else ""), encoding="utf-8")
                    sft_path.write_text("\n".join(sft_lines) + ("\n" if sft_lines else ""), encoding="utf-8")

            args = SimpleNamespace(
                staging_root=str(staging_root),
                output_root=str(output_root),
                views="local",
                train_target_samples=4,
                difficulty_ratios="empty=0,easy=0.25,medium=0.25,hard=0.25,very_hard=0.25",
                intersection_target_ratio=0.5,
                difficulty_seed=7,
                duplicate_policy="last",
                copy_mode="copy",
                train_candidate_jsonl="",
                resume=False,
                coord_range=1000,
            )
            finalize_stages(args)

            train_path = output_root / "local256" / "phase_a" / "train.jsonl"
            records = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 4)
            self.assertEqual({record["id"] for record in records}, {item[0] for item in rows})
            for record in records:
                self.assertTrue((output_root / "local256" / record["image"]).is_file())
                semantic_sft_record_counts(record, strict=True, require_prompt=True)
            summary = json.loads((output_root / "build_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["semantic_validation_passed"])
            self.assertEqual(summary["stage_version"], STAGE_VERSION)
            self.assertTrue((output_root / "semantic_schema_report.json").is_file())

            candidate_path = root / "candidate_train.jsonl"
            candidate_path.write_text(
                "\n".join([
                    json.dumps({"id": "easy_plain"}),
                    json.dumps({"id": "medium_inter"}),
                ]) + "\n",
                encoding="utf-8",
            )
            filtered_output = root / "filtered"
            filtered_args = SimpleNamespace(
                staging_root=str(staging_root),
                output_root=str(filtered_output),
                views="local",
                train_target_samples=2,
                difficulty_ratios="empty=0,easy=0.5,medium=0.5,hard=0,very_hard=0",
                intersection_target_ratio=0.5,
                difficulty_seed=7,
                duplicate_policy="last",
                copy_mode="copy",
                train_candidate_jsonl=str(candidate_path),
                resume=False,
                coord_range=1000,
            )
            finalize_stages(filtered_args)
            filtered_records = [
                json.loads(line)
                for line in (filtered_output / "local256" / "phase_a" / "train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual({record["id"] for record in filtered_records}, {"easy_plain", "medium_inter"})
            filtered_summary = json.loads(
                (filtered_output / "build_summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(filtered_summary["train_candidate_filter"]["selection_is_subset"])
            self.assertEqual(filtered_summary["train_candidate_filter"]["unique_ids"], 2)

    def test_selective_archive_extraction_keeps_only_builder_inputs_and_deletes_archive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "source" / "sample_a"
            required = {
                "inter_patch_tif/0_inter.tif": b"image",
                "patch_tif/0_edit_poly.tif": b"mask",
                "label_check_crop/Lane.geojson": b'{"type":"FeatureCollection","features":[]}',
                "label_check_crop/intersection.geojson": b'{"type":"FeatureCollection","features":[]}',
            }
            for relative, payload in required.items():
                path = source_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            junk = source_root / "unused" / "large_debug.bin"
            junk.parent.mkdir(parents=True, exist_ok=True)
            junk.write_bytes(b"unused")

            archive = root / "sample_bundle.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(source_root, arcname="sample_a")
            shutil.rmtree(root / "source")

            dataset_common.extract_archives(
                root,
                delete_archive=True,
                workers=1,
                selective=True,
            )

            extracted_root = root / "sample_bundle" / "sample_a"
            for relative, payload in required.items():
                self.assertEqual((extracted_root / relative).read_bytes(), payload)
            self.assertFalse((extracted_root / "unused" / "large_debug.bin").exists())
            self.assertFalse(archive.exists())
            self.assertEqual(len(dataset_common.find_sample_roots(root)), 1)

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
        self.assertIsNone(lane_type_name(22))

    def test_lane_type_metadata_keeps_explicit_lane_classes(self):
        self.assertEqual(lane_type_name(2), "right_turn")
        self.assertEqual(lane_type_name("2.0"), "right_turn")
        self.assertEqual(lane_type_name(1), "common")
        self.assertEqual(lane_type_name(4), "waiting_area")
        self.assertEqual(lane_type_name(18), "bus_lane")
        self.assertEqual(lane_type_name(25), "main_auxiliary_connector")
        self.assertEqual(lane_type_name(None), "other")
        self.assertEqual(lane_type_name(20), "other")
        self.assertEqual(normalize_lane_type_code(" 2.0 "), 2)

    def test_intersection_type_mapping_keeps_known_classes_and_uses_other_fallback(self):
        self.assertEqual(
            intersection_type_name({"IntersectionType": 1, "IntersectionSubType": 1})[0],
            "common",
        )
        self.assertEqual(
            intersection_type_name({"intersection_type": "1", "intersection_subtype": "2"})[0],
            "t_intersection",
        )
        self.assertEqual(intersection_type_name({"IntersectionType": 3})[0], "small_untyped")
        self.assertEqual(intersection_type_name({"IntersectionType": 4})[0], "t_lane_change_area")
        self.assertEqual(intersection_type_name({"IntersectionType": 99})[0], "other")
        self.assertEqual(intersection_type_name({})[0], "other")

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
                    "lane_type": "common",
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
        self.assertIn('include "lane_type"', prompt)
        self.assertIn('include "intersection_type"', prompt)
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
