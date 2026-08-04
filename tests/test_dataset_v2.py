import json
import random
import shutil
import tarfile
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

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
from data_process.build_dataset_v2_staged import (
    STAGE_MARKER,
    STAGE_VERSION,
    finalize_stages,
    stable_sample_split,
)
from data_process.fixed_source_splits import (
    assignment_manifest_id,
    load_fixed_source_split_manifest,
    split_for_raw_sample,
    validate_fixed_holdout_coverage,
)
from data_process.state_update_dataset_common import (
    build_sft_record,
    centered_target_roi,
    extract_centered_context,
    IGNORED_LANE_TYPE_CODES,
    intersection_type_name,
    lane_type_name,
    normalize_lane_type_code,
    required_archive_member,
    semantic_sft_record_counts,
)
from scripts.tools.build_rc_dataset_v2_from_obs import (
    DEFAULT_OUTPUT_OBS_ROOT,
    ObsutilBackend,
)
from scripts.tools.build_rc_dataset_v2_streaming_from_obs import (
    build_stage_command,
    completed_stage,
    parse_args as parse_streaming_args,
)
from scripts.tools.build_rc_dataset_v2_context512_windows import (
    build_compact_id_filter,
    subset_spec,
    verify_id_pairing,
)
from scripts.tools.create_fixed_source_split_manifest import (
    add_representativeness_scores,
    main as create_fixed_source_split_manifest,
    select_source_balanced,
    target_profile,
)
from scripts.tools.build_rc_dataset_v2_rawlane_pose_800k_fixed_eval_windows import (
    FIXED_REUSE_INTERSECTION_RATIO,
    bootstrap_command as fixed_eval_bootstrap_command,
    final_build_command as fixed_eval_final_build_command,
    parse_args as parse_fixed_eval_build_args,
)
from scripts.tools.build_rc_dataset_v2_rawlane_pose_800k_windows import (
    DIFFICULTY_RATIOS as RAWLANE_POSE_DIFFICULTY_RATIOS,
    parse_args as parse_rawlane_pose_args,
    run_streaming_builder as run_rawlane_pose_builder,
)
from scripts.tools.build_rc_dataset_v2_rawlane_pose_context512_roi256_550k_from_staging_windows import (
    DIFFICULTY_RATIOS as CONTEXT_550K_DIFFICULTY_RATIOS,
    TARGET_SAMPLES as CONTEXT_550K_TARGET_SAMPLES,
    finalize_command as context_550k_finalize_command,
    parse_args as parse_context_550k_args,
)
from scripts.tools.build_rc_dataset_v2_rawlane_pose_three_image_800k_from_staging_windows import (
    IMAGE_ROLES as THREE_IMAGE_ROLES,
    finalization_command as three_image_finalization_command,
    parse_args as parse_three_image_args,
    synthesize_clean_context_from_local256,
    transform_record as transform_three_image_record,
    validate_stage_compatibility as validate_three_image_staging,
)


class DatasetV2ContextTest(unittest.TestCase):
    def test_context_550k_wrapper_reuses_staging_without_obs(self):
        args = parse_context_550k_args([
            "--staging-root", r"D:\bootstrap\staging",
            "--work-root", r"D:\fixed",
            "--fixed-source-split-manifest", r"D:\splits\v1.json",
            "--resume",
        ])
        command = [str(item) for item in context_550k_finalize_command(
            args,
            Path(r"D:\bootstrap\staging"),
            Path(r"D:\fixed\output"),
            Path(r"D:\splits\v1.json"),
        )]
        self.assertEqual(command[command.index("--views") + 1], "context")
        self.assertEqual(
            int(command[command.index("--train-target-samples") + 1]),
            CONTEXT_550K_TARGET_SAMPLES,
        )
        self.assertEqual(
            command[command.index("--difficulty-ratios") + 1],
            CONTEXT_550K_DIFFICULTY_RATIOS,
        )
        self.assertIn("--repartition-existing-stages-by-fixed-manifest", command)
        self.assertIn("--resume", command)
        self.assertNotIn("--source-obs-root", command)
        self.assertNotIn("--stage", command)

    def test_pose_archive_member_and_two_image_record(self):
        member = tarfile.TarInfo("sample/patch_tif/0_pose.tif")
        member.size = 12
        member.type = tarfile.REGTYPE
        self.assertTrue(required_archive_member(member))

        row = {
            "id": "sample_x00000_y00000",
            "image": "images/train/sample/sample_x00000_y00000.png",
            "pose_image": "pose_images/train/sample/sample_x00000_y00000.png",
            "raw_lane_image": "raw_lane_images/train/sample/sample_x00000_y00000.png",
            "incoming_traces": [],
            "incoming_intersections": [],
            "target_lines": [],
            "meta": {"x0": 0, "y0": 0},
        }
        record = build_sft_record(
            row,
            256,
            True,
            "a",
            raw_lane_overlay=True,
            pose_second_image=True,
            save_raw_lane_image=True,
        )
        self.assertEqual(record["images"], [row["image"], row["pose_image"]])
        self.assertEqual(record["raw_lane_image"], row["raw_lane_image"])
        self.assertNotIn(record["raw_lane_image"], record["images"])
        prompt = record["conversations"][0]["value"]
        self.assertEqual(prompt.count("<image>"), 2)
        self.assertIn("historical vehicle-trajectory image", prompt)
        self.assertNotIn("additional evidence", prompt)
        self.assertNotIn("driving direction", prompt)
        self.assertEqual(
            record["meta"]["input_image_roles"],
            ["bev_road_structure", "historical_vehicle_trajectory"],
        )

    def test_three_image_record_uses_clean_bev_rawlane_then_pose(self):
        row = {
            "id": "sample_x00000_y00000",
            "image": "images/train/sample/sample_x00000_y00000.png",
            "raw_lane_image": "raw_lane_images/train/sample/sample_x00000_y00000.png",
            "pose_image": "pose_images/train/sample/sample_x00000_y00000.png",
            "incoming_traces": [],
            "incoming_intersections": [],
            "target_lines": [],
            "meta": {"x0": 0, "y0": 0},
        }
        record = build_sft_record(
            row,
            256,
            True,
            "a",
            raw_lane_overlay=False,
            raw_lane_separate_image=True,
            pose_second_image=True,
            save_raw_lane_image=True,
        )
        self.assertEqual(
            record["images"],
            [row["image"], row["raw_lane_image"], row["pose_image"]],
        )
        self.assertEqual(record["meta"]["input_image_roles"], THREE_IMAGE_ROLES)
        self.assertTrue(record["meta"]["raw_lane_active_model_input"])
        self.assertNotIn("raw_lane_overlay", record["meta"])
        prompt = record["conversations"][0]["value"]
        self.assertEqual(prompt.count("<image>"), 3)
        self.assertIn("first image is the clean BEV road-structure image", prompt)
        self.assertIn("second image is a lane image predicted by a PV camera model", prompt)
        self.assertIn("third image is a historical vehicle-trajectory image", prompt)
        self.assertNotIn("white lane overlay", prompt)

    def test_three_image_postprocess_rewrites_overlay_pose_record(self):
        record = {
            "id": "sample_x00000_y00000",
            "image": "images/train/sample/sample_x00000_y00000.png",
            "images": [
                "images/train/sample/sample_x00000_y00000.png",
                "pose_images/train/sample/sample_x00000_y00000.png",
            ],
            "raw_lane_image": "raw_lane_images/train/sample/sample_x00000_y00000.png",
            "meta": {
                "source_index": 0,
                "coord_mode": "norm1000",
                "coord_range": 1000,
                "pixel_patch_size": 256,
                "context_image_size": 256,
                "raw_lane_overlay": True,
                "raw_lane_overlay_source": "patch_tif/0_lane.tif",
            },
            "conversations": [
                {"from": "human", "value": "<image>\n<image>\nold prompt"},
                {"from": "gpt", "value": '{"lines":[]}'},
            ],
        }
        transformed = transform_three_image_record(record, "train")
        self.assertEqual(
            transformed["images"],
            [
                record["image"],
                record["raw_lane_image"],
                record["images"][1],
            ],
        )
        self.assertEqual(transformed["pose_image"], record["images"][1])
        self.assertFalse(transformed["meta"]["raw_lane_overlay"])
        self.assertNotIn("raw_lane_overlay_source", transformed["meta"])
        self.assertEqual(transformed["meta"]["input_image_roles"], THREE_IMAGE_ROLES)
        prompt = transformed["conversations"][0]["value"]
        self.assertEqual(prompt.count("<image>"), 3)
        self.assertNotIn("white lane overlay", prompt)

    def test_three_image_recipe_only_finalizes_existing_staging(self):
        args = parse_three_image_args([
            "--aux-staging-root", r"D:\aux\staging",
            "--fixed-source-split-manifest", r"D:\splits\fixed.json",
            "--resume",
        ])
        command = [str(item) for item in three_image_finalization_command(
            args,
            Path(r"D:\three_image\output"),
        )]
        self.assertIn("data_process/build_dataset_v2_staged.py", command)
        self.assertIn("--repartition-existing-stages-by-fixed-manifest", command)
        self.assertEqual(command[command.index("--views") + 1], "both")
        self.assertNotIn("--source-obs-root", command)
        self.assertNotIn("--raw-lane-overlay", command)
        self.assertIn("--resume", command)

    def test_three_image_clean_context_can_be_reconstructed_from_local_tiles(self):
        colors = {
            (0, 0): (255, 0, 0),
            (256, 0): (0, 255, 0),
            (0, 256): (0, 0, 255),
            (256, 256): (255, 255, 0),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "stage"
            image_root = stage / "variants" / "local256" / "images" / "train" / "tile"
            image_root.mkdir(parents=True)
            for (x0, y0), color in colors.items():
                patch_id = f"tile_x{x0:05d}_y{y0:05d}"
                Image.new("RGB", (256, 256), color).save(image_root / f"{patch_id}.png")
            record = {
                "id": "tile_x00128_y00128",
                "meta": {
                    "tile_id": "tile",
                    "x0": 128,
                    "y0": 128,
                    "context_image_size": 512,
                    "context_box_full": [0, 0, 512, 512],
                    "source_image_size": [512, 512],
                },
            }
            output = root / "context.png"
            mode = synthesize_clean_context_from_local256(stage, record, output)
            self.assertEqual(mode, "mosaic_from_local256")
            with Image.open(output) as image:
                self.assertEqual(image.getpixel((64, 64)), colors[(0, 0)])
                self.assertEqual(image.getpixel((320, 64)), colors[(256, 0)])
                self.assertEqual(image.getpixel((64, 320)), colors[(0, 256)])
                self.assertEqual(image.getpixel((320, 320)), colors[(256, 256)])

    def test_three_image_preflight_prefers_separate_clean_context_staging(self):
        geometry = {
            "source_index": 0,
            "target_patch_size": 256,
            "context_size": 512,
            "train_stride": 128,
            "eval_test_stride": 256,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local_stage = root / "clean_local" / "source_00"
            context_stage = root / "clean_context" / "source_00"
            auxiliary_stage = root / "auxiliary" / "source_00"
            for stage in (local_stage, context_stage, auxiliary_stage):
                stage.mkdir(parents=True)
            (local_stage / STAGE_MARKER).write_text(
                json.dumps({**geometry, "variants": ["local256"], "raw_lane_overlay": False}),
                encoding="utf-8",
            )
            (context_stage / STAGE_MARKER).write_text(
                json.dumps({
                    **geometry,
                    "variants": ["context512_roi256"],
                    "raw_lane_overlay": False,
                }),
                encoding="utf-8",
            )
            (auxiliary_stage / STAGE_MARKER).write_text(
                json.dumps({
                    **geometry,
                    "variants": ["local256", "context512_roi256"],
                    "raw_lane_overlay": True,
                    "save_raw_lane_image": True,
                    "pose_second_image": True,
                }),
                encoding="utf-8",
            )
            resolved = validate_three_image_staging(
                root / "clean_local",
                root / "auxiliary",
                root / "clean_context",
            )
            self.assertEqual(resolved[0]["local256"], local_stage)
            self.assertEqual(resolved[0]["context512_roi256"], context_stage)

    def test_streaming_stage_command_enables_pose_second_image(self):
        args = parse_streaming_args([
            "--work-root", "work",
            "--raw-lane-overlay",
            "--require-raw-lane",
            "--save-raw-lane-image",
            "--pose-second-image",
            "--pose-threshold", "3",
        ])
        command = [str(item) for item in build_stage_command(
            args,
            Path("raw/source"),
            Path("stage/source"),
            Path("raw"),
            0,
            "obs://bucket/source/",
            256,
            256,
            256,
            128,
            None,
            False,
            "local",
        )]
        self.assertIn("--pose-second-image", command)
        self.assertEqual(command[command.index("--pose-threshold") + 1], "3.0")
        self.assertIn("--raw-lane-overlay", command)
        self.assertIn("--save-raw-lane-image", command)

    def test_streaming_stage_command_enables_separate_rawlane_image(self):
        args = parse_streaming_args([
            "--work-root", "work",
            "--require-raw-lane",
            "--save-raw-lane-image",
            "--raw-lane-separate-image",
            "--pose-second-image",
        ])
        command = [str(item) for item in build_stage_command(
            args,
            Path("raw/source"),
            Path("stage/source"),
            Path("raw"),
            0,
            "obs://bucket/source/",
            256,
            512,
            256,
            128,
            None,
            False,
            "both",
        )]
        self.assertIn("--raw-lane-separate-image", command)
        self.assertIn("--save-raw-lane-image", command)
        self.assertIn("--pose-second-image", command)
        self.assertNotIn("--raw-lane-overlay", command)

    def test_dual_resolution_streaming_stage_command(self):
        args = parse_streaming_args([
            "--work-root", "work",
            "--secondary-local256-staging-root", "stage256",
            "--views", "both",
        ])
        self.assertEqual(args.secondary_local256_train_stride, 128)
        command = [str(item) for item in build_stage_command(
            args,
            Path("raw/source"),
            Path("stage256/source"),
            Path("raw"),
            3,
            "obs://bucket/source/",
            256,
            256,
            256,
            128,
            None,
            True,
            "local",
        )]
        self.assertIn("--delete-input-root-after-stage", command)
        self.assertEqual(command[command.index("--views") + 1], "local")
        self.assertEqual(command[command.index("--patch-size") + 1], "256")
        self.assertEqual(command[command.index("--train-stride") + 1], "128")
        primary_command = [str(item) for item in build_stage_command(
            args,
            Path("raw/source"),
            Path("stage/source"),
            Path("raw"),
            3,
            "obs://bucket/source/",
            256,
            512,
            256,
            128,
            None,
            False,
        )]
        self.assertEqual(primary_command[primary_command.index("--views") + 1], "both")

    def test_streaming_parser_accepts_finalize_copy_mode(self):
        args = parse_streaming_args([
            "--work-root", "work",
            "--copy-mode", "copy",
        ])
        self.assertEqual(args.copy_mode, "copy")

    def test_rawlane_pose_recipe_reserves_nonblack_empty_samples(self):
        self.assertEqual(
            RAWLANE_POSE_DIFFICULTY_RATIOS,
            "empty=0.05,easy=0.25,medium=0.33,hard=0.27,very_hard=0.10",
        )

    def test_rawlane_pose_reuse_staging_runs_finalize_without_streaming_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            staging_root = root / "bootstrap_staging"
            staging_root.mkdir()
            args = parse_rawlane_pose_args([
                "--work-root", str(root / "fixed"),
                "--reuse-staging-root", str(staging_root),
                "--fixed-source-split-manifest", str(root / "fixed.json"),
                "--resume",
            ])
            paths = {"output_root": root / "fixed" / "output"}
            with patch(
                "scripts.tools.build_rc_dataset_v2_rawlane_pose_800k_windows.run"
            ) as mocked_run:
                run_rawlane_pose_builder(paths, args)
            command = [str(item) for item in mocked_run.call_args.args[0]]
            self.assertIn("data_process/build_dataset_v2_staged.py", command)
            self.assertIn("--repartition-existing-stages-by-fixed-manifest", command)
            self.assertNotIn("scripts/tools/build_rc_dataset_v2_streaming_from_obs.py", command)

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

    def test_fixed_large_map_manifest_assigns_complement_to_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fixed.json"
            eval_ids = ["map_eval_a", "map_eval_b"]
            test_ids = ["map_test_a"]
            path.write_text(json.dumps({
                "format_version": "rc_fixed_source_split_v1",
                "manifest_id": assignment_manifest_id(eval_ids, test_ids),
                "raw_sample_ids_by_split": {"eval": eval_ids, "test": test_ids},
            }), encoding="utf-8")
            manifest = load_fixed_source_split_manifest(path)
            self.assertEqual(split_for_raw_sample("map_eval_a", manifest), "eval")
            self.assertEqual(split_for_raw_sample("map_test_a", manifest), "test")
            self.assertEqual(split_for_raw_sample("new_future_map", manifest), "train")
            report = validate_fixed_holdout_coverage(
                {
                    "train": ["new_future_map"],
                    "eval": eval_ids,
                    "test": test_ids,
                },
                manifest,
            )
            self.assertEqual(report["status"], "passed")
            with self.assertRaisesRegex(ValueError, "leaked into train"):
                validate_fixed_holdout_coverage(
                    {
                        "train": ["map_eval_a", "new_future_map"],
                        "eval": ["map_eval_b"],
                        "test": test_ids,
                    },
                    manifest,
                )

    def test_streaming_stage_command_and_resume_bind_fixed_manifest_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixed_path = root / "fixed.json"
            fixed_path.write_text(json.dumps({
                "format_version": "rc_fixed_source_split_v1",
                "raw_sample_ids_by_split": {"eval": ["eval_map"], "test": ["test_map"]},
            }), encoding="utf-8")
            args = parse_streaming_args([
                "--work-root", str(root / "work"),
                "--fixed-source-split-manifest", str(fixed_path),
            ])
            command = [str(item) for item in build_stage_command(
                args,
                root / "raw" / "source",
                root / "stage" / "source",
                root / "raw",
                0,
                "obs://bucket/source/",
                256,
                512,
                256,
                128,
                None,
                False,
                "both",
            )]
            self.assertEqual(
                command[command.index("--fixed-source-split-manifest") + 1],
                str(fixed_path),
            )
            manifest = load_fixed_source_split_manifest(fixed_path)
            stage_root = root / "stage_marker"
            stage_root.mkdir()
            marker = {
                "stage_version": STAGE_VERSION,
                "semantic_validation_passed": True,
                "raw_sample_count": 2,
                "split_record_counts": {"eval": 1, "test": 1},
                "train_candidate_filter": {"sha256": ""},
                "fixed_source_split": {"file_sha256": manifest["file_sha256"]},
            }
            (stage_root / "stage_complete.json").write_text(json.dumps(marker), encoding="utf-8")
            self.assertTrue(completed_stage(
                stage_root,
                expected_fixed_split_sha256=manifest["file_sha256"],
            ))
            self.assertFalse(completed_stage(
                stage_root,
                expected_fixed_split_sha256="different",
            ))

    def test_fixed_manifest_selection_balances_seven_sources(self):
        candidates = []
        for source_index in range(7):
            for map_index in range(4):
                candidates.append({
                    "raw_sample_id": f"source_{source_index}_map_{map_index}",
                    "source_index": source_index,
                    "source_uri": f"obs://source/{source_index}",
                    "base_patch_count": 100 + map_index,
                    "intersection_patch_count": 30,
                    "intersection_ratio": 30 / (100 + map_index),
                    "difficulty_counts": {
                        "easy": 30,
                        "medium": 33,
                        "hard": 27,
                        "very_hard": 10 + map_index,
                    },
                })
        profile = target_profile(candidates)
        add_representativeness_scores(candidates, profile)
        eval_items = select_source_balanced(candidates, 14, 7, set())
        eval_ids = {item["raw_sample_id"] for item in eval_items}
        test_items = select_source_balanced(candidates, 7, 8, eval_ids)
        self.assertEqual(Counter(item["source_index"] for item in eval_items), Counter({i: 2 for i in range(7)}))
        self.assertEqual(Counter(item["source_index"] for item in test_items), Counter({i: 1 for i in range(7)}))
        self.assertFalse(eval_ids & {item["raw_sample_id"] for item in test_items})

    def test_fixed_manifest_cli_reads_completed_staging(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            staging_root = Path(temp_dir) / "staging"
            for source_index in range(7):
                stage_root = staging_root / f"source_{source_index:02d}"
                records_root = stage_root / "records"
                records_root.mkdir(parents=True)
                (stage_root / "stage_complete.json").write_text(json.dumps({
                    "source_index": source_index,
                    "source_uri": f"obs://source/{source_index}",
                }), encoding="utf-8")
                rows = []
                for map_index in range(3):
                    raw_id = f"source_{source_index}_map_{map_index}"
                    for patch_index in range(2):
                        rows.append(json.dumps({
                            "id": f"{raw_id}_patch_{patch_index}",
                            "raw_sample_id": raw_id,
                            "source_index": source_index,
                            "grid_kind": "base",
                            "difficulty": "medium",
                            "has_intersection": patch_index == 0,
                        }))
                (records_root / "train.index.jsonl").write_text(
                    "\n".join(rows) + "\n", encoding="utf-8"
                )
                for split in ("eval", "test"):
                    (records_root / f"{split}.index.jsonl").write_text("", encoding="utf-8")
            output = Path(temp_dir) / "fixed.json"
            create_fixed_source_split_manifest([
                "--staging-root", str(staging_root),
                "--output", str(output),
                "--eval-count", "14",
                "--test-count", "7",
            ])
            manifest = load_fixed_source_split_manifest(output)
            self.assertEqual(len(manifest["eval_ids"]), 14)
            self.assertEqual(len(manifest["test_ids"]), 7)
            payload = json.loads(output.read_text(encoding="utf-8"))
            source_counts = Counter(
                item["source_index"] for item in payload["selected_sources"].values()
            )
            self.assertEqual(source_counts, Counter({i: 3 for i in range(7)}))

    def test_one_command_fixed_eval_builder_bootstraps_then_builds_fixed(self):
        args = parse_fixed_eval_build_args([
            "--bootstrap-work-root", r"D:\bootstrap",
            "--fixed-work-root", r"D:\fixed",
            "--manifest-path", r"D:\splits\v1.json",
            "--obsutil-path", r"C:\tools\obsutil.exe",
            "--resume",
        ])
        bootstrap = [str(item) for item in fixed_eval_bootstrap_command(args)]
        self.assertIn("--stage-only", bootstrap)
        self.assertIn("--resume", bootstrap)
        final = [str(item) for item in fixed_eval_final_build_command(
            args, Path(r"D:\splits\v1.json")
        )]
        self.assertEqual(
            final[final.index("--fixed-source-split-manifest") + 1],
            r"D:\splits\v1.json",
        )
        self.assertEqual(
            final[final.index("--reuse-staging-root") + 1],
            str(Path(r"D:\bootstrap").resolve() / "staging_rawlane_pose_256_context"),
        )
        self.assertEqual(
            float(final[final.index("--intersection-target-ratio") + 1]),
            FIXED_REUSE_INTERSECTION_RATIO,
        )
        self.assertNotIn("--stage-only", final)

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

    def test_finalize_repartitions_bootstrap_staging_without_raw_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            staging_root = root / "bootstrap_staging"
            stage_root = staging_root / "source_00"
            records_root = stage_root / "records"
            variant_records = records_root / "local256"
            variant_records.mkdir(parents=True)
            rows = {
                "train": [
                    ("map_eval_base", "map_eval", "base"),
                    ("map_eval_shift", "map_eval", "translated"),
                ],
                "eval": [("map_train_base", "map_train", "base")],
                "test": [("map_test_base", "map_test", "base")],
            }
            for source_split, split_rows in rows.items():
                index_lines = []
                sft_lines = []
                for patch_id, raw_sample_id, grid_kind in split_rows:
                    image = f"images/{source_split}/{raw_sample_id}/{patch_id}.png"
                    pose_image = f"pose_images/{source_split}/{raw_sample_id}/{patch_id}.png"
                    raw_lane_image = (
                        f"raw_lane_images/{source_split}/{raw_sample_id}/{patch_id}.png"
                    )
                    for relative in (image, pose_image, raw_lane_image):
                        image_path = stage_root / "variants" / "local256" / relative
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        image_path.write_bytes(b"png")
                    index_lines.append(json.dumps({
                        "id": patch_id,
                        "raw_sample_id": raw_sample_id,
                        "source_index": 0,
                        "split": source_split,
                        "stratum": "easy",
                        "difficulty": "easy",
                        "difficulty_score": 1.0,
                        "has_intersection": False,
                        "grid_kind": grid_kind,
                        "image": image,
                    }))
                    target = {"lines": [{
                        "category": "centerline",
                        "lane_type": "common",
                        "start_type": "cut",
                        "end_type": "cut",
                        "points": [[0, 500], [1000, 500]],
                    }]}
                    sft_lines.append(json.dumps({
                        "id": patch_id,
                        "image": image,
                        "images": [image, pose_image],
                        "raw_lane_image": raw_lane_image,
                        "meta": {},
                        "conversations": [
                            {
                                "from": "human",
                                "value": "<image> <image> lane_type intersection_type",
                            },
                            {"from": "gpt", "value": json.dumps(target)},
                        ],
                    }))
                (records_root / f"{source_split}.index.jsonl").write_text(
                    "\n".join(index_lines) + "\n", encoding="utf-8"
                )
                (variant_records / f"{source_split}.jsonl").write_text(
                    "\n".join(sft_lines) + "\n", encoding="utf-8"
                )
            (stage_root / "stage_complete.json").write_text(json.dumps({
                "source_index": 0,
                "variants": ["local256"],
                "stage_version": STAGE_VERSION,
                "semantic_schema_version": dataset_common.SEMANTIC_SCHEMA_VERSION,
                "semantic_validation_passed": True,
                "target_patch_size": 256,
                "train_stride": 128,
                "eval_test_stride": 256,
                "fixed_source_split": None,
            }), encoding="utf-8")
            manifest_path = root / "fixed.json"
            manifest_path.write_text(json.dumps({
                "format_version": "rc_fixed_source_split_v1",
                "raw_sample_ids_by_split": {
                    "eval": ["map_eval"],
                    "test": ["map_test"],
                },
            }), encoding="utf-8")
            output_root = root / "final"
            finalize_stages(SimpleNamespace(
                staging_root=str(staging_root),
                output_root=str(output_root),
                views="local",
                train_target_samples=1,
                difficulty_ratios="empty=0,easy=1,medium=0,hard=0,very_hard=0",
                intersection_target_ratio=0.0,
                difficulty_seed=7,
                duplicate_policy="last",
                copy_mode="copy",
                train_candidate_jsonl="",
                difficulty_override_jsonl="",
                difficulty_rule_version="",
                resume=False,
                patch_size=256,
                context_size=512,
                coord_range=1000,
                fixed_source_split_manifest=str(manifest_path),
                allow_missing_fixed_holdouts=False,
                repartition_existing_stages_by_fixed_manifest=True,
            ))

            phase_root = output_root / "local256" / "phase_a"
            records_by_split = {
                split: [
                    json.loads(line)
                    for line in (phase_root / f"{split}.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                for split in ("train", "eval", "test")
            }
            self.assertEqual([item["id"] for item in records_by_split["train"]], ["map_train_base"])
            self.assertEqual([item["id"] for item in records_by_split["eval"]], ["map_eval_base"])
            self.assertEqual([item["id"] for item in records_by_split["test"]], ["map_test_base"])
            for split, records in records_by_split.items():
                self.assertTrue(records[0]["image"].startswith(f"images/{split}/"))
                self.assertTrue((output_root / "local256" / records[0]["image"]).is_file())
                self.assertTrue(records[0]["images"][1].startswith(f"pose_images/{split}/"))
                self.assertTrue(
                    records[0]["raw_lane_image"].startswith(f"raw_lane_images/{split}/")
                )
                for relative in [*records[0]["images"], records[0]["raw_lane_image"]]:
                    self.assertTrue((output_root / "local256" / relative).is_file())
            summary = json.loads((output_root / "build_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["source_stage_split_repartition"]["enabled"])
            self.assertEqual(summary["fixed_source_split_coverage"]["status"], "passed")

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
            "very_easy": 0,
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
