from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts.tools.build_rc_e2e_jsonl_from_original_manifest import convert_manifest
from scripts.tools.prepare_rc_e2e_inference_dataset import is_original_engine_debug_tif, prepare_dataset


def test_original_engine_debug_tif_detection_is_scoped():
    generated = Path("/root/scene/debug_base/nn_output/inter_patch_tif/0_inter.tif")
    source = Path(
        "/root/scene/rc_one_patch_release/center_line_v2/inter_patch_tif/0_inter.tif"
    )

    assert is_original_engine_debug_tif(generated)
    assert not is_original_engine_debug_tif(source)


def test_prepare_context512_roi256_dataset(tmp_path):
    input_root = tmp_path / "raw"
    tif_dir = (
        input_root
        / "scene_001"
        / "rc_one_patch_release"
        / "center_line_v2"
        / "inter_patch_tif"
    )
    tif_dir.mkdir(parents=True)
    image = np.full((256, 256, 3), 127, dtype=np.uint8)
    Image.fromarray(image).save(tif_dir / "0_inter.tif")

    output_root = tmp_path / "prepared"
    summary = prepare_dataset(
        SimpleNamespace(
            input_root=str(input_root),
            output_root=str(output_root),
            view_mode="context512_roi256",
            target_size=256,
            context_size=512,
            stride=256,
            coord_range=1000,
            black_ratio_threshold=0.98,
            include_intersections=True,
            max_tifs=0,
            max_patches=0,
        )
    )

    assert summary["patch_count"] == 1
    record = json.loads((output_root / "infer.jsonl").read_text(encoding="utf-8"))
    assert record["image"] == "images/scene_001/0_inter/0_0.png"
    assert record["meta"]["target_roi_in_image"] == [128, 128, 384, 384]
    assert record["meta"]["patch_width"] == 256
    assert "Coordinates are relative to the target ROI" in record["conversations"][0]["value"]

    context = Image.open(output_root / record["image"])
    assert context.size == (512, 512)
    array = np.asarray(context)
    assert np.all(array[128:384, 128:384] == 127)
    assert np.all(array[:128] == 0)


def test_prepare_local256_550k_v1_dataset(tmp_path):
    input_root = tmp_path / "raw"
    tif_dir = (
        input_root
        / "scene_001"
        / "rc_one_patch_release"
        / "center_line_v2"
        / "inter_patch_tif"
    )
    tif_dir.mkdir(parents=True)
    Image.fromarray(np.full((256, 256, 3), 91, dtype=np.uint8)).save(tif_dir / "0_inter.tif")

    output_root = tmp_path / "prepared"
    summary = prepare_dataset(
        SimpleNamespace(
            input_root=str(input_root),
            output_root=str(output_root),
            view_mode="local256",
            target_size=256,
            context_size=256,
            stride=256,
            coord_range=1000,
            prompt_profile="local256_550k_v1",
            black_ratio_threshold=0.98,
            include_intersections=True,
            max_tifs=0,
            max_patches=0,
        )
    )

    assert summary["prompt_profile"] == "local256_550k_v1"
    record = json.loads((output_root / "infer.jsonl").read_text(encoding="utf-8"))
    prompt = record["conversations"][0]["value"]
    assert '"right_turn"' in prompt
    assert '"waiting_area"' not in prompt
    assert Image.open(output_root / record["image"]).size == (256, 256)


def test_prepare_local512_550k_v1_dataset(tmp_path):
    input_root = tmp_path / "raw"
    tif_dir = (
        input_root
        / "scene_001"
        / "rc_one_patch_release"
        / "center_line_v2"
        / "inter_patch_tif"
    )
    tif_dir.mkdir(parents=True)
    Image.fromarray(np.full((512, 512, 3), 63, dtype=np.uint8)).save(tif_dir / "0_inter.tif")

    output_root = tmp_path / "prepared"
    summary = prepare_dataset(
        SimpleNamespace(
            input_root=str(input_root),
            output_root=str(output_root),
            view_mode="local512",
            target_size=512,
            context_size=512,
            stride=512,
            coord_range=1000,
            prompt_profile="local512_550k_v1",
            black_ratio_threshold=1.0,
            include_intersections=True,
            max_tifs=0,
            max_patches=0,
        )
    )

    assert summary["patch_count"] == 1
    assert summary["prompt_profile"] == "local512_550k_v1"
    record = json.loads((output_root / "infer.jsonl").read_text(encoding="utf-8"))
    assert record["meta"]["target_roi_in_image"] == [0, 0, 512, 512]
    assert record["meta"]["patch_width"] == 512
    assert record["meta"]["patch_height"] == 512
    prompt = record["conversations"][0]["value"]
    assert "original 512x512 image patch" in prompt
    assert '"waiting_area"' not in prompt
    assert Image.open(output_root / record["image"]).size == (512, 512)


def test_prepare_rawlane_local256_uses_precomposited_lane_tif(tmp_path):
    input_root = tmp_path / "raw"
    centerline_root = input_root / "scene_rawlane" / "rc_one_patch_release" / "center_line_v2"
    inter_dir = centerline_root / "inter_patch_tif"
    lane_dir = centerline_root / "lane_patch_tif"
    inter_dir.mkdir(parents=True)
    lane_dir.mkdir(parents=True)

    inter = np.full((256, 256, 3), 91, dtype=np.uint8)
    rawlane = np.full((256, 256, 3), 37, dtype=np.uint8)
    rawlane[20:24, 30:34] = 255
    Image.fromarray(inter).save(inter_dir / "0_inter.tif")
    Image.fromarray(rawlane).save(lane_dir / "0_lane.tif")

    output_root = tmp_path / "prepared_rawlane"
    summary = prepare_dataset(
        SimpleNamespace(
            input_root=str(input_root),
            output_root=str(output_root),
            view_mode="local256",
            target_size=256,
            context_size=256,
            stride=256,
            coord_range=1000,
            prompt_profile="rawlane_local256_550k_v1",
            input_raster="rawlane",
            black_ratio_threshold=1.0,
            include_intersections=True,
            max_tifs=0,
            max_patches=0,
        )
    )

    assert summary["input_raster"] == "rawlane"
    assert summary["raw_lane_overlay"] is True
    record = json.loads((output_root / "infer.jsonl").read_text(encoding="utf-8"))
    assert record["meta"]["input_raster"] == "rawlane"
    assert record["meta"]["model_source_tif"].replace("\\", "/").endswith("lane_patch_tif/0_lane.tif")
    assert "white lane overlay predicted by a PV camera model" in record["conversations"][0]["value"]
    rendered = np.asarray(Image.open(output_root / record["image"]))
    assert np.array_equal(rendered, rawlane)


def test_prepare_rawlane_local256_requires_aligned_lane_tif(tmp_path):
    input_root = tmp_path / "raw"
    inter_dir = (
        input_root
        / "scene_missing_rawlane"
        / "rc_one_patch_release"
        / "center_line_v2"
        / "inter_patch_tif"
    )
    inter_dir.mkdir(parents=True)
    Image.fromarray(np.full((256, 256, 3), 91, dtype=np.uint8)).save(inter_dir / "0_inter.tif")

    with pytest.raises(FileNotFoundError, match="RawLane input TIF"):
        prepare_dataset(
            SimpleNamespace(
                input_root=str(input_root),
                output_root=str(tmp_path / "prepared_missing"),
                view_mode="local256",
                target_size=256,
                context_size=256,
                stride=256,
                coord_range=1000,
                prompt_profile="rawlane_local256_550k_v1",
                input_raster="rawlane",
                black_ratio_threshold=1.0,
                include_intersections=True,
                max_tifs=0,
                max_patches=0,
            )
        )


def test_prepare_rawlane_context512_roi256_matches_training_prompt(tmp_path):
    input_root = tmp_path / "raw"
    centerline_root = input_root / "scene_rawlane_context" / "rc_one_patch_release" / "center_line_v2"
    inter_dir = centerline_root / "inter_patch_tif"
    lane_dir = centerline_root / "lane_patch_tif"
    inter_dir.mkdir(parents=True)
    lane_dir.mkdir(parents=True)

    inter = np.full((512, 512, 3), 91, dtype=np.uint8)
    rawlane = np.full((512, 512, 3), 37, dtype=np.uint8)
    rawlane[20:24, 30:34] = 255
    Image.fromarray(inter).save(inter_dir / "0_inter.tif")
    Image.fromarray(rawlane).save(lane_dir / "0_lane.tif")

    output_root = tmp_path / "prepared_rawlane_context"
    summary = prepare_dataset(
        SimpleNamespace(
            input_root=str(input_root),
            output_root=str(output_root),
            view_mode="context512_roi256",
            target_size=256,
            context_size=512,
            stride=256,
            coord_range=1000,
            prompt_profile="rawlane_context512_roi256_200k_v1",
            input_raster="rawlane",
            black_ratio_threshold=1.0,
            include_intersections=True,
            max_tifs=0,
            max_patches=0,
        )
    )

    assert summary["patch_count"] == 4
    assert summary["input_raster"] == "rawlane"
    assert summary["prompt_profile"] == "rawlane_context512_roi256_200k_v1"
    records = [json.loads(line) for line in (output_root / "infer.jsonl").read_text(encoding="utf-8").splitlines()]
    record = records[0]
    prompt = record["conversations"][0]["value"]
    assert record["meta"]["target_roi_in_image"] == [128, 128, 384, 384]
    assert "white lane overlay predicted by a PV camera model" in prompt
    assert "central 256x256 target ROI [128,128,384,384)" in prompt
    assert '"waiting_area"' in prompt
    rendered = np.asarray(Image.open(output_root / record["image"]))
    assert rendered.shape == (512, 512, 3)
    assert np.array_equal(rendered[128:384, 128:384], rawlane[:256, :256])


def test_prepare_context_dataset_threshold_one_skips_only_fully_black_target(tmp_path):
    input_root = tmp_path / "raw"
    tif_dir = (
        input_root
        / "scene_black_filter"
        / "rc_one_patch_release"
        / "center_line_v2"
        / "inter_patch_tif"
    )
    tif_dir.mkdir(parents=True)
    image = np.zeros((256, 512, 3), dtype=np.uint8)
    image[0, 256] = 1
    Image.fromarray(image).save(tif_dir / "0_inter.tif")

    output_root = tmp_path / "prepared"
    summary = prepare_dataset(
        SimpleNamespace(
            input_root=str(input_root),
            output_root=str(output_root),
            view_mode="context512_roi256",
            target_size=256,
            context_size=512,
            stride=256,
            coord_range=1000,
            prompt_profile="current",
            black_ratio_threshold=1.0,
            include_intersections=True,
            max_tifs=0,
            max_patches=0,
        )
    )

    assert summary["black_ratio_threshold"] == 1.0
    assert summary["black_ratio_comparison"] == ">="
    assert summary["skipped_black"] == 1
    assert summary["patch_count"] == 1
    record = json.loads((output_root / "infer.jsonl").read_text(encoding="utf-8"))
    assert record["meta"]["col"] == 1


def test_convert_original_crop_manifest_keeps_images_and_adds_norm1000_metadata(tmp_path):
    output_root = tmp_path / "original_crop"
    image_path = output_root / "images" / "scene_001" / "0_inter" / "2_3.png"
    image_path.parent.mkdir(parents=True)
    Image.fromarray(np.full((256, 256, 3), 73, dtype=np.uint8)).save(image_path)
    manifest_path = output_root / "patch_manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "id": "scene_001",
                    "tif": "0_inter",
                    "patch": "2_3.png",
                    "row": 2,
                    "col": 3,
                    "image_path": image_path.as_posix(),
                }
            ]
        ),
        encoding="utf-8",
    )

    summary = convert_manifest(manifest_path, output_root)

    assert summary["crop_backend"] == "original_rc_e2e_split_inter_tif_for_inference"
    record = json.loads((output_root / "infer.jsonl").read_text(encoding="utf-8"))
    assert record["id"] == "scene_001_0_2_3"
    assert record["image"] == "images/scene_001/0_inter/2_3.png"
    assert record["meta"]["target_roi_in_image"] == [0, 0, 256, 256]
    assert record["meta"]["coord_mode"] == "norm1000"
    assert record["meta"]["crop_black_ratio_threshold"] == 1.0
    assert record["meta"]["crop_black_ratio_comparison"] == ">="
    assert record["meta"]["x0"] == 768
    assert record["meta"]["y0"] == 512
    assert '"right_turn"' in record["conversations"][0]["value"]


def test_convert_original_crop512_manifest_uses_full_local512_prompt(tmp_path):
    output_root = tmp_path / "original_crop512"
    image_path = output_root / "images" / "scene_512" / "0_inter" / "2_3.png"
    image_path.parent.mkdir(parents=True)
    Image.fromarray(np.full((512, 512, 3), 73, dtype=np.uint8)).save(image_path)
    manifest_path = output_root / "patch_manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "id": "scene_512",
                    "tif": "0_inter",
                    "patch": "2_3.png",
                    "row": 2,
                    "col": 3,
                    "image_path": image_path.as_posix(),
                }
            ]
        ),
        encoding="utf-8",
    )

    summary = convert_manifest(
        manifest_path,
        output_root,
        prompt_profile="current",
        patch_size=512,
    )

    assert summary["view_mode"] == "local512"
    assert summary["patch_size"] == 512
    record = json.loads((output_root / "infer.jsonl").read_text(encoding="utf-8"))
    assert record["meta"]["target_roi_in_image"] == [0, 0, 512, 512]
    assert record["meta"]["x0"] == 1536
    assert record["meta"]["y0"] == 1024
    prompt = record["conversations"][0]["value"]
    assert "original 512x512 image patch" in prompt
    assert '"waiting_area"' in prompt
    assert '"intersection_type"' in prompt
