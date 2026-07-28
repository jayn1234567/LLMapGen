from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts.tools.build_rc_e2e_jsonl_from_original_manifest import convert_manifest
from scripts.tools.prepare_rc_e2e_inference_dataset import prepare_dataset


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
