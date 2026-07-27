from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

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
