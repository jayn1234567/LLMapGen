from __future__ import annotations

import json

import numpy as np
from PIL import Image

from scripts.tools.build_e2e_black_ratio_manifest import build_manifest


def test_build_manifest_uses_target_roi_and_final_record_id(tmp_path):
    image_root = tmp_path / "dataset"
    image_path = image_root / "images" / "sample.png"
    image_path.parent.mkdir(parents=True)
    array = np.full((4, 4, 3), 255, dtype=np.uint8)
    array[1:3, 1:2] = 0
    Image.fromarray(array).save(image_path)

    infer_jsonl = image_root / "infer.jsonl"
    infer_jsonl.write_text(
        json.dumps(
            {
                "id": "scene_0_0_0",
                "image": "images/sample.png",
                "meta": {"target_roi_in_image": [1, 1, 3, 3]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_json = image_root / "patch_black_ratio_manifest.json"

    summary = build_manifest(infer_jsonl, image_root, output_json, workers=1)
    manifest = json.loads(output_json.read_text(encoding="utf-8"))

    assert summary["records"] == 1
    assert manifest[0]["id"] == "scene_0_0_0"
    assert manifest[0]["black_ratio"] == 0.5
