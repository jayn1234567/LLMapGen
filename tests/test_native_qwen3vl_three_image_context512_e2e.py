import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


if importlib.util.find_spec("rasterio") is None:
    sys.modules["rasterio"] = types.ModuleType("rasterio")

from scripts.tools.prepare_rc_e2e_three_image_local256_dataset import (
    VIEW_CONTEXT512_ROI256,
    prepare_dataset,
    required_auxiliary_tifs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY = (
    REPO_ROOT
    / "scripts/qwen3vl_native/test/"
    "run_and_eval_rc_e2e_three_image_context512_roi256_800k_"
    "qwen3vl8b_lora_maxlen3072_npu.sh"
)


def _source_paths(tmp_path: Path) -> Path:
    centerline = tmp_path / "scene_001" / "rc_one_patch_release" / "center_line_v2"
    inter = centerline / "inter_patch_tif" / "0_inter.tif"
    inter.parent.mkdir(parents=True)
    inter.touch()
    auxiliary = required_auxiliary_tifs(inter)
    for role in ("raw_lane", "pose"):
        path = auxiliary[role]
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return inter


def test_three_image_context512_builder_centers_all_inputs_on_roi256(monkeypatch, tmp_path):
    import scripts.tools.prepare_rc_e2e_three_image_local256_dataset as module

    inter = _source_paths(tmp_path)
    clean = np.full((3, 256, 256), 17, dtype=np.uint8)
    raw_lane = np.full((3, 256, 256), 31, dtype=np.uint8)
    pose = np.full((3, 256, 256), 47, dtype=np.uint8)
    monkeypatch.setattr(module, "discover_inter_tifs", lambda _: [inter])
    monkeypatch.setattr(module, "scene_id_for_tif", lambda _: "scene_001")
    monkeypatch.setattr(module, "_read_masked_clean", lambda *_: clean)
    monkeypatch.setattr(
        module,
        "_read_masked_binary",
        lambda path, _: raw_lane if path.name.endswith("_rawlane.tif") else pose,
    )

    output = tmp_path / "prepared_context"
    summary = prepare_dataset(
        SimpleNamespace(
            input_root=tmp_path,
            output_root=output,
            view_mode=VIEW_CONTEXT512_ROI256,
            patch_size=256,
            context_size=512,
            stride=256,
            coord_range=1000,
            black_ratio_threshold=1.0,
            max_tifs=0,
            max_patches=0,
        )
    )

    record = json.loads((output / "infer.jsonl").read_text(encoding="utf-8"))
    assert summary["view_mode"] == VIEW_CONTEXT512_ROI256
    assert summary["target_size"] == 256
    assert summary["context_size"] == 512
    assert summary["target_roi_in_image"] == [128, 128, 384, 384]
    assert record["meta"]["view_mode"] == VIEW_CONTEXT512_ROI256
    assert record["meta"]["target_roi_in_image"] == [128, 128, 384, 384]
    assert record["meta"]["input_image_size"] == 512

    for relative_path, expected_value in zip(record["images"], (17, 31, 47)):
        image = np.asarray(Image.open(output / relative_path))
        assert image.shape == (512, 512, 3)
        assert np.all(image[:128] == 0)
        assert np.all(image[:, :128] == 0)
        assert np.all(image[128:384, 128:384] == expected_value)

    prompt = record["conversations"][0]["value"]
    assert prompt.count("<image>") == 3
    assert "Each input image is a 512x512 aligned context image" in prompt
    assert "central 256x256 target ROI [128,128,384,384)" in prompt
    assert "Coordinates are relative to the target ROI" in prompt


def test_three_image_builder_keeps_local256_as_backward_compatible_default(
    monkeypatch, tmp_path
):
    import scripts.tools.prepare_rc_e2e_three_image_local256_dataset as module

    inter = _source_paths(tmp_path)
    image = np.full((3, 256, 256), 23, dtype=np.uint8)
    monkeypatch.setattr(module, "discover_inter_tifs", lambda _: [inter])
    monkeypatch.setattr(module, "scene_id_for_tif", lambda _: "scene_001")
    monkeypatch.setattr(module, "_read_masked_clean", lambda *_: image)
    monkeypatch.setattr(module, "_read_masked_binary", lambda *_: image)

    output = tmp_path / "prepared_local"
    summary = prepare_dataset(
        SimpleNamespace(
            input_root=tmp_path,
            output_root=output,
            patch_size=256,
            stride=256,
            coord_range=1000,
            black_ratio_threshold=1.0,
            max_tifs=0,
            max_patches=0,
        )
    )

    record = json.loads((output / "infer.jsonl").read_text(encoding="utf-8"))
    assert summary["view_mode"] == "local256"
    assert summary["context_size"] == 256
    assert summary["target_roi_in_image"] == [0, 0, 256, 256]
    assert record["meta"]["input_image_size"] == 256
    for relative_path in record["images"]:
        assert Image.open(output / relative_path).size == (256, 256)


def test_context512_maxlen3072_entry_preserves_original_roi256_grid_contract():
    content = ENTRY.read_text(encoding="utf-8")

    assert "CHECKPOINT_OBS_PATH=${1:-${CHECKPOINT_OBS_PATH:-}}" in content
    assert "--view-mode context512_roi256" in content
    assert "--patch-size 256" in content
    assert "--context-size 512" in content
    assert 'summary.get("target_roi_in_image") != [128, 128, 384, 384]' in content
    assert "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}" in content
    assert "NPROC_PER_NODE=${NPROC_PER_NODE:-4}" in content
    assert "PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-32}" in content
    assert "MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}" in content
    assert "PREDICTION_COORD_SCALE=0.256" in content
    assert "ORIGINAL_E2E_LANE_GRID_SIZE=256" in content
    assert "WINDOW_SIZE=256" in content
    assert "INTERSECTION_STRIDE=256" in content
    assert "RUN_ALL_EVAL=True" in content
    assert "RUN_LOW_EVAL=True" in content
    assert "RUN_HIGH_EVAL=True" in content
    assert "GT_EMPTY_SUPPRESSION=${GT_EMPTY_SUPPRESSION:-True}" in content
    assert 'PREDICTION_DIR="${RAW_RESULT_DIR}"' in content
