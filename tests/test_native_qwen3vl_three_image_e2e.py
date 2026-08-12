import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.tools.merge_native_qwen3vl_inference_shards import merge_shards
from scripts.tools.prepare_rc_e2e_three_image_local256_dataset import (
    IMAGE_ROLES,
    PROMPT_CONTRACT_VERSION,
    prepare_dataset,
    required_auxiliary_tifs,
)
from scripts.tools.split_jsonl_for_inference import split_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY = (
    REPO_ROOT
    / "scripts/qwen3vl_native/test/"
    "run_and_eval_rc_e2e_three_image_local256_800k_qwen3vl8b_lora_npu.sh"
)
INFER = REPO_ROOT / "mllm/native_qwen3vl/infer.py"


def _source_paths(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    centerline = tmp_path / "scene_001" / "rc_one_patch_release" / "center_line_v2"
    inter = centerline / "inter_patch_tif" / "0_inter.tif"
    inter.parent.mkdir(parents=True)
    inter.touch()
    auxiliary = required_auxiliary_tifs(inter)
    for path in auxiliary.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return inter, auxiliary


def test_three_image_e2e_builder_preserves_training_order_and_prompt(monkeypatch, tmp_path):
    import scripts.tools.prepare_rc_e2e_three_image_local256_dataset as module

    inter, _ = _source_paths(tmp_path)
    clean = np.full((3, 256, 256), 17, dtype=np.uint8)
    raw_lane = np.zeros((3, 256, 256), dtype=np.uint8)
    raw_lane[:, 20:24, 30:34] = 255
    pose = np.zeros((3, 256, 256), dtype=np.uint8)
    pose[:, 100:104, 120:124] = 255
    monkeypatch.setattr(module, "discover_inter_tifs", lambda _: [inter])
    monkeypatch.setattr(module, "scene_id_for_tif", lambda _: "scene_001")
    monkeypatch.setattr(module, "_read_masked_clean", lambda *_: clean)
    monkeypatch.setattr(
        module,
        "_read_masked_binary",
        lambda path, _: raw_lane if path.name.endswith("lane.tif") else pose,
    )

    output = tmp_path / "prepared"
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
    assert summary["patch_count"] == 1
    assert summary["prompt_contract_version"] == PROMPT_CONTRACT_VERSION
    assert record["meta"]["input_image_roles"] == list(IMAGE_ROLES)
    assert [Path(path).parts[0] for path in record["images"]] == [
        "images",
        "raw_lane_images",
        "pose_images",
    ]
    prompt = record["conversations"][0]["value"]
    assert prompt.count("<image>") == 3
    assert "The first image is the clean BEV road-structure image." in prompt
    assert "The second image is a lane image predicted by a PV camera model." in prompt
    assert "The third image is a historical vehicle-trajectory image." in prompt
    assert "white lines are predicted lanes" not in prompt


def test_three_image_e2e_builder_rejects_missing_pose(tmp_path):
    inter, auxiliary = _source_paths(tmp_path)
    auxiliary["pose"].unlink()
    args = SimpleNamespace(
        input_root=tmp_path,
        output_root=tmp_path / "prepared",
        patch_size=256,
        stride=256,
        coord_range=1000,
        black_ratio_threshold=1.0,
        max_tifs=0,
        max_patches=0,
    )

    with pytest.raises(FileNotFoundError, match="missing auxiliary TIFs"):
        prepare_dataset(args)


def test_split_and_merge_native_shards_are_complete(tmp_path):
    input_jsonl = tmp_path / "infer.jsonl"
    records = [{"id": f"sample_{index}"} for index in range(5)]
    input_jsonl.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    split_root = tmp_path / "split"
    split_summary = split_jsonl(input_jsonl, split_root, num_shards=2)
    assert split_summary["shard_counts"] == [3, 2]

    shard_root = tmp_path / "shards"
    for rank, ids in enumerate((("sample_0", "sample_2", "sample_4"), ("sample_1", "sample_3"))):
        root = shard_root / f"shard_{rank:05d}"
        root.mkdir(parents=True)
        payload = [
            {
                "record_id": record_id,
                "prediction_json": '{"lines":[]}',
                "parse_ok": True,
            }
            for record_id in ids
        ]
        (root / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "merged"
    report = merge_shards(
        split_root / "selected.jsonl",
        shard_root,
        output,
        output / "json",
        reset=True,
    )
    merged = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["complete"] is True
    assert report["prediction_files"] == 5
    assert [item["record_id"] for item in merged] == [record["id"] for record in records]


def test_formal_three_image_e2e_entry_requires_checkpoint_and_runs_original_metrics():
    content = ENTRY.read_text(encoding="utf-8")

    assert "CHECKPOINT_OBS_PATH=${1:-${CHECKPOINT_OBS_PATH:-}}" in content
    assert "Set CHECKPOINT_OBS_PATH" in content
    assert "Qwen3-VL-8B-Instruct" in content
    assert "prepare_rc_e2e_three_image_local256_dataset.py" in content
    assert "split_jsonl_for_inference.py" in content
    assert "merge_native_qwen3vl_inference_shards.py" in content
    assert "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-2,3,4,5,6,7}" in content
    assert "--model-base \"${QWEN3VL_PATH}\"" in content
    assert "--max-new-tokens \"${MAX_NEW_TOKENS}\"" in content
    assert "PER_DEVICE_INFER_BATCH_SIZE" not in content
    assert "RUN_ALL_EVAL=True" in content
    assert "RUN_LOW_EVAL=True" in content
    assert "RUN_HIGH_EVAL=True" in content
    assert "GT_EMPTY_SUPPRESSION=${GT_EMPTY_SUPPRESSION:-True}" in content
    assert "build_rc_e2e_patch_gt_presence.py" in content
    assert "eval_rc_e2e_context512_roi256_checkpoint12504_patch_metrics.sh" not in content
    assert "RUN_INTERSECTION_E2E=${RUN_INTERSECTION_E2E:-True}" in content
    assert 'PREDICTION_DIR="${RAW_RESULT_DIR}"' in content
    assert "WINDOW_SIZE=256" in content
    assert "INTERSECTION_STRIDE=256" in content
    assert "ORIGINAL_E2E_LANE_GRID_SIZE=256" in content
    assert "PREDICTION_COORD_SCALE=0.256" in content
    assert "INTERSECTION_EVAL_ONLY_TYPE1=${INTERSECTION_EVAL_ONLY_TYPE1:-False}" in content
    assert "INTERSECTION_GT_EMPTY_SUPPRESSION=${INTERSECTION_GT_EMPTY_SUPPRESSION:-False}" in content
    assert "eval_local512_predictions_original_intersection_e2e_npu.sh" in content
    assert "RESET_EXISTING_MODEL_OUTPUTS=False" in content


def test_native_infer_reports_di_throughput():
    content = INFER.read_text(encoding="utf-8")
    assert 'print(f"DI_throughput: {len(results) / inference_elapsed:.2f} samples/s/npu"' in content
