import json
from pathlib import Path

import torch

from mllm.native_qwen3vl.infer import (
    _completion_token_ids,
    _merge_native_processor_inputs,
    run_batch,
)
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.tools.merge_native_qwen3vl_inference_shards import merge_shards
from scripts.tools.prepare_rc_e2e_three_image_local256_dataset import (
    IMAGE_ROLES,
    PROMPT_CONTRACT_VERSION,
    prepare_dataset,
    required_auxiliary_tifs,
    validate_evaluation_alignment,
)
from scripts.tools.split_jsonl_for_inference import split_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY = (
    REPO_ROOT
    / "scripts/qwen3vl_native/test/"
    "run_and_eval_rc_e2e_three_image_local256_800k_qwen3vl8b_lora_npu.sh"
)
INFER = REPO_ROOT / "mllm/native_qwen3vl/infer.py"


class _BatchProcessor:
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

    tokenizer = _Tokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del messages, tokenize, add_generation_prompt
        return "fake prompt"

    def __call__(self, *, text, images, return_tensors):
        del text, return_tensors
        token_count = 2 + len(images)
        return {
            "input_ids": torch.arange(1, token_count + 1).unsqueeze(0),
            "attention_mask": torch.ones(1, token_count, dtype=torch.long),
            "pixel_values": torch.zeros(len(images), 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 16, 16]] * len(images)),
        }

    def batch_decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return ['{"lines":[]}'] * int(token_ids.shape[0])


class _BatchModel:
    generation_config = SimpleNamespace(pad_token_id=0, eos_token_id=2)

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        completion = torch.tensor(
            [[101, 102]] * int(input_ids.shape[0]),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat((input_ids, completion), dim=1)


def _source_paths(tmp_path: Path) -> tuple[Path, dict[str, Path | None]]:
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
    assert summary["evaluation_root"] is None
    assert record["meta"]["input_image_roles"] == list(IMAGE_ROLES)
    assert record["meta"]["mask_tif"] is None
    assert record["meta"]["raw_lane_image_source"] == "lane_patch_tif/0_lane.tif"
    assert record["meta"]["pose_image_source"] == "lane_patch_tif/0_pose.tif"
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
    assert auxiliary["pose"] is not None
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


def test_inference_and_evaluation_raster_alignment_is_required(monkeypatch, tmp_path):
    import scripts.tools.prepare_rc_e2e_three_image_local256_dataset as module

    inference_tif, _ = _source_paths(tmp_path / "inference")
    evaluation_tif, _ = _source_paths(tmp_path / "evaluation")

    def fake_discover(root):
        return [evaluation_tif] if Path(root) == tmp_path / "evaluation" else [inference_tif]

    metadata = {
        "width": 512,
        "height": 256,
        "crs": "EPSG:32650",
        "transform": [0.2, 0.0, 1.0, 0.0, -0.2, 2.0, 0.0, 0.0, 1.0],
        "bounds": [1.0, -49.2, 103.4, 2.0],
        "grid_rows": 1,
        "grid_cols": 2,
    }
    monkeypatch.setattr(module, "discover_inter_tifs", fake_discover)
    monkeypatch.setattr(module, "_raster_grid_metadata", lambda *_: metadata)

    report = validate_evaluation_alignment(
        [inference_tif], tmp_path / "evaluation", patch_size=256, require_exact_keys=True
    )
    assert report["ok"] is True
    assert report["matched_tif_count"] == 1

    monkeypatch.setattr(module, "discover_inter_tifs", lambda _: [])
    with pytest.raises(ValueError, match="alignment failed"):
        validate_evaluation_alignment(
            [inference_tif], tmp_path / "evaluation", patch_size=256, require_exact_keys=True
        )


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
    assert "eval_patches.zip" in content
    assert "e2e_data.zip" in content
    assert "INFERENCE_E2E_DATA_OBS_PATH" in content
    assert "EVALUATION_E2E_DATA_OBS_PATH" in content
    assert '--evaluation-root "${EVALUATION_E2E_DATA_ROOT}"' in content
    assert "prepare_rc_e2e_three_image_local256_dataset.py" in content
    assert "split_jsonl_for_inference.py" in content
    assert "merge_native_qwen3vl_inference_shards.py" in content
    assert "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-2,3,4,5,6,7}" in content
    assert "--model-base \"${QWEN3VL_PATH}\"" in content
    assert "--max-new-tokens \"${MAX_NEW_TOKENS}\"" in content
    assert "PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-1}" in content
    assert '--per-device-infer-batch-size "${PER_DEVICE_INFER_BATCH_SIZE}"' in content
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
    assert "--per-device-infer-batch-size" in content


def test_native_processor_inputs_merge_with_left_padding_and_image_grids():
    first = {
        "input_ids": torch.tensor([[11, 12]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "pixel_values": torch.tensor([[1.0]]),
        "image_grid_thw": torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]),
    }
    second = {
        "input_ids": torch.tensor([[21, 22, 23]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "pixel_values": torch.tensor([[2.0]]),
        "image_grid_thw": torch.tensor([[[10, 11, 12], [13, 14, 15], [16, 17, 18]]]),
    }

    merged, lengths = _merge_native_processor_inputs([first, second], pad_token_id=0)

    assert lengths == [2, 3]
    assert merged["input_ids"].tolist() == [[0, 11, 12], [21, 22, 23]]
    assert merged["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]
    assert merged["pixel_values"].shape == (2, 1)
    assert merged["image_grid_thw"].shape == (6, 3)


def test_native_batch_completion_slicing_handles_left_padding():
    input_ids = torch.tensor([[0, 11, 12], [21, 22, 23]])
    attention_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    output_ids = torch.tensor([
        [0, 11, 12, 101, 102],
        [21, 22, 23, 201, 202],
    ])

    assert _completion_token_ids(output_ids, input_ids, attention_mask, 0).tolist() == [101, 102]
    assert _completion_token_ids(output_ids, input_ids, attention_mask, 1).tolist() == [201, 202]


def test_native_run_batch_preserves_record_order_and_matches_single_sample_results(tmp_path):
    image_paths = []
    records = []
    for index in range(2):
        image_path = tmp_path / f"sample-{index}.png"
        Image.new("RGB", (8, 8), color=(index * 50, 0, 0)).save(image_path)
        image_paths.append(image_path)
        records.append(
            (
                index,
                {
                    "id": f"sample-{index}",
                    "image": image_path.name,
                    "conversations": [{"from": "human", "value": "<image> map"}],
                    "meta": {
                        "coord_mode": "norm1000",
                        "coord_range": 1000,
                        "patch_width": 256,
                        "patch_height": 256,
                    },
                },
            )
        )

    args = SimpleNamespace(
        image_folder=str(tmp_path),
        prompt="<image> map",
        system_prompt=None,
        max_new_tokens=16,
        temperature=0.0,
        default_patch_size=256,
        coord_range=1000,
        coord_mode="auto",
        map_task="lane_intersection",
        model_name_or_path="fake-checkpoint",
    )
    batched = run_batch(_BatchModel(), _BatchProcessor(), records, args, torch.device("cpu"))
    singles = [
        run_batch(
            _BatchModel(), _BatchProcessor(), [record], args, torch.device("cpu")
        )[0]
        for record in records
    ]

    assert [item["record_id"] for item in batched] == ["sample-0", "sample-1"]
    assert [item["prediction_json"] for item in batched] == [
        item["prediction_json"] for item in singles
    ]
    assert [item["image"] for item in batched] == [str(path) for path in image_paths]
    assert all(item["parse_ok"] for item in batched)
