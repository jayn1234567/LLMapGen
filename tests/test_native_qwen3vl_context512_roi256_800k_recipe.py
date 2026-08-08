from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL = REPO_ROOT / "scripts/qwen3vl_native/train/train_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_800k_qwen3vl8b_lora_npu.sh"
SMOKE = REPO_ROOT / "scripts/npu/test/smoke_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_800k_native_qwen3vl8b_lora_npu.sh"
TRAIN_ENTRY = REPO_ROOT / "mllm/native_qwen3vl/train_sft.py"
DATA_ENTRY = REPO_ROOT / "mllm/native_qwen3vl/data.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_formal_context512_recipe_uses_released_800k_package():
    content = _read(FORMAL)

    assert "context512_roi256_rawpos/context512_roi256_rawlane_pose_800k.tar" in content
    assert "EXPECTED_SOURCE_TRAIN_SAMPLES=${EXPECTED_SOURCE_TRAIN_SAMPLES:-800000}" in content
    assert "TRAIN_SAMPLE_LIMIT=${TRAIN_SAMPLE_LIMIT:-0}" in content
    assert "EXPECTED_DATASET_VARIANTS=${EXPECTED_DATASET_VARIANTS:-context512_roi256_rawlane_pose_800k,context512_roi256_rawlane_pose,rawlane_pose_three_image_context512_roi256_800k}" in content
    assert "--expected-image-size 512" in content


def test_formal_context512_recipe_enforces_roi_coordinate_contract():
    content = _read(FORMAL)

    assert "EXPECTED_CONTEXT_IMAGE_SIZE=${EXPECTED_CONTEXT_IMAGE_SIZE:-512}" in content
    assert "EXPECTED_TARGET_SIZE=${EXPECTED_TARGET_SIZE:-256}" in content
    assert "EXPECTED_VIEW_MODE=${EXPECTED_VIEW_MODE:-context512_roi256,context_center_roi}" in content
    assert 'expected_roi = [margin, margin, margin + expected_target_size, margin + expected_target_size]' in content
    assert 'meta.get("target_roi_in_image") != expected_roi' in content
    assert 'Coordinates are relative to the target ROI, not the full context image.' in content
    assert '--expected_context_image_size "${EXPECTED_CONTEXT_IMAGE_SIZE}"' in content
    assert '--expected_target_size "${EXPECTED_TARGET_SIZE}"' in content
    assert '--expected_view_mode "${EXPECTED_VIEW_MODE}"' in content


def test_formal_context512_recipe_matches_validated_native_lora_training_path():
    content = _read(FORMAL)

    assert "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}" in content
    assert "TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-128}" in content
    assert "NUM_EPOCHS=${NUM_EPOCHS:-8}" in content
    assert "MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}" in content
    assert "LORA_ENABLE=True" in content
    assert "VISION_LORA_ENABLE=True" in content
    assert "MERGER_LORA_ENABLE=True" in content
    assert "VERIFY_LORA_GRADIENTS=${VERIFY_LORA_GRADIENTS:-True}" in content
    assert "--deepspeed" not in content
    assert "transformers==4.57.3" in content
    assert "peft==0.18.0" in content
    assert "--gradient_checkpointing_kwargs '{\"use_reentrant\": false}'" in content


def test_context512_smoke_reuses_formal_recipe_and_full_adapter_gates():
    content = _read(SMOKE)

    assert str(FORMAL.relative_to(REPO_ROOT)).replace("\\", "/") in content
    assert "MAX_STEPS=${MAX_STEPS:-5}" in content
    assert "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}" in content
    assert "context512_roi256_rawpos/context512_roi256_rawlane_pose_800k.tar" in content
    assert "EXPECTED_TORCH_PREFIX=2.4.0" in content
    assert "native-lora-gradient-check" in content
    assert "Saved LoRA-B weights did not change from zero" in content
    assert "context=512 target=256 roi=[128, 128, 384, 384]" in content
    assert "expected_geometry=context=512 target=256" in content
    assert "DI-LIKE SMOKE PASSED" in content


def test_native_data_and_train_entries_expose_context_geometry_runtime_gate():
    data_content = _read(DATA_ENTRY)
    train_content = _read(TRAIN_ENTRY)

    assert "expected_context_image_size" in data_content
    assert "expected_target_size" in data_content
    assert "expected_view_modes" in data_content
    assert "target_roi_in_image" in data_content
    assert "expected_context_image_size" in train_content
    assert "expected_target_size" in train_content
    assert "expected_view_mode" in train_content
