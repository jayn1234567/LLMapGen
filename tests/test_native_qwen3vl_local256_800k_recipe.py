from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL = REPO_ROOT / "scripts/qwen3vl_native/train/train_sft_stage_a_lane_intersection_datasetv2_three_image_local256_800k_qwen3vl8b_lora_npu.sh"
SMOKE = REPO_ROOT / "scripts/npu/test/smoke_sft_stage_a_lane_intersection_datasetv2_three_image_local256_800k_native_qwen3vl8b_lora_npu.sh"
SETUP = REPO_ROOT / "scripts/npu/setup/create_mllm_native_qwen3vl_torch240_npu_env_from_infer.sh"
TRAIN_ENTRY = REPO_ROOT / "mllm/native_qwen3vl/train_sft.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_formal_recipe_uses_verified_assets_and_requested_batch():
    content = _read(FORMAL)

    assert "local256_rawpos/local256_rawlane_pose_800k.tar" in content
    assert "jjh/checkpoints/Qwen3-VL-8B-Instruct/" in content
    assert "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}" in content
    assert "TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-128}" in content
    assert "TRAIN_SAMPLE_LIMIT=${TRAIN_SAMPLE_LIMIT:-0}" in content
    assert "EXPECTED_SOURCE_TRAIN_SAMPLES=${EXPECTED_SOURCE_TRAIN_SAMPLES:-800000}" in content
    assert "MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}" in content
    assert "NUM_EPOCHS=${NUM_EPOCHS:-8}" in content
    assert "EXPECTED_NUM_IMAGES=${EXPECTED_NUM_IMAGES:-3}" in content


def test_formal_recipe_trains_all_three_native_lora_groups_without_zero():
    content = _read(FORMAL)

    assert "-m mllm.native_qwen3vl.train_sft" in content
    assert "LORA_ENABLE=True" in content
    assert "VISION_LORA_ENABLE=True" in content
    assert "MERGER_LORA_ENABLE=True" in content
    assert '--vision_lora_enable "${VISION_LORA_ENABLE}"' in content
    assert '--merger_lora_enable "${MERGER_LORA_ENABLE}"' in content
    assert "ENABLE_EVAL=False" in content
    assert "--deepspeed" not in content
    assert "deepspeed==" not in content
    assert "zero_shards" not in content
    assert "numpy==1.26.4" in content
    assert "opencv-python-headless==4.11.0.86" in content
    assert "protobuf==4.25.7" in content
    assert "huggingface-hub==0.36.2" not in content
    assert "--gradient_checkpointing_kwargs '{\"use_reentrant\": false}'" in content
    assert '--expected_num_images "${EXPECTED_NUM_IMAGES}"' in content
    assert '--verify_lora_gradients "${VERIFY_LORA_GRADIENTS}"' in content
    assert '--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}"' in content
    assert "Native DI runtime preflight failed" in content


def test_visual_lora_forces_non_reentrant_gradient_checkpointing():
    content = _read(TRAIN_ENTRY)

    assert "configure_gradient_checkpointing(model_args, training_args)" in content
    assert 'training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}' in content
    assert "Native Qwen3-VL visual LoRA requires non-reentrant" in content
    assert "gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs" in content
    assert "trainer.prepare_lora_gradient_audit()" in content
    assert "trainer.assert_lora_gradients_verified()" in content
    assert "[native-lora-gradient-check]" in content


def test_formal_recipe_checks_local256_three_image_contract():
    content = _read(FORMAL)

    assert "--expected-image-size 256" in content
    assert "--require-three-image-rawlane-pose" in content
    assert "three_image_roles_concise_v2" in content
    assert 'expected_roles = ["bev_road_structure", "pv_camera_raw_lane", "historical_vehicle_trajectory"]' in content
    assert "metadata_roles = multi.get(\"image_roles\")" in content
    assert "metadata_order = multi.get(\"image_order\")" in content


def test_local_smoke_isolated_torch240_and_checks_merger_adapter():
    content = _read(SMOKE)

    assert str(FORMAL.relative_to(REPO_ROOT)).replace("\\", "/") in content
    assert "mllm-native-qwen3vl-torch240-py311" in content
    assert 'torch.__version__.startswith("2.4.0")' in content
    assert 'torch_npu.__version__.startswith("2.4.0")' in content
    assert 'torchvision.__version__.startswith("0.19.0")' in content
    assert 'transformers.__version__ != "4.57.3"' in content
    assert 'peft.__version__ != "0.18.0"' in content
    assert "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}" in content
    assert "adapter_config.json" in content
    assert 'target_counts = {group: 0 for group in ("language", "vision", "merger")}' in content
    assert '"merger" in lowered or "linear_fc" in lowered' in content
    assert 'lowered == "qkv" or lowered.endswith(".qkv")' in content
    assert 'lowered.endswith("attn.proj")' in content
    assert "group_for_state_name" in content
    assert "non-reentrant visual-LoRA checkpointing was not confirmed" in content
    assert "one or more LoRA groups never produced a finite non-zero gradient" in content
    assert "Saved LoRA-B weights did not change from zero" in content
    assert "optimizer.pt scheduler.pt" in content
    assert "EXPECTED_TORCH_PREFIX=2.4.0" in content


def test_setup_clones_stable_torch240_environment_without_reinstalling_torch():
    content = _read(SETUP)

    assert "mllm-infer-torch240-py311" in content
    assert "mllm-native-qwen3vl-torch240-py311" in content
    assert 'conda create -y -p "${ENV_DIR}" --clone "${SOURCE_ENV_PREFIX}"' in content
    assert 'TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==4.57.3}"' in content
    assert 'PEFT_SPEC="${PEFT_SPEC:-peft==0.18.0}"' in content
    assert "pip install torch==" not in content
    assert '"opencv-python-headless==4.10.0.84"' in content
    assert '"rasterio==1.4.4"' in content
