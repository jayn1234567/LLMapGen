# NPU Scripts

This folder contains Ascend NPU / DI-platform launchers for the cleaned DINOv2 centerline route.

Current formal entrypoints:

- `train/train_dinov2_centerline_qwen_lora_npu.sh`
- `test/test_dinov2_centerline_qwen_lora_npu.sh`

These scripts intentionally wrap the same Python entrypoints used on GPU:

- `scripts/train_dinov2_centerline.py`
- `scripts/predict_dinov2_centerline.py`

All paths should be supplied through environment variables. Avoid hard-coded personal paths in NPU launchers.
