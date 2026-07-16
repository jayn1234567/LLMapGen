#!/usr/bin/env python3
"""Verify that an exported DINOv3 tower is loadable by the MLLM vision wrapper."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--device", choices=("auto", "npu", "cuda", "cpu"), default="auto")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--select-layer", type=int, default=24)
    parser.add_argument("--expected-tokens", type=int, default=1024)
    parser.add_argument("--expected-hidden-size", type=int, default=1024)
    parser.add_argument("--expected-num-layers", type=int, default=24)
    parser.add_argument("--expected-register-tokens", type=int, default=4)
    parser.add_argument("--output-json", default="")
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested in {"auto", "npu"}:
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            if requested == "npu":
                raise RuntimeError("NPU was requested, but torch_npu is not installed.")
        else:
            if hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.set_device(0)
                return torch.device("npu:0")
            if requested == "npu":
                raise RuntimeError("NPU was requested, but torch.npu.is_available() is false.")
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    if requested in {"npu", "cuda"}:
        raise RuntimeError(f"Requested device is unavailable: {requested}")
    return torch.device("cpu")


def validate_export_files(vision_tower: Path) -> dict[str, object]:
    required = ("config.json", "preprocessor_config.json")
    missing = [name for name in required if not (vision_tower / name).is_file()]
    weight_candidates = sorted(
        path.name
        for pattern in (
            "model.safetensors",
            "model.safetensors.index.json",
            "model-*.safetensors",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "pytorch_model-*.bin",
        )
        for path in vision_tower.glob(pattern)
    )
    if not weight_candidates:
        missing.append("DINOv3 weight file")
    if missing:
        raise FileNotFoundError(
            f"Incomplete exported vision tower at {vision_tower}: missing {missing}"
        )
    return {
        "required_files": list(required),
        "weight_files": sorted(set(weight_candidates)),
        "has_private_seg_metadata": (vision_tower / "private_seg_metadata.json").is_file(),
    }


def main() -> None:
    args = build_parser().parse_args()
    vision_tower_path = Path(args.vision_tower).expanduser().resolve()
    if not vision_tower_path.is_dir():
        raise FileNotFoundError(f"Vision tower directory not found: {vision_tower_path}")
    files_report = validate_export_files(vision_tower_path)
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type in {"npu", "cuda"} else torch.float32

    loader_name = "mllm.DINOv3VisionTower"
    loader_fallback_reason = None
    try:
        from mllm.model.multimodal_encoder.dinov3_encoder import DINOv3VisionTower

        wrapper_args = SimpleNamespace(
            mm_vision_select_layer=int(args.select_layer),
            mm_vision_select_feature="patch",
            unfreeze_mm_vision_tower=False,
            input_image_size=int(args.input_size),
            deepstack_visual_indexes=None,
            vision_layer_fusion_indexes=None,
            vision_layer_fusion_type="mean",
        )
        tower = DINOv3VisionTower(str(vision_tower_path), wrapper_args, delay_load=False)
        tower.to(device=device, dtype=dtype)
        tower.eval()
        config = tower.config
        num_layers = int(tower.num_layers)
        num_patches_per_side = int(tower.num_patches_per_side)
        num_patches = int(tower.num_patches)

        def run_forward(pixel_values: torch.Tensor) -> tuple[torch.Tensor, object, int]:
            main_features, deepstack_features = tower(pixel_values)
            return main_features, deepstack_features, int(tower.select_layer_idx)

    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("mllm"):
            raise
        from transformers import AutoImageProcessor, DINOv3ViTModel

        loader_name = "transformers.DINOv3ViTModel (MLLM-compatible fallback)"
        loader_fallback_reason = repr(exc)
        print(
            "[verify-dinov3-tower] optional MLLM import unavailable; "
            f"using HF-compatible fallback: {loader_fallback_reason}",
            flush=True,
        )
        AutoImageProcessor.from_pretrained(str(vision_tower_path), local_files_only=True)
        hf_model = DINOv3ViTModel.from_pretrained(
            str(vision_tower_path), local_files_only=True
        )
        hf_model.to(device=device, dtype=dtype)
        hf_model.eval()
        config = hf_model.config
        num_layers = int(config.num_hidden_layers)
        num_patches_per_side = int(args.input_size) // int(config.patch_size)
        num_patches = num_patches_per_side**2
        raw_select_layer = int(args.select_layer)
        if raw_select_layer >= 0:
            resolved_select_layer = min(raw_select_layer, num_layers)
        else:
            resolved_select_layer = min(num_layers + raw_select_layer, num_layers - 1)
        resolved_select_layer = max(0, resolved_select_layer)
        prefix_tokens = 1 + int(getattr(config, "num_register_tokens", 0) or 0)

        def run_forward(pixel_values: torch.Tensor) -> tuple[torch.Tensor, object, int]:
            outputs = hf_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            if resolved_select_layer == len(outputs.hidden_states) - 1:
                selected = outputs.last_hidden_state
            else:
                selected = outputs.hidden_states[resolved_select_layer]
            return selected[:, prefix_tokens:], None, resolved_select_layer

    checks = {
        "hidden_size": int(config.hidden_size),
        "num_layers": num_layers,
        "patch_size": int(config.patch_size),
        "num_register_tokens": int(getattr(config, "num_register_tokens", 0) or 0),
        "input_size": int(args.input_size),
        "num_patches_per_side": num_patches_per_side,
        "num_patches": num_patches,
    }
    expected = {
        "hidden_size": int(args.expected_hidden_size),
        "num_layers": int(args.expected_num_layers),
        "num_register_tokens": int(args.expected_register_tokens),
        "num_patches": int(args.expected_tokens),
    }
    mismatches = {
        key: {"actual": checks[key], "expected": expected_value}
        for key, expected_value in expected.items()
        if checks[key] != expected_value
    }
    if mismatches:
        raise ValueError(f"DINOv3 configuration mismatch: {json.dumps(mismatches)}")

    pixel_values = torch.randn(
        1,
        3,
        int(args.input_size),
        int(args.input_size),
        device=device,
        dtype=dtype,
    )
    started = time.perf_counter()
    with torch.no_grad():
        main_features, deepstack_features, resolved_select_layer = run_forward(pixel_values)
        if device.type == "npu":
            torch.npu.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - started, 1e-6)

    actual_shape = list(main_features.shape)
    expected_shape = [1, int(args.expected_tokens), int(args.expected_hidden_size)]
    if actual_shape != expected_shape:
        raise ValueError(f"Unexpected MLLM feature shape: {actual_shape}, expected {expected_shape}")
    if deepstack_features is not None:
        raise ValueError("No-DeepStack verification unexpectedly returned DeepStack features.")
    if not bool(torch.isfinite(main_features).all().item()):
        raise FloatingPointError("Exported vision tower produced non-finite features.")

    throughput = 1.0 / elapsed
    report = {
        "status": "passed",
        "vision_tower": str(vision_tower_path),
        "device": str(device),
        "dtype": str(dtype),
        "loader": loader_name,
        "loader_fallback_reason": loader_fallback_reason,
        "select_layer": int(args.select_layer),
        "resolved_select_layer": resolved_select_layer,
        "feature_shape": actual_shape,
        "forward_seconds": elapsed,
        "throughput_samples_per_second_per_npu": throughput,
        "config": checks,
        "files": files_report,
    }
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else vision_tower_path.parent / "vision_tower_verify.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[verify-dinov3-tower] {json.dumps(report, ensure_ascii=True)}", flush=True)
    print(f"DI_throughput: {throughput:.2f} samples/s/npu", flush=True)


if __name__ == "__main__":
    main()
