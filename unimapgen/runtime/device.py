"""Small device helpers shared by training and inference entrypoints.

The centerline route is primarily written against PyTorch and Hugging Face
Trainer. Ascend NPU support is activated lazily by importing torch_npu only
when the caller asks for NPU or uses auto device selection on an NPU machine.
"""

from __future__ import annotations

import os
from typing import Optional

import torch


def _import_torch_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return True


def configure_npu_environment() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "1800")


def has_npu() -> bool:
    if not _import_torch_npu():
        return False
    npu_backend = getattr(torch, "npu", None)
    if npu_backend is None:
        return False
    try:
        return bool(npu_backend.is_available())
    except Exception:
        return False


def maybe_enable_npu_runtime(requested_backend: str = "auto") -> str:
    backend = str(requested_backend or "auto").strip().lower()
    if backend not in {"auto", "cuda", "npu", "cpu"}:
        raise ValueError(f"Unsupported device backend: {requested_backend!r}")

    if backend == "cpu":
        return "cpu"
    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend was requested, but torch.cuda.is_available() is false.")
        return "cuda"
    if backend == "npu":
        configure_npu_environment()
        if not has_npu():
            raise RuntimeError("NPU backend was requested, but torch_npu/NPU is not available.")
        return "npu"

    if has_npu():
        configure_npu_environment()
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_torch_device(requested_device: str = "auto") -> str:
    device = str(requested_device or "auto").strip().lower()
    if not device or device == "auto":
        return maybe_enable_npu_runtime("auto")
    if device.startswith("npu"):
        maybe_enable_npu_runtime("npu")
        return device
    if device.startswith("cuda"):
        maybe_enable_npu_runtime("cuda")
        return device
    if device == "cpu":
        return "cpu"
    return str(requested_device).strip()


def resolve_ddp_backend(resolved_backend: str, explicit_backend: str = "") -> Optional[str]:
    explicit = str(explicit_backend or "").strip()
    if explicit:
        return explicit
    backend = str(resolved_backend or "").strip().lower()
    if backend == "npu":
        return "hccl"
    if backend == "cuda":
        return "nccl"
    return None


def is_npu_device(device: str) -> bool:
    return str(device or "").strip().lower().startswith("npu")


def is_cuda_device(device: str) -> bool:
    return str(device or "").strip().lower().startswith("cuda")


def is_accelerator_device(device: str) -> bool:
    return is_cuda_device(device) or is_npu_device(device)


def seed_npu_if_available(seed: int) -> None:
    if not has_npu():
        return
    try:
        torch.npu.manual_seed_all(int(seed))
    except Exception:
        return
