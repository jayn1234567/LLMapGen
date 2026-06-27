"""Runtime helpers for device and platform integration."""

from .device import (
    has_npu,
    is_accelerator_device,
    is_cuda_device,
    is_npu_device,
    maybe_enable_npu_runtime,
    resolve_ddp_backend,
    resolve_torch_device,
    seed_npu_if_available,
)

__all__ = [
    "has_npu",
    "is_accelerator_device",
    "is_cuda_device",
    "is_npu_device",
    "maybe_enable_npu_runtime",
    "resolve_ddp_backend",
    "resolve_torch_device",
    "seed_npu_if_available",
]
