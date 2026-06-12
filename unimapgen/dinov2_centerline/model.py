"""Model facade for the minimal DINOv2 centerline route.

The implementation lives in the historical model module because older
checkpoints depend on those class names and save/load conventions.  This file
keeps the cleaned route readable by giving it a single import location.
"""

from __future__ import annotations

from unimapgen.models.qwen3_rc_dinov2_centerline_json_sft import (
    Qwen3RCDinoCenterlineJSONSFTModel,
    save_qwen3_rc_dinov2_centerline_json_modules,
)

__all__ = [
    "Qwen3RCDinoCenterlineJSONSFTModel",
    "save_qwen3_rc_dinov2_centerline_json_modules",
]

