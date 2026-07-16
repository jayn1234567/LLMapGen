from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch.nn as nn

from .dinov2_segmentation import (
    Dinov2RoadSegmentationModel,
    enable_dinov2_gradient_checkpointing,
)


class Dinov3RoadSegmentationModel(Dinov2RoadSegmentationModel):
    """DINOv3 segmentation model that exports a standard MLLM vision tower."""

    def __init__(
        self,
        vision_encoder: nn.Module,
        *,
        input_size: int = 512,
        hidden_state_indices: Sequence[int] = (24,),
        projection_channels: int = 256,
        num_classes: int = 2,
        decoder_type: str = "legacy_single_layer",
        vision_unfreeze_last_n_blocks: int = -1,
        vision_lora_enable: bool = False,
        vision_lora_r: int = 8,
        vision_lora_alpha: float = 16.0,
        vision_lora_dropout: float = 0.0,
        vision_lora_target_modules: str | Sequence[str] = "query,value",
    ) -> None:
        if vision_lora_enable:
            raise ValueError(
                "Dinov3RoadSegmentationModel currently supports full or tail-block "
                "fine-tuning, not the DINOv2 query/value LoRA wrapper."
            )
        super().__init__(
            vision_encoder,
            input_size=input_size,
            hidden_state_indices=hidden_state_indices,
            projection_channels=projection_channels,
            num_classes=num_classes,
            decoder_type=decoder_type,
            vision_unfreeze_last_n_blocks=vision_unfreeze_last_n_blocks,
            vision_lora_enable=False,
            vision_lora_r=vision_lora_r,
            vision_lora_alpha=vision_lora_alpha,
            vision_lora_dropout=vision_lora_dropout,
            vision_lora_target_modules=vision_lora_target_modules,
            allow_register_tokens=True,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        input_size: int = 512,
        hidden_state_indices: Sequence[int] = (24,),
        projection_channels: int = 256,
        gradient_checkpointing: bool = True,
        decoder_type: str = "legacy_single_layer",
        vision_unfreeze_last_n_blocks: int = -1,
        vision_lora_enable: bool = False,
        vision_lora_r: int = 8,
        vision_lora_alpha: float = 16.0,
        vision_lora_dropout: float = 0.0,
        vision_lora_target_modules: str | Sequence[str] = "query,value",
    ) -> "Dinov3RoadSegmentationModel":
        from transformers import DINOv3ViTModel

        vision_encoder = DINOv3ViTModel.from_pretrained(
            str(model_name_or_path),
            local_files_only=True,
        )
        gradient_checkpointing_mode = "disabled"
        if gradient_checkpointing:
            gradient_checkpointing_mode = enable_dinov2_gradient_checkpointing(
                vision_encoder,
                adapter_only=False,
            )
        model = cls(
            vision_encoder,
            input_size=input_size,
            hidden_state_indices=hidden_state_indices,
            projection_channels=projection_channels,
            decoder_type=decoder_type,
            vision_unfreeze_last_n_blocks=vision_unfreeze_last_n_blocks,
            vision_lora_enable=vision_lora_enable,
            vision_lora_r=vision_lora_r,
            vision_lora_alpha=vision_lora_alpha,
            vision_lora_dropout=vision_lora_dropout,
            vision_lora_target_modules=vision_lora_target_modules,
        )
        model.gradient_checkpointing_mode = gradient_checkpointing_mode
        return model
