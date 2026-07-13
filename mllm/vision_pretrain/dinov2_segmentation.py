from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGroupNormGELU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )


class MultiLayerRoadDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        projection_channels: int = 256,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.layer_projections = nn.ModuleList(
            [nn.Conv2d(hidden_size, projection_channels, kernel_size=1) for _ in range(num_layers)]
        )
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        middle_channels = max(64, projection_channels // 2)
        low_channels = max(32, middle_channels // 2)
        self.refine = nn.Sequential(
            ConvGroupNormGELU(projection_channels, projection_channels),
            ConvGroupNormGELU(projection_channels, projection_channels),
        )
        self.up_stage1 = ConvGroupNormGELU(projection_channels, middle_channels)
        self.up_stage2 = ConvGroupNormGELU(middle_channels, low_channels)
        self.output_refine = ConvGroupNormGELU(low_channels, low_channels)
        self.classifier = nn.Conv2d(low_channels, num_classes, kernel_size=1)

    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        *,
        prefix_tokens: int,
        patch_height: int,
        patch_width: int,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if len(hidden_states) != len(self.layer_norms):
            raise ValueError(f"Expected {len(self.layer_norms)} hidden states, got {len(hidden_states)}")
        features = []
        expected_tokens = patch_height * patch_width
        for hidden_state, norm, projection in zip(
            hidden_states,
            self.layer_norms,
            self.layer_projections,
        ):
            tokens = norm(hidden_state)[:, prefix_tokens:]
            if tokens.shape[1] != expected_tokens:
                raise ValueError(
                    f"DINOv2 produced {tokens.shape[1]} patch tokens, expected {expected_tokens} "
                    f"for grid {patch_height}x{patch_width}."
                )
            feature = tokens.transpose(1, 2).reshape(
                tokens.shape[0],
                tokens.shape[2],
                patch_height,
                patch_width,
            )
            features.append(projection(feature))

        weights = torch.softmax(self.layer_weights, dim=0).to(dtype=features[0].dtype)
        fused = sum(weight * feature for weight, feature in zip(weights, features))
        fused = self.refine(fused)
        fused = F.interpolate(fused, scale_factor=2.0, mode="bilinear", align_corners=False)
        fused = self.up_stage1(fused)
        fused = F.interpolate(fused, scale_factor=2.0, mode="bilinear", align_corners=False)
        fused = self.up_stage2(fused)
        fused = F.interpolate(fused, size=output_size, mode="bilinear", align_corners=False)
        return self.classifier(self.output_refine(fused))


class Dinov2RoadSegmentationModel(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        *,
        input_size: int = 518,
        hidden_state_indices: Sequence[int] = (6, 12, 18, 24),
        projection_channels: int = 256,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.input_size = int(input_size)
        self.hidden_state_indices = tuple(int(index) for index in hidden_state_indices)
        config = vision_encoder.config
        self.patch_size = int(config.patch_size)
        if self.input_size % self.patch_size != 0:
            raise ValueError(
                f"input_size={self.input_size} must be divisible by DINOv2 patch_size={self.patch_size}."
            )
        self.num_register_tokens = int(getattr(config, "num_register_tokens", 0) or 0)
        if self.num_register_tokens:
            raise ValueError(
                "This recipe requires non-register DINOv2 to match Jiangjihua's vision tower; "
                f"the supplied model has {self.num_register_tokens} register tokens."
            )
        if not self.hidden_state_indices:
            raise ValueError("hidden_state_indices cannot be empty.")
        self.decoder = MultiLayerRoadDecoder(
            hidden_size=int(config.hidden_size),
            num_layers=len(self.hidden_state_indices),
            projection_channels=int(projection_channels),
            num_classes=int(num_classes),
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        input_size: int = 518,
        hidden_state_indices: Sequence[int] = (6, 12, 18, 24),
        projection_channels: int = 256,
        gradient_checkpointing: bool = True,
    ) -> "Dinov2RoadSegmentationModel":
        from transformers import Dinov2Model

        vision_encoder = Dinov2Model.from_pretrained(
            model_name_or_path,
            local_files_only=True,
        )
        if gradient_checkpointing:
            vision_encoder.gradient_checkpointing_enable()
        vision_encoder.requires_grad_(True)
        return cls(
            vision_encoder,
            input_size=input_size,
            hidden_state_indices=hidden_state_indices,
            projection_channels=projection_channels,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_encoder(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        selected = []
        for index in self.hidden_state_indices:
            resolved_index = index if index >= 0 else len(hidden_states) + index
            if not 0 <= resolved_index < len(hidden_states):
                raise IndexError(
                    f"hidden_state_index={index} resolves to {resolved_index}, but model returned "
                    f"{len(hidden_states)} hidden states."
                )
            selected.append(hidden_states[resolved_index])
        patch_height = pixel_values.shape[-2] // self.patch_size
        patch_width = pixel_values.shape[-1] // self.patch_size
        return self.decoder(
            selected,
            prefix_tokens=1 + self.num_register_tokens,
            patch_height=patch_height,
            patch_width=patch_width,
            output_size=(pixel_values.shape[-2], pixel_values.shape[-1]),
        )

    def head_state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu() for key, value in self.decoder.state_dict().items()}

    def save_vision_tower(self, output_dir: str | Path, image_processor: object, metadata: dict) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        state_dict = {
            key: value.detach().cpu()
            for key, value in self.vision_encoder.state_dict().items()
        }
        self.vision_encoder.save_pretrained(
            str(output_path),
            state_dict=state_dict,
            safe_serialization=True,
        )
        image_processor.save_pretrained(str(output_path))
        (output_path / "private_seg_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
