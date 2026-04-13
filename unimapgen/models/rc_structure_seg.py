from __future__ import annotations

from typing import Any, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from unimapgen.models.encoders.satellite_encoder import SatelliteEncoder


def choose_group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(choose_group_count(int(out_channels)), int(out_channels)),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegRefineBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(int(channels), int(channels)),
            ConvNormAct(int(channels), int(channels)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StructureSegDecoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_stages: int, out_channels: int) -> None:
        super().__init__()
        self.input_proj = ConvNormAct(int(in_channels), int(hidden_dim), kernel_size=1, padding=0)
        self.stages = nn.ModuleList([SegRefineBlock(int(hidden_dim)) for _ in range(max(1, int(num_stages)))])
        self.head = nn.Conv2d(int(hidden_dim), int(out_channels), kernel_size=1)

    def forward(self, x: torch.Tensor, output_size: int) -> torch.Tensor:
        x = self.input_proj(x)
        for stage in self.stages:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = stage(x)
        if x.shape[-2:] != (int(output_size), int(output_size)):
            x = F.interpolate(x, size=(int(output_size), int(output_size)), mode="bilinear", align_corners=False)
        return self.head(x)


class RCStructureSegModel(nn.Module):
    def __init__(
        self,
        *,
        encoder_type: str,
        output_size: int = 512,
        decoder_dim: int = 256,
        num_classes: int = 1,
        encoder_input_pad_size: int = 0,
        encoder_input_pad_fill_rgb: Sequence[float] = (10.0 / 255.0, 12.0 / 255.0, 18.0 / 255.0),
        resnet_weights_path: str = "",
        dinov2_model_name_or_path: str = "",
        dinov2_local_files_only: bool = True,
        dinov2_unfreeze_last_n_blocks: int = 12,
    ) -> None:
        super().__init__()
        self.encoder_type = str(encoder_type).strip().lower()
        self.output_size = int(output_size)
        self.decoder_dim = int(decoder_dim)
        self.num_classes = max(1, int(num_classes))
        self.encoder_input_pad_size = max(0, int(encoder_input_pad_size))
        if len(tuple(encoder_input_pad_fill_rgb)) != 3:
            raise ValueError("encoder_input_pad_fill_rgb must contain exactly 3 values.")
        self.register_buffer(
            "encoder_input_pad_fill",
            torch.tensor(list(encoder_input_pad_fill_rgb), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        if self.encoder_type in {"resnet", "resnet50", "resnet50_fpn"}:
            self.encoder = SatelliteEncoder(
                use_fallback=True,
                fallback_backbone="resnet50_fpn",
                fallback_pretrained=not bool(str(resnet_weights_path).strip()),
                fallback_weights_path=str(resnet_weights_path).strip(),
                fallback_dim=int(decoder_dim),
                out_hw=None,
            )
            self.encoder.freeze_backbone_only()
            decoder_stages = 2
        elif self.encoder_type in {"dinov2", "dinov2_vitl14", "vitl14"}:
            model_name = str(dinov2_model_name_or_path).strip()
            if not model_name:
                raise ValueError("dinov2_model_name_or_path is required for DINOv2 structure segmentation.")
            self.encoder = SatelliteEncoder(
                model_name=model_name,
                local_files_only=bool(dinov2_local_files_only),
                use_fallback=False,
                out_hw=None,
                patch_size=14,
                drop_cls_token=True,
                normalize_input=True,
            )
            # DINOv2 采用“冻结大部分 backbone，只解冻最后几层”的稳定微调策略。
            self._freeze_dinov2_tail(unfreeze_last_n=int(dinov2_unfreeze_last_n_blocks))
            decoder_stages = 4
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        self.decoder = StructureSegDecoder(
            in_channels=int(self.encoder.hidden_size),
            hidden_dim=int(self.decoder_dim),
            num_stages=int(decoder_stages),
            out_channels=int(self.num_classes),
        )

    def _freeze_dinov2_tail(self, unfreeze_last_n: int) -> None:
        backbone = getattr(self.encoder, "model", None)
        if backbone is None:
            raise RuntimeError("DINOv2 encoder backend is missing.")
        for param in backbone.parameters():
            param.requires_grad = False

        layer_list = None
        candidates: list[list[str]] = [
            ["encoder", "layer"],
            ["encoder", "layers"],
            ["layers"],
            ["blocks"],
        ]
        for path in candidates:
            node: Any = backbone
            ok = True
            for name in path:
                if not hasattr(node, name):
                    ok = False
                    break
                node = getattr(node, name)
            if ok and isinstance(node, (nn.ModuleList, list, tuple)):
                layer_list = node
                break
        if layer_list is None:
            raise RuntimeError("Unable to find DINOv2 transformer blocks for partial unfreeze.")

        total_blocks = len(layer_list)
        keep_n = max(0, min(int(unfreeze_last_n), total_blocks))
        start_idx = max(0, total_blocks - keep_n)
        # 只放开最后 N 个 block，既保留通用视觉表征，又给结构分割留任务适配空间。
        for idx, block in enumerate(layer_list):
            req = idx >= start_idx
            for param in block.parameters():
                param.requires_grad = bool(req)

        for attr_name in ("layernorm", "norm", "post_layernorm"):
            if hasattr(backbone, attr_name):
                module = getattr(backbone, attr_name)
                if isinstance(module, nn.Module):
                    for param in module.parameters():
                        param.requires_grad = True

    def backbone_trainable_parameters(self) -> Iterable[nn.Parameter]:
        if self.encoder_type.startswith("resnet"):
            return (p for p in self.encoder.parameters() if p.requires_grad)
        backbone = getattr(self.encoder, "model", None)
        if backbone is None:
            return []
        return (p for p in backbone.parameters() if p.requires_grad)

    def head_trainable_parameters(self) -> Iterable[nn.Parameter]:
        encoder_param_ids = {id(p) for p in self.backbone_trainable_parameters()}
        return (p for p in self.parameters() if p.requires_grad and id(p) not in encoder_param_ids)

    def parameter_groups(self, *, backbone_lr: float, head_lr: float, weight_decay: float) -> List[dict[str, Any]]:
        backbone_params = [p for p in self.backbone_trainable_parameters() if p.requires_grad]
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in self.parameters() if p.requires_grad and id(p) not in backbone_ids]
        groups: List[dict[str, Any]] = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": float(backbone_lr), "weight_decay": float(weight_decay)})
        if head_params:
            groups.append({"params": head_params, "lr": float(head_lr), "weight_decay": float(weight_decay)})
        return groups

    def _maybe_center_pad_image(self, image: torch.Tensor) -> torch.Tensor:
        target_size = max(int(self.encoder_input_pad_size), int(image.shape[-2]), int(image.shape[-1]))
        if target_size == int(image.shape[-2]) and target_size == int(image.shape[-1]):
            return image

        batch, channels, height, width = image.shape
        top = max(0, (target_size - int(height)) // 2)
        left = max(0, (target_size - int(width)) // 2)
        padded = self.encoder_input_pad_fill.to(dtype=image.dtype, device=image.device).expand(
            int(batch),
            int(channels),
            int(target_size),
            int(target_size),
        ).clone()
        padded[:, :, top : top + int(height), left : left + int(width)] = image
        return padded

    @staticmethod
    def _center_crop_square(x: torch.Tensor, target_size: int) -> torch.Tensor:
        if x.shape[-2] == int(target_size) and x.shape[-1] == int(target_size):
            return x
        if x.shape[-2] < int(target_size) or x.shape[-1] < int(target_size):
            raise ValueError(
                f"Cannot center-crop tensor of spatial size {tuple(x.shape[-2:])} to {(int(target_size), int(target_size))}"
            )
        top = (int(x.shape[-2]) - int(target_size)) // 2
        left = (int(x.shape[-1]) - int(target_size)) // 2
        return x[:, :, top : top + int(target_size), left : left + int(target_size)]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        encoder_input = self._maybe_center_pad_image(image)
        # 编码器可能吃的是 pad 后尺寸，最后统一裁回目标 patch 大小，保证 mask 对齐。
        features = self.encoder.forward_features(encoder_input)["dense_features"]
        if features.ndim != 4:
            raise ValueError(f"Expected 4D dense features, got {tuple(features.shape)}")
        decoder_output_size = max(int(self.output_size), int(encoder_input.shape[-2]), int(encoder_input.shape[-1]))
        logits = self.decoder(features, output_size=decoder_output_size)
        return self._center_crop_square(logits, target_size=int(self.output_size))
