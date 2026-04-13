import math
from collections import OrderedDict
from pathlib import Path
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoImageProcessor, AutoModel
except Exception:  # pragma: no cover
    AutoImageProcessor = None
    AutoModel = None

try:
    from torchvision.models import resnet50
    from torchvision.ops import FeaturePyramidNetwork
    from torchvision.ops.misc import FrozenBatchNorm2d
except Exception:  # pragma: no cover
    resnet50 = None
    FeaturePyramidNetwork = None
    FrozenBatchNorm2d = None


class SimpleBEVEncoder(nn.Module):
    def __init__(self, in_ch: int, channels: Sequence[int], d_model: int, out_hw: Tuple[int, int]) -> None:
        super().__init__()
        layers = []
        prev = int(in_ch)
        for c in channels:
            layers.extend(
                [
                    nn.Conv2d(prev, int(c), kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(int(c)),
                    nn.GELU(),
                ]
            )
            prev = int(c)
        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Conv2d(prev, int(d_model), kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((int(out_hw[0]), int(out_hw[1])))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.proj(x)
        x = self.pool(x)
        b, d, h, w = x.shape
        return x.view(b, d, h * w).transpose(1, 2).contiguous()


class ResNetFPNEncoder(nn.Module):
    """ResNet50 + FPN encoder that pools multi-scale features into fixed visual tokens."""

    expects_normalized_input = True

    def __init__(
        self,
        out_dim: int,
        out_hw: Tuple[int, int],
        pretrained: bool = False,
        weights_path: str = "",
    ) -> None:
        super().__init__()
        if resnet50 is None or FeaturePyramidNetwork is None or FrozenBatchNorm2d is None:
            raise RuntimeError("torchvision ResNet50/FPN is unavailable.")

        weights = None
        local_weights_path = str(weights_path).strip()
        if bool(pretrained) and not local_weights_path:
            try:
                from torchvision.models import ResNet50_Weights

                weights = ResNet50_Weights.IMAGENET1K_V2
            except Exception:
                weights = None

        backbone = resnet50(weights=weights, norm_layer=FrozenBatchNorm2d)
        if local_weights_path:
            state = torch.load(local_weights_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise RuntimeError(f"Unsupported local ResNet50 weights format: {local_weights_path}")
            cleaned = {}
            for key, value in state.items():
                name = str(key)
                if name.startswith("module."):
                    name = name[len("module.") :]
                if name.startswith("backbone."):
                    name = name[len("backbone.") :]
                cleaned[name] = value
            missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
            print(
                f"[SatelliteEncoder] loaded local ResNet50 weights: {local_weights_path} "
                f"(missing={len(missing)} unexpected={len(unexpected)})"
            )
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=int(out_dim),
        )
        self.pool = nn.AdaptiveAvgPool2d(tuple(out_hw))
        self.hidden_size = int(out_dim)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        feats = OrderedDict(
            {
                "c2": c2,
                "c3": c3,
                "c4": c4,
                "c5": c5,
            }
        )
        pyramid = self.fpn(feats)
        base_key = next(iter(pyramid.keys()))
        target_hw = pyramid[base_key].shape[-2:]

        fused = None
        for feat in pyramid.values():
            if feat.shape[-2:] != target_hw:
                feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
            fused = feat if fused is None else fused + feat
        fused = fused / float(len(pyramid))

        pooled = self.pool(fused)
        b, d, h, w = pooled.shape
        tokens = pooled.view(b, d, h * w).transpose(1, 2).contiguous()
        return {
            "tokens": tokens,
            "dense_features": fused,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["tokens"]

    def freeze_backbone_body(self) -> None:
        # Approximate HRMapNet's `frozen_stages=1`: freeze the shallow stem and
        # first residual stage, while keeping later backbone stages and the FPN trainable.
        for module in (self.stem, self.layer1):
            for param in module.parameters():
                param.requires_grad = False
        for module in (self.layer2, self.layer3, self.layer4, self.fpn):
            for param in module.parameters():
                param.requires_grad = True


class SatelliteEncoder(nn.Module):
    """
    Paper-aligned satellite encoder interface.
    - Preferred: DINOv2 family from HuggingFace.
    - Fallback: lightweight CNN token encoder for offline/debug.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        local_files_only: bool = False,
        use_fallback: bool = False,
        fallback_backbone: str = "resnet50_fpn",
        fallback_pretrained: bool = False,
        fallback_weights_path: str = "",
        fallback_channels=(32, 64, 128),
        fallback_hw: Tuple[int, int] = (8, 8),
        fallback_dim: int = 256,
        out_hw: Tuple[int, int] = (8, 8),
        patch_size: int = 14,
        drop_cls_token: bool = True,
        normalize_input: bool = True,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.use_fallback = bool(use_fallback) or (AutoModel is None)
        self.hidden_size = int(fallback_dim)
        self.out_hw = (int(out_hw[0]), int(out_hw[1])) if out_hw is not None else None
        self.fallback_backbone = str(fallback_backbone).strip().lower()
        self.fallback_pretrained = bool(fallback_pretrained)
        self.fallback_weights_path = str(fallback_weights_path).strip()
        self.patch_size = max(1, int(patch_size))
        self.drop_cls_token = bool(drop_cls_token)
        self.normalize_input = bool(normalize_input)
        self.register_buffer(
            "pixel_mean",
            torch.tensor(list(image_mean), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(list(image_std), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        if not self.use_fallback:
            try:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only=bool(local_files_only),
                )
                self.hidden_size = int(getattr(self.model.config, "hidden_size", fallback_dim))
                if AutoImageProcessor is not None:
                    try:
                        proc = AutoImageProcessor.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            local_files_only=bool(local_files_only),
                        )
                        mean = getattr(proc, "image_mean", None)
                        std = getattr(proc, "image_std", None)
                        if isinstance(mean, (list, tuple)) and isinstance(std, (list, tuple)) and len(mean) == 3 and len(std) == 3:
                            self.pixel_mean.copy_(torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
                            self.pixel_std.copy_(torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
                    except Exception:
                        pass
                print(f"[SatelliteEncoder] use DINO backbone: {model_name} (hidden={self.hidden_size})")
            except Exception:
                self.use_fallback = True

        if self.use_fallback:
            fb_hw = self.out_hw if self.out_hw is not None else tuple(fallback_hw)
            if self.fallback_backbone in {"resnet50_fpn", "resnet-fpn", "resnet50"} and resnet50 is not None:
                local_weights = self.fallback_weights_path
                if local_weights and not Path(local_weights).is_file():
                    print(f"[SatelliteEncoder] local fallback weights not found: {local_weights}")
                    local_weights = ""
                self.model = ResNetFPNEncoder(
                    out_dim=int(fallback_dim),
                    out_hw=fb_hw,
                    pretrained=self.fallback_pretrained,
                    weights_path=local_weights,
                )
                self.hidden_size = int(self.model.hidden_size)
                if local_weights:
                    print(f"[SatelliteEncoder] use fallback CNN backbone: ResNet50+FPN (local weights: {local_weights})")
                elif self.fallback_pretrained:
                    print("[SatelliteEncoder] use fallback CNN backbone: ResNet50+FPN (torchvision pretrained)")
                else:
                    print("[SatelliteEncoder] use fallback CNN backbone: ResNet50+FPN")
            else:
                self.model = SimpleBEVEncoder(
                    in_ch=3,
                    channels=fallback_channels,
                    d_model=fallback_dim,
                    out_hw=fb_hw,
                )
                print("[SatelliteEncoder] use fallback CNN backbone: SimpleBEVEncoder")

    def forward_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.use_fallback:
            x = image
            if self.normalize_input and bool(getattr(self.model, "expects_normalized_input", False)):
                x = (x - self.pixel_mean.to(dtype=x.dtype, device=x.device)) / self.pixel_std.to(
                    dtype=x.dtype, device=x.device
                ).clamp_min(1e-6)
            if hasattr(self.model, "forward_features"):
                return self.model.forward_features(x)
            tokens = self.model(x)
            b, t, d = tokens.shape
            side = max(1, int(round(math.sqrt(float(t)))))
            if side * side != t:
                dense = tokens.transpose(1, 2).contiguous().unsqueeze(-1)
            else:
                dense = tokens.transpose(1, 2).contiguous().view(b, d, side, side)
            return {
                "tokens": tokens,
                "dense_features": dense,
            }
        x = image
        if self.normalize_input:
            x = (x - self.pixel_mean.to(dtype=x.dtype, device=x.device)) / self.pixel_std.to(dtype=x.dtype, device=x.device).clamp_min(1e-6)
        out = self.model(pixel_values=x)
        tok = out.last_hidden_state
        if self.drop_cls_token and tok.shape[1] > 1:
            tok = tok[:, 1:, :]
        dense = self._tokens_to_dense_patch_map(tok, h=int(image.shape[-2]), w=int(image.shape[-1]))
        if self.out_hw is not None:
            pooled_dense = F.adaptive_avg_pool2d(dense, output_size=self.out_hw)
            tok = pooled_dense.flatten(2).transpose(1, 2).contiguous()
        return {
            "tokens": tok,
            "dense_features": dense,
        }

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward_features(image)["tokens"]

    def freeze_backbone_only(self) -> None:
        if self.use_fallback and isinstance(self.model, ResNetFPNEncoder):
            self.model.freeze_backbone_body()
            print("[SatelliteEncoder] applied HRMapNet-like freeze: stem+layer1 frozen, layer2-4+FPN trainable")
            return
        for param in self.parameters():
            param.requires_grad = False
        print("[SatelliteEncoder] backbone-only freeze unsupported for current encoder; froze full encoder")

    def _tokens_to_dense_patch_map(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, t, d = tokens.shape
        gh = max(1, int(h) // self.patch_size)
        gw = max(1, int(w) // self.patch_size)
        if gh * gw != t:
            side = max(1, int(round(math.sqrt(float(t)))))
            gh, gw = side, side
        n = gh * gw
        if n != t:
            if n < t:
                tokens = tokens[:, :n, :]
            else:
                pad = tokens.new_zeros((b, n - t, d))
                tokens = torch.cat([tokens, pad], dim=1)
        return tokens.view(b, gh, gw, d).permute(0, 3, 1, 2).contiguous()

    def _pool_patch_tokens(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, t, d = tokens.shape
        gh = max(1, int(h) // self.patch_size)
        gw = max(1, int(w) // self.patch_size)
        if gh * gw != t:
            side = max(1, int(round(math.sqrt(float(t)))))
            gh, gw = side, side
        n = gh * gw
        if n != t:
            if n < t:
                tokens = tokens[:, :n, :]
            else:
                pad = tokens.new_zeros((b, n - t, d))
                tokens = torch.cat([tokens, pad], dim=1)
        feat = tokens.view(b, gh, gw, d).permute(0, 3, 1, 2).contiguous()
        pooled = F.adaptive_avg_pool2d(feat, output_size=self.out_hw)
        return pooled.flatten(2).transpose(1, 2).contiguous()
