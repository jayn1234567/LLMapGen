import math
import os
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoImageProcessor, AutoModel
except Exception:  # pragma: no cover
    AutoImageProcessor = None
    AutoModel = None


def resolve_hf_snapshot_path(path: str) -> str:
    path = str(path)
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    snapshots_dir = os.path.join(path, "snapshots")
    refs_main = os.path.join(path, "refs", "main")
    if os.path.isfile(refs_main):
        with open(refs_main, "r", encoding="utf-8") as f:
            ref = f.read().strip()
        cand = os.path.join(snapshots_dir, ref)
        if os.path.isfile(os.path.join(cand, "config.json")):
            return cand
    if os.path.isdir(snapshots_dir):
        snaps = sorted(
            [
                os.path.join(snapshots_dir, x)
                for x in os.listdir(snapshots_dir)
                if os.path.isfile(os.path.join(snapshots_dir, x, "config.json"))
            ]
        )
        if snaps:
            return snaps[-1]
    raise FileNotFoundError(f"Unable to resolve HuggingFace snapshot under: {path}")


class DINOv2LaneSeg(nn.Module):
    def __init__(
        self,
        backbone_path: str,
        freeze_backbone: bool = True,
        local_files_only: bool = True,
        normalize_input: bool = True,
        decoder_dim: int = 256,
        image_mean: Optional[Sequence[float]] = None,
        image_std: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if AutoModel is None:
            raise RuntimeError("transformers is required for DINOv2LaneSeg")

        self.backbone_path = resolve_hf_snapshot_path(backbone_path)
        self.backbone = AutoModel.from_pretrained(
            self.backbone_path,
            local_files_only=bool(local_files_only),
            trust_remote_code=True,
        )
        self.hidden_size = int(getattr(self.backbone.config, "hidden_size", 1024))
        self.patch_size = int(getattr(self.backbone.config, "patch_size", 14))
        self.normalize_input = bool(normalize_input)

        mean = list(image_mean) if image_mean is not None else [0.485, 0.456, 0.406]
        std = list(image_std) if image_std is not None else [0.229, 0.224, 0.225]
        if AutoImageProcessor is not None:
            try:
                proc = AutoImageProcessor.from_pretrained(
                    self.backbone_path,
                    local_files_only=bool(local_files_only),
                    trust_remote_code=True,
                )
                if getattr(proc, "image_mean", None) is not None:
                    mean = list(proc.image_mean)
                if getattr(proc, "image_std", None) is not None:
                    std = list(proc.image_std)
            except Exception:
                pass
        self.register_buffer("pixel_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

        if bool(freeze_backbone):
            for p in self.backbone.parameters():
                p.requires_grad = False

        mid = max(64, int(decoder_dim))
        low = max(32, mid // 2)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.hidden_size, mid, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, mid // 32), num_channels=mid),
            nn.GELU(),
            nn.Conv2d(mid, low, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, low // 32), num_channels=low),
            nn.GELU(),
            nn.Conv2d(low, 1, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image
        if self.normalize_input:
            x = (x - self.pixel_mean.to(dtype=x.dtype, device=x.device)) / self.pixel_std.to(
                dtype=x.dtype, device=x.device
            ).clamp_min(1e-6)

        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            out = self.backbone(pixel_values=x)
            tokens = out.last_hidden_state[:, 1:, :]

        feat = self._tokens_to_map(tokens)
        logits = self.decoder(feat)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)

    def _tokens_to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t, c = tokens.shape
        side = int(round(math.sqrt(float(t))))
        if side * side != t:
            gh = side
            gw = max(1, t // max(1, gh))
            if gh * gw != t:
                raise ValueError(f"Cannot reshape token sequence of length {t} to 2D grid")
        else:
            gh = side
            gw = side
        return tokens.transpose(1, 2).reshape(b, c, gh, gw).contiguous()
