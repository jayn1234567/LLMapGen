import math
import time
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from unimapgen.models.unimapgen_v1 import SimpleBEVEncoder

try:
    from transformers import AutoImageProcessor, AutoModel
except Exception:  # pragma: no cover
    AutoImageProcessor = None
    AutoModel = None


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
                print(f"[Init] Loading DINO backbone from {model_name}", flush=True)
                load_start = time.time()
                self.model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only=bool(local_files_only),
                    torch_dtype="auto",
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
                print(
                    f"[SatelliteEncoder] use DINO backbone: {model_name} "
                    f"(hidden={self.hidden_size}, load={time.time() - load_start:.1f}s)",
                    flush=True,
                )
            except Exception:
                self.use_fallback = True

        if self.use_fallback:
            fb_hw = self.out_hw if self.out_hw is not None else tuple(fallback_hw)
            self.model = SimpleBEVEncoder(
                in_ch=3,
                channels=fallback_channels,
                d_model=fallback_dim,
                out_hw=fb_hw,
            )
            print("[SatelliteEncoder] use fallback CNN backbone")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.use_fallback:
            return self.model(image)
        x = image
        if self.normalize_input:
            x = (x - self.pixel_mean.to(dtype=x.dtype, device=x.device)) / self.pixel_std.to(dtype=x.dtype, device=x.device).clamp_min(1e-6)
        out = self.model(pixel_values=x)
        tok = out.last_hidden_state
        if self.drop_cls_token and tok.shape[1] > 1:
            tok = tok[:, 1:, :]
        if self.out_hw is not None:
            tok = self._pool_patch_tokens(tok, h=int(image.shape[-2]), w=int(image.shape[-1]))
        return tok

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
