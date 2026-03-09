import torch
import torch.nn as nn

from unimapgen.models.unimapgen_v1 import SimpleBEVEncoder


class PVEncoder(nn.Module):
    """
    Practical PV encoder for reproduction scaffold.
    Paper mentions 3DConv + Qwen2-VL-ViT; here we keep the same interface:
    - temporal aggregation via lightweight 3D conv
    - tokenization via image token encoder
    """

    def __init__(self, d_model: int, cnn_channels=(32, 64, 128), memory_tokens_hw=(2, 4)) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.GELU(),
            nn.Conv3d(8, 3, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.GELU(),
        )
        self.image_encoder = SimpleBEVEncoder(
            in_ch=3,
            channels=cnn_channels,
            d_model=d_model,
            out_hw=tuple(memory_tokens_hw),
        )

    def forward(self, pv_images: torch.Tensor) -> torch.Tensor:
        # pv_images: [B, L, C, H, W]
        b, l, c, h, w = pv_images.shape
        x = pv_images.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, L, H, W]
        x = self.temporal(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * l, c, h, w)
        tok = self.image_encoder(x)  # [B*L, M, D]
        return tok.view(b, l * tok.shape[1], tok.shape[2])
