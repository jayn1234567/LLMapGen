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

    def __init__(
        self,
        d_model: int,
        cnn_channels=(32, 64, 128),
        memory_tokens_hw=(2, 4),
        num_frames_per_camera: int = 1,
        pool_frames_per_camera: bool = False,
        use_camera_embedding: bool = False,
        max_camera_groups: int = 16,
    ) -> None:
        super().__init__()
        self.num_frames_per_camera = max(1, int(num_frames_per_camera))
        self.pool_frames_per_camera = bool(pool_frames_per_camera)
        self.use_camera_embedding = bool(use_camera_embedding)
        self.max_camera_groups = max(1, int(max_camera_groups))
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
        if self.use_camera_embedding:
            self.camera_embedding = nn.Embedding(self.max_camera_groups, int(d_model))
        else:
            self.camera_embedding = None

    def forward(self, pv_images: torch.Tensor) -> torch.Tensor:
        # pv_images: [B, L, C, H, W]
        b, l, c, h, w = pv_images.shape
        if l % self.num_frames_per_camera == 0:
            num_groups = l // self.num_frames_per_camera
            frames_per_group = self.num_frames_per_camera
            x = pv_images.view(b, num_groups, frames_per_group, c, h, w)
        else:
            num_groups = 1
            frames_per_group = l
            x = pv_images.view(b, 1, frames_per_group, c, h, w)
        x = x.reshape(b * num_groups, frames_per_group, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B*num_groups, C, T, H, W]
        x = self.temporal(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        if self.pool_frames_per_camera:
            x = x.mean(dim=1)  # [B*num_groups, C, H, W]
            tok = self.image_encoder(x)  # [B*num_groups, M, D]
            tok = tok.view(b, num_groups, tok.shape[1], tok.shape[2])
            if self.camera_embedding is not None:
                camera_ids = torch.arange(num_groups, device=tok.device).clamp(max=self.max_camera_groups - 1)
                camera_emb = self.camera_embedding(camera_ids).view(1, num_groups, 1, tok.shape[-1])
                tok = tok + camera_emb.to(dtype=tok.dtype)
            return tok.view(b, num_groups * tok.shape[2], tok.shape[3])

        x = x.view(b * num_groups * frames_per_group, c, h, w)
        tok = self.image_encoder(x)  # [B*num_groups*frames_per_group, M, D]
        tok = tok.view(b, num_groups, frames_per_group, tok.shape[1], tok.shape[2])
        if self.camera_embedding is not None:
            camera_ids = torch.arange(num_groups, device=tok.device).clamp(max=self.max_camera_groups - 1)
            camera_emb = self.camera_embedding(camera_ids).view(1, num_groups, 1, 1, tok.shape[-1])
            tok = tok + camera_emb.to(dtype=tok.dtype)
        return tok.view(b, num_groups * frames_per_group * tok.shape[3], tok.shape[4])
