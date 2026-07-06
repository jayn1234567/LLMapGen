"""
DINO model configuration registry.

Each entry defines the model specs needed for vision tower setup.
The correct config is selected automatically based on the vision_tower path:
  - 'dinov2-large' -> DINOv2-L
  - 'dinov3-vitb16' -> DINOv3-B
  - 'dinov3-vitl16' -> DINOv3-L

To add a new variant, just add an entry to DINO_CONFIGS.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class DinoConfig:
    encoder_type: str
    num_layers: int
    hidden_size: int
    patch_size: int
    image_size: int
    num_register_tokens: int
    deepstack_visual_indexes: List[int]

    @property
    def skip_tokens(self):
        return 1 + self.num_register_tokens

    @property
    def num_patches_per_side(self):
        return self.image_size // self.patch_size

    @property
    def num_patches(self):
        return self.num_patches_per_side ** 2


DINO_CONFIGS: Dict[str, DinoConfig] = {
    # ------------- DINOv2 -------------
    "dinov2-large": DinoConfig(
        encoder_type="dinov2",
        num_layers=24,
        hidden_size=1024,
        patch_size=14,
        image_size=518,
        num_register_tokens=0,
        deepstack_visual_indexes=[6, 12, 18, 23],
    ),

    # ------------- DINOv3 ViT -------------
    "dinov3-vits16": DinoConfig(
        encoder_type="dinov3",
        num_layers=12,
        hidden_size=384,
        patch_size=16,
        image_size=224,
        num_register_tokens=4,
        deepstack_visual_indexes=[3, 6, 9, 11],
    ),
    "dinov3-vitb16": DinoConfig(
        encoder_type="dinov3",
        num_layers=12,
        hidden_size=768,
        patch_size=16,
        image_size=224,
        num_register_tokens=4,
        deepstack_visual_indexes=[3, 6, 9, 11],
    ),
    "dinov3-vitl16": DinoConfig(
        encoder_type="dinov3",
        num_layers=24,
        hidden_size=1024,
        patch_size=16,
        image_size=224,
        num_register_tokens=4,
        deepstack_visual_indexes=[6, 12, 18, 23],
    ),
    "dinov3-vith16plus": DinoConfig(
        encoder_type="dinov3",
        num_layers=24,
        hidden_size=1280,
        patch_size=16,
        image_size=224,
        num_register_tokens=4,
        deepstack_visual_indexes=[6, 12, 18, 23],
    ),
}

PATH_ALIAS_MAP = {
    "dinov2-large": "dinov2-large",
    "dinov2_large": "dinov2-large",
    "dinov2-l": "dinov2-large",
    "dinov2_l": "dinov2-large",
    "dinov3-small": "dinov3-vits16",
    "dinov3_small": "dinov3-vits16",
    "dinov3-s": "dinov3-vits16",
    "dinov3_s": "dinov3-vits16",
    "dinov3-base": "dinov3-vitb16",
    "dinov3_base": "dinov3-vitb16",
    "dinov3-b": "dinov3-vitb16",
    "dinov3_b": "dinov3-vitb16",
    "dinov3-large": "dinov3-vitl16",
    "dinov3_large": "dinov3-vitl16",
    "dinov3-l": "dinov3-vitl16",
    "dinov3_l": "dinov3-vitl16",
    "dinov3-huge": "dinov3-vith16plus",
    "dinov3_huge": "dinov3-vith16plus",
    "dinov3-h": "dinov3-vith16plus",
    "dinov3_h": "dinov3-vith16plus",
}


def get_dino_config(vision_tower_path: str, input_image_size: Optional[int] = None) -> DinoConfig:
    path_lower = vision_tower_path.lower()

    for key, cfg in DINO_CONFIGS.items():
        if key in path_lower:
            d = cfg.__dict__.copy()
            if input_image_size is not None:
                d["image_size"] = input_image_size
            return DinoConfig(**d)

    for alias, key in PATH_ALIAS_MAP.items():
        if alias in path_lower:
            cfg = DINO_CONFIGS[key]
            d = cfg.__dict__.copy()
            if input_image_size is not None:
                d["image_size"] = input_image_size
            return DinoConfig(**d)
    raise KeyError(f"Cannot determine DINO variant from path: {vision_tower_path}. "
                   f"Known path keys: {list(DINO_CONFIGS.keys()) + list(PATH_ALIAS_MAP.keys())}")
