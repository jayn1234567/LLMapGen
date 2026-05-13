import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .dinov2_encoder import DINOv2VisionTower
from .dinov3_encoder import DINOv3VisionTower
from .mobileclip_encoder import MobileCLIPVisionTower
from .dino_config import get_dino_config


def _normalize_dino_type(value):
    value = str(value or "").lower()
    if "dinov2" in value or value in ("dino_v2", "dino2"):
        return "dinov2"
    if "dinov3" in value or value in ("dino_v3", "dino3"):
        return "dinov3"
    return ""


def _apply_dino_config(vision_tower_cfg, dino_cfg):
    vision_tower_cfg.mm_vision_tower_type = dino_cfg.encoder_type
    if getattr(vision_tower_cfg, 'input_image_size', None) is None:
        vision_tower_cfg.input_image_size = dino_cfg.image_size
    if getattr(vision_tower_cfg, 'disable_deepstack', False):
        vision_tower_cfg.deepstack_visual_indexes = None
        print("DeepStack disabled: using ViT main feature only.")
        return
    if getattr(vision_tower_cfg, 'deepstack_visual_indexes', None) is None:
        vision_tower_cfg.deepstack_visual_indexes = dino_cfg.deepstack_visual_indexes


def _resolve_dino_type(vision_tower_cfg, vision_tower_str, img_size):
    variant = getattr(vision_tower_cfg, 'dino_variant', None) or None
    config_type = _normalize_dino_type(getattr(vision_tower_cfg, 'mm_vision_tower_type', None))
    path_type = _normalize_dino_type(vision_tower_str)
    is_dino_request = variant is not None or config_type or "dinov" in vision_tower_str.lower()
    if not is_dino_request:
        return ""

    try:
        dino_cfg = get_dino_config(vision_tower_str, input_image_size=img_size, variant=variant)
        _apply_dino_config(vision_tower_cfg, dino_cfg)
        return dino_cfg.encoder_type
    except KeyError:
        if variant is not None:
            raise

    dino_type = config_type or path_type
    if dino_type:
        vision_tower_cfg.mm_vision_tower_type = dino_type
        return dino_type

    raise ValueError(
        f"Cannot determine DINO vision tower type from '{vision_tower_str}'. "
        "Pass --dino_variant or set mm_vision_tower_type to dinov2/dinov3."
    )


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    if vision_tower is None:
        raise ValueError("Missing vision tower path")
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    vision_tower_str = str(vision_tower)
    vision_tower_lower = vision_tower_str.lower()
    img_size = getattr(vision_tower_cfg, 'input_image_size', None) or None

    vit_type = _resolve_dino_type(vision_tower_cfg, vision_tower_str, img_size)
    if vit_type == "dinov3":
        return DINOv3VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if vit_type == "dinov2":
        return DINOv2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if "dinov" in vision_tower_lower:
        raise ValueError(
            f"Ambiguous DINO vision tower '{vision_tower}'. "
            "Set mm_vision_tower_type to dinov2/dinov3 or use a vision_tower path "
            "whose folder name contains an exact DINO variant such as dinov3-vitl16."
        )

    if is_absolute_path_exists or vision_tower_str.startswith("openai") or vision_tower_str.startswith("laion") or "ShareGPT4V" in vision_tower_str:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "mobileclip" in vision_tower_lower:
        return MobileCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
