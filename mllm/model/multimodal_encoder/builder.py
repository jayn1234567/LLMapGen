import os
from .dinov2_encoder import DINOv2VisionTower
from .dino_config import get_dino_config


def _build_dinov3_vision_tower(vision_tower, vision_tower_cfg, **kwargs):
    # Keep DINOv2-only environments usable when their Transformers build does
    # not export the newer DINOv3 classes. DINOv3 still fails immediately with
    # its original actionable import error when that backbone is requested.
    from .dinov3_encoder import DINOv3VisionTower

    return DINOv3VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)


def _build_dinov3_private_seg_vision_tower(vision_tower, vision_tower_cfg, **kwargs):
    from .dinov3_private_seg_encoder import DINOv3PrivateSegVisionTower

    return DINOv3PrivateSegVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)


def _build_siglip_vision_tower(vision_tower, vision_tower_cfg, **kwargs):
    from .siglip_encoder import SigLIPVisionTower

    return SigLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)


def _build_clip_vision_tower(vision_tower, vision_tower_cfg, use_s2, **kwargs):
    from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2

    tower_class = CLIPVisionTowerS2 if use_s2 else CLIPVisionTower
    return tower_class(vision_tower, args=vision_tower_cfg, **kwargs)


def _build_mobileclip_vision_tower(vision_tower, vision_tower_cfg, **kwargs):
    from .mobileclip_encoder import MobileCLIPVisionTower

    return MobileCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)


def _normalize_dino_type(vision_tower_type):
    vision_tower_type = str(vision_tower_type or "").lower()
    if vision_tower_type in ("dinov2", "dino_v2", "dino2"):
        return "dinov2"
    if vision_tower_type in ("dinov3", "dino_v3", "dino3"):
        return "dinov3"
    if vision_tower_type in (
        "dinov3_private_seg",
        "dinov3_private",
        "private_dinov3",
        "dinov3_seg",
    ):
        return "dinov3_private_seg"
    return ""


def _normalize_multi_vision_type(vision_tower_type):
    vision_tower_type = str(vision_tower_type or "").lower()
    if vision_tower_type in ("multi_moe", "multivision_moe", "multi_vision_moe", "dual_dino_moe", "dual_vision_moe"):
        return "multi_moe"
    if vision_tower_type in (
        "multi_concat",
        "multi_vision_concat",
        "dual_vision_concat",
        "prismatic_concat",
        "dino_siglip_concat",
        "dinov2_siglip_concat",
        "dinov3_siglip_concat",
    ):
        return "multi_concat"
    return ""


def _normalize_siglip_type(vision_tower_type):
    vision_tower_type = str(vision_tower_type or "").lower()
    if vision_tower_type in ("siglip", "siglip2", "siglip_vision"):
        return "siglip"
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


def _build_single_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    if vision_tower is None:
        raise ValueError("Missing vision tower path")
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    vision_tower_str = str(vision_tower)
    vision_tower_lower = vision_tower_str.lower()
    img_size = getattr(vision_tower_cfg, 'input_image_size', None) or None

    vit_type = _normalize_dino_type(getattr(vision_tower_cfg, 'mm_vision_tower_type', ''))
    if vit_type in ("dinov2", "dinov3", "dinov3_private_seg"):
        if "dinov2" in vision_tower_lower or "dinov3" in vision_tower_lower:
            try:
                dino_cfg = get_dino_config(vision_tower_str, input_image_size=img_size)
                if vit_type == "dinov3_private_seg":
                    if getattr(vision_tower_cfg, 'input_image_size', None) is None:
                        vision_tower_cfg.input_image_size = dino_cfg.image_size
                    vision_tower_cfg.mm_vision_tower_type = "dinov3_private_seg"
                    vision_tower_cfg.deepstack_visual_indexes = None
                else:
                    _apply_dino_config(vision_tower_cfg, dino_cfg)
            except KeyError:
                vision_tower_cfg.mm_vision_tower_type = vit_type
        else:
            vision_tower_cfg.mm_vision_tower_type = vit_type

    if vit_type == "dinov3_private_seg":
        vision_tower_cfg.mm_vision_tower_type = "dinov3_private_seg"
        vision_tower_cfg.deepstack_visual_indexes = None
        return _build_dinov3_private_seg_vision_tower(vision_tower, vision_tower_cfg, **kwargs)
    if vit_type == "dinov3":
        return _build_dinov3_vision_tower(vision_tower, vision_tower_cfg, **kwargs)
    if vit_type == "dinov2":
        return DINOv2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if _normalize_siglip_type(getattr(vision_tower_cfg, 'mm_vision_tower_type', '')):
        vision_tower_cfg.mm_vision_tower_type = "siglip"
        return _build_siglip_vision_tower(vision_tower, vision_tower_cfg, **kwargs)

    if "dinov3" in vision_tower_lower or "dinov2" in vision_tower_lower:
        dino_cfg = get_dino_config(vision_tower_str, input_image_size=img_size)
        _apply_dino_config(vision_tower_cfg, dino_cfg)
        if dino_cfg.encoder_type == "dinov3":
            return _build_dinov3_vision_tower(vision_tower, vision_tower_cfg, **kwargs)
        return DINOv2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if "dinov" in vision_tower_lower:
        raise ValueError(
            f"Ambiguous DINO vision tower '{vision_tower}'. "
            "Set mm_vision_tower_type to dinov2/dinov3 or use a vision_tower path "
            "whose folder name contains an exact DINO variant such as dinov3-vitl16."
        )

    if "siglip" in vision_tower_lower:
        vision_tower_cfg.mm_vision_tower_type = "siglip"
        return _build_siglip_vision_tower(vision_tower, vision_tower_cfg, **kwargs)

    if is_absolute_path_exists or vision_tower_str.startswith("openai") or vision_tower_str.startswith("laion") or "ShareGPT4V" in vision_tower_str:
        return _build_clip_vision_tower(
            vision_tower,
            vision_tower_cfg,
            use_s2,
            **kwargs,
        )
    elif "mobileclip" in vision_tower.lower():
        return _build_mobileclip_vision_tower(vision_tower, vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower_type = getattr(vision_tower_cfg, 'mm_vision_tower_type', '')
    multi_vision_type = _normalize_multi_vision_type(vision_tower_type)
    if multi_vision_type:
        from .multi_moe_encoder import MultiVisionMoEVisionTower
        vision_tower_cfg.mm_vision_tower_type = multi_vision_type
        return MultiVisionMoEVisionTower(
            vision_tower_cfg,
            single_tower_builder=_build_single_vision_tower,
            **kwargs,
        )

    return _build_single_vision_tower(vision_tower_cfg, **kwargs)
