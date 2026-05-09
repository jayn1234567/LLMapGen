import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .dinov2_encoder import DINOv2VisionTower
from .dinov3_encoder import DINOv3VisionTower
from .mobileclip_encoder import MobileCLIPVisionTower
from .dino_config import get_dino_config


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)

    if "dinov" in vision_tower.lower():
        try:
            img_size = getattr(vision_tower_cfg, 'input_image_size', None) or None
            variant = getattr(vision_tower_cfg, 'dino_variant', None) or None
            dino_cfg = get_dino_config(vision_tower, input_image_size=img_size, variant=variant)
            if getattr(vision_tower_cfg, 'input_image_size', None) is None:
                vision_tower_cfg.input_image_size = dino_cfg.image_size
            if getattr(vision_tower_cfg, 'deepstack_visual_indexes', None) is None:
                vision_tower_cfg.deepstack_visual_indexes = dino_cfg.deepstack_visual_indexes
        except KeyError:
            pass

    if "dinov3" in vision_tower.lower():
        return DINOv3VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "dinov2" in vision_tower.lower():
        return DINOv2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "mobileclip" in vision_tower.lower():
        return MobileCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
