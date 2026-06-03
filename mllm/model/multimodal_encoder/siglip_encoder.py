from contextlib import nullcontext

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoImageProcessor, SiglipVisionModel


def _vision_config(config):
    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None:
        return vision_config
    return config


class SigLIPVisionTower(nn.Module):
    """SigLIP vision tower with the same interface as DINO/CLIP towers."""

    mm_vision_tower_type = "siglip"

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        self.tune_vision_tower = getattr(args, "unfreeze_mm_vision_tower", False)
        self.input_image_size = getattr(args, "input_image_size", None)

        self.deepstack_visual_indexes = None
        self.deepstack_mergers = None

        if self.tune_vision_tower:
            print("SigLIP vision tower is set to tunable")

        if not delay_load:
            self.load_model()
        else:
            try:
                config = AutoConfig.from_pretrained(
                    self.vision_tower_name,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                self.cfg_only = _vision_config(config)
            except (OSError, EnvironmentError):
                self.cfg_only = None
            if self.input_image_size is not None and self.cfg_only is not None:
                self.cfg_only.image_size = self.input_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f"{self.vision_tower_name} is already loaded, `load_model` called again, skipping.")
            return

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.vision_tower_name,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.vision_tower = SiglipVisionModel.from_pretrained(
            self.vision_tower_name,
            device_map=device_map,
            local_files_only=True,
        )
        if not self.tune_vision_tower:
            self.vision_tower.requires_grad_(False)

        target_size = self.input_image_size or getattr(self.vision_tower.config, "image_size", None)
        if target_size is not None:
            print(f"Using SigLIP input image size: {target_size}")
            if hasattr(self.image_processor, "size"):
                self.image_processor.size = {"height": target_size, "width": target_size}
            if hasattr(self.image_processor, "crop_size"):
                self.image_processor.crop_size = {"height": target_size, "width": target_size}
        self._target_size = target_size
        self.cfg_only = self.vision_tower.config
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        hidden_states = image_forward_outs.hidden_states
        image_features = hidden_states[self.select_layer]
        if self.select_feature in ("patch", "cls_patch"):
            return image_features
        raise ValueError(f"Unexpected select feature for SigLIP: {self.select_feature}")

    def forward(self, images):
        return self.forward_images(images, freeze_vision=not self.tune_vision_tower)

    def forward_images(self, images, freeze_vision=False):
        vision_context = torch.no_grad() if freeze_vision else nullcontext()
        if type(images) is list:
            image_features = []
            for image in images:
                with vision_context:
                    image_forward_out = self.vision_tower(
                        image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                        output_hidden_states=True,
                    )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
            return image_features[0] if len(image_features) == 1 else image_features

        with vision_context:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
        return self.feature_select(image_forward_outs)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        target_size = self._target_size if self.is_loaded else getattr(self.config, "image_size", None)
        return int(target_size) // int(self.config.patch_size)

    @property
    def num_patches(self):
        return self.num_patches_per_side ** 2
