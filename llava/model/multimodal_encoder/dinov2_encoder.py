import torch
import torch.nn as nn

from transformers import AutoConfig, AutoImageProcessor, Dinov2Model

from .deepstack import DeepStack


class DINOv2VisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.tune_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
        self.input_image_size = getattr(args, 'input_image_size', None)

        self.deepstack_visual_indexes = getattr(args, 'deepstack_visual_indexes', None)
        self.deepstack = None

        if self.tune_vision_tower:
            print("DINOv2 vision tower is set to tunable")

        if not delay_load:
            self.load_model()
        elif self.tune_vision_tower:
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name, local_files_only=True)
            if self.input_image_size is not None:
                self.cfg_only.image_size = self.input_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, `load_model` called again, skipping.')
            return

        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name, local_files_only=True)
        self.vision_tower = Dinov2Model.from_pretrained(
            self.vision_tower_name,
            device_map=device_map,
            local_files_only=True,
        )
        if not self.tune_vision_tower:
            self.vision_tower.requires_grad_(False)

        target_size = self.input_image_size or self.vision_tower.config.image_size
        if target_size is not None:
            print(f"Using DINOv2 input image size: {target_size}")
            self.image_processor.size = {"shortest_edge": target_size}
            self.image_processor.crop_size = {"height": target_size, "width": target_size}

        if self.deepstack_visual_indexes is not None:
            self._build_deepstack()

        self.cfg_only = self.vision_tower.config
        self.is_loaded = True

    def _build_deepstack(self):
        num_layers = len(self.deepstack_visual_indexes)
        hidden_size = self.vision_tower.config.hidden_size
        self.deepstack = DeepStack(hidden_size, num_layers)
        print(f"DeepStack enabled: layers={self.deepstack_visual_indexes}, "
              f"num_selected={num_layers}, hidden_size={hidden_size}")

    def feature_select(self, image_forward_outs):
        if self.deepstack is not None:
            hidden_states = image_forward_outs.hidden_states
            selected = [hidden_states[i] for i in self.deepstack_visual_indexes]
            if self.select_feature == 'patch':
                selected = [hs[:, 1:] for hs in selected]
            elif self.select_feature == 'cls_patch':
                pass
            else:
                raise ValueError(f'Unexpected select feature: {self.select_feature}')
            return self.deepstack(selected)

        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images):
        if self.tune_vision_tower:
            return self.forward_images(images)
        with torch.no_grad():
            return self.forward_images(images)

    def forward_images(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

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
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
