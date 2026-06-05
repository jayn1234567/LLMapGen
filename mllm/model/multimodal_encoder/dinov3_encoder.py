import os
from contextlib import nullcontext

import torch
import torch.nn as nn

from transformers import AutoImageProcessor
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from .deepstack import build_deepstack_mergers
from .visual_layer_fusion import VisualLayerFusion


class DINOv3VisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.tune_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
        self.input_image_size = getattr(args, 'input_image_size', None)

        self.deepstack_visual_indexes = getattr(args, 'deepstack_visual_indexes', None)
        self.deepstack_mergers = None
        self.vision_layer_fusion_indexes = getattr(args, 'vision_layer_fusion_indexes', None)
        self.vision_layer_fusion_type = getattr(args, 'vision_layer_fusion_type', 'mean')
        self.vision_layer_fusion = None
        self._preferred_dtype = torch.bfloat16

        if self.tune_vision_tower:
            print("DINOv3 vision tower is set to tunable")

        if not delay_load:
            self.load_model()
        else:
            try:
                self.cfg_only = DINOv3ViTConfig.from_pretrained(self.vision_tower_name, local_files_only=True)
            except (OSError, EnvironmentError):
                self.cfg_only = None
            if self.input_image_size is not None and self.cfg_only is not None:
                self.cfg_only.image_size = self.input_image_size

    def _stable_dtype_for_device(self):
        if not hasattr(self, "vision_tower"):
            return self._preferred_dtype
        device = next(self.vision_tower.parameters()).device
        if device.type in ("cuda", "npu"):
            return torch.bfloat16
        return torch.float32

    def _keep_stable_dtype(self):
        if not hasattr(self, "vision_tower"):
            return
        dtype = next(self.vision_tower.parameters()).dtype
        if dtype == torch.float16:
            stable_dtype = self._stable_dtype_for_device()
            self.vision_tower.to(dtype=stable_dtype)
            if self.deepstack_mergers is not None:
                self.deepstack_mergers.to(dtype=stable_dtype)
            if self.vision_layer_fusion is not None:
                self.vision_layer_fusion.to(dtype=stable_dtype)

    def _apply(self, fn):
        module = super()._apply(fn)
        self._keep_stable_dtype()
        return module

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, `load_model` called again, skipping.')
            return

        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name, local_files_only=True)
        self.vision_tower = DINOv3ViTModel.from_pretrained(
            self.vision_tower_name,
            device_map=device_map,
            local_files_only=True,
        )
        if not self.tune_vision_tower:
            self.vision_tower.requires_grad_(False)

        target_size = self.input_image_size or self.vision_tower.config.image_size
        if target_size is not None:
            print(f"Using DINOv3 input image size: {target_size}")
            if hasattr(self.image_processor, 'size'):
                self.image_processor.size = {"shortest_edge": target_size}
            if hasattr(self.image_processor, 'crop_size'):
                self.image_processor.crop_size = {"height": target_size, "width": target_size}

        self.num_layers = self.vision_tower.config.num_hidden_layers
        self.num_register_tokens = self.vision_tower.config.num_register_tokens
        self.skip_tokens = 1 + self.num_register_tokens  # CLS + register
        self._target_size = target_size
        self._resolve_select_layer_index()

        if self.deepstack_visual_indexes is not None:
            self._build_deepstack()
        if self.vision_layer_fusion_indexes is not None:
            self._build_vision_layer_fusion()

        self.cfg_only = self.vision_tower.config
        self.is_loaded = True
        self._keep_stable_dtype()

    def load_model_from_checkpoint(self, checkpoint_dir):
        vit_config_path = os.path.join(checkpoint_dir, 'vit_config.json')
        if not os.path.isfile(vit_config_path):
            raise FileNotFoundError(f"vit_config.json not found in {checkpoint_dir}")
        vit_config = DINOv3ViTConfig.from_pretrained(vit_config_path)

        self.image_processor = AutoImageProcessor.from_pretrained(checkpoint_dir, local_files_only=True)
        self.vision_tower = DINOv3ViTModel(vit_config)
        if not self.tune_vision_tower:
            self.vision_tower.requires_grad_(False)

        target_size = self.input_image_size or self.vision_tower.config.image_size
        if target_size is not None:
            print(f"Using DINOv3 input image size (from checkpoint): {target_size}")
            if hasattr(self.image_processor, 'size'):
                self.image_processor.size = {"shortest_edge": target_size}
            if hasattr(self.image_processor, 'crop_size'):
                self.image_processor.crop_size = {"height": target_size, "width": target_size}

        self.num_layers = self.vision_tower.config.num_hidden_layers
        self.num_register_tokens = self.vision_tower.config.num_register_tokens
        self.skip_tokens = 1 + self.num_register_tokens
        self._target_size = target_size
        self._resolve_select_layer_index()

        if self.deepstack_visual_indexes is not None:
            self._build_deepstack()
        if self.vision_layer_fusion_indexes is not None:
            self._build_vision_layer_fusion()

        self.cfg_only = self.vision_tower.config
        self.is_loaded = True
        self._keep_stable_dtype()

    def _resolve_select_layer_index(self):
        raw = self.select_layer
        if raw >= 0:
            self.select_layer_idx = raw
        else:
            self.select_layer_idx = self.num_layers + raw
        self.select_layer_idx = max(0, min(self.select_layer_idx, self.num_layers - 1))

    def _build_deepstack(self):
        vit_hidden_size = self.vision_tower.config.hidden_size
        self.deepstack_mergers = build_deepstack_mergers(
            vit_hidden_size=vit_hidden_size,
            llm_hidden_size=vit_hidden_size,
            num_mergers=len(self.deepstack_visual_indexes),
        )
        print(f"DeepStack (real injection) enabled: ViT layers={self.deepstack_visual_indexes}, "
              f"num={len(self.deepstack_visual_indexes)}, main_layer={self.select_layer_idx}")

    def _build_vision_layer_fusion(self):
        vit_hidden_size = self.vision_tower.config.hidden_size
        self.vision_layer_fusion = VisualLayerFusion(
            hidden_size=vit_hidden_size,
            num_layers=len(self.vision_layer_fusion_indexes),
            fusion_type=self.vision_layer_fusion_type,
        )
        print(
            "Vision layer fusion enabled: "
            f"type={self.vision_layer_fusion_type}, "
            f"ViT layers={self.vision_layer_fusion_indexes}, "
            f"main_layer_replaced={self.select_layer_idx}"
        )

    def set_llm_hidden_size(self, llm_hidden_size):
        if self.deepstack_mergers is not None:
            vit_hidden_size = self.vision_tower.config.hidden_size
            self.deepstack_mergers = build_deepstack_mergers(
                vit_hidden_size=vit_hidden_size,
                llm_hidden_size=llm_hidden_size,
                num_mergers=len(self.deepstack_visual_indexes),
            )
            self._keep_stable_dtype()

    def feature_select(self, image_forward_outs):
        hidden_states = image_forward_outs.hidden_states

        def select_features_from_layer(layer_idx):
            layer_idx = max(0, min(layer_idx, len(hidden_states) - 1))
            features = hidden_states[layer_idx]
            if self.select_feature == 'patch':
                return features[:, self.skip_tokens:]
            if self.select_feature == 'cls_patch':
                return features
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        if self.vision_layer_fusion is not None:
            fusion_features = [
                select_features_from_layer(idx)
                for idx in self.vision_layer_fusion_indexes
            ]
            main_features = self.vision_layer_fusion(fusion_features)
        else:
            main_features = select_features_from_layer(self.select_layer_idx)
        if not torch.isfinite(main_features).all():
            raise RuntimeError(
                "DINOv3 produced non-finite visual features. "
                "DINOv3 should run in bfloat16 or float32, not float16."
            )

        if self.deepstack_mergers is not None:
            deepstack_features = []
            for i, idx in enumerate(self.deepstack_visual_indexes):
                hs = select_features_from_layer(idx)
                deepstack_features.append(self.deepstack_mergers[i](hs))
            return main_features, deepstack_features

        return main_features, None

    def forward(self, images):
        return self.forward_images(images, freeze_vision=not self.tune_vision_tower)

    def forward_images(self, images, freeze_vision=False):
        self._keep_stable_dtype()
        vision_context = torch.no_grad() if freeze_vision else nullcontext()
        if type(images) is list:
            main_features = []
            deepstack_features = None
            for image in images:
                with vision_context:
                    image_forward_out = self.vision_tower(
                        image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                        output_hidden_states=True,
                    )
                mf, df = self.feature_select(image_forward_out)
                mf = mf.to(image.dtype)
                main_features.append(mf)
                if df is not None:
                    if deepstack_features is None:
                        deepstack_features = [[] for _ in range(len(df))]
                    for j, d in enumerate(df):
                        deepstack_features[j].append(d.to(image.dtype))
            if deepstack_features is not None:
                deepstack_features = [torch.cat(dlist, dim=0) for dlist in deepstack_features]
            return main_features[0] if len(main_features) == 1 else main_features, deepstack_features

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
        return self._target_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self._target_size // self.config.patch_size) ** 2
