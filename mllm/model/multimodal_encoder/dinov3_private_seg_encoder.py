import os
from contextlib import nullcontext

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor, AutoModel

from .dinov3_checkpoint import (
    apply_scripted_dinov3_processor_stats,
    load_dinov3_vision_checkpoint,
)


class DINOv3PrivateSegVisionTower(nn.Module):
    """DINOv3 tower for private segmentation-trained scripted checkpoints.

    This intentionally follows the LLMapGen SatelliteEncoder route: build the
    DINOv3 architecture from config, then load the full scripted checkpoint such
    as dinov3_lora.pt. The base HF/OBS model directory is used for config,
    processor and model code, not as the trusted source of weights.
    """

    mm_vision_tower_type = "dinov3_private_seg"

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = str(vision_tower)
        self.select_layer = int(args.mm_vision_select_layer)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        self.tune_vision_tower = bool(getattr(args, "unfreeze_mm_vision_tower", False))
        self.unfreeze_last_n_blocks = int(getattr(args, "mm_vision_unfreeze_last_n_blocks", -1))
        self.input_image_size = getattr(args, "input_image_size", None)
        self.vision_tower_checkpoint = (
            getattr(args, "vision_tower_checkpoint", None)
            or getattr(args, "mm_vision_tower_checkpoint", None)
            or ""
        )
        self.deepstack_visual_indexes = None
        self.deepstack_mergers = None
        self.vision_layer_fusion = None

        if not delay_load:
            self.load_model()
        else:
            try:
                self.cfg_only = AutoConfig.from_pretrained(
                    self.vision_tower_name,
                    local_files_only=True,
                    trust_remote_code=True,
                )
            except (OSError, EnvironmentError):
                self.cfg_only = None
            if self.input_image_size is not None and self.cfg_only is not None:
                self.cfg_only.image_size = int(self.input_image_size)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f"{self.vision_tower_name} is already loaded, `load_model` called again, skipping.")
            return
        if device_map is not None:
            print(
                "[DINOv3PrivateSegVisionTower] device_map is ignored while building from config; "
                "Trainer/Accelerate will place the module.",
                flush=True,
            )

        checkpoint = str(self.vision_tower_checkpoint or "").strip()
        if not checkpoint:
            raise ValueError(
                "DINOv3PrivateSegVisionTower requires --vision_tower_checkpoint "
                "pointing to the private segmentation checkpoint, e.g. dinov3_lora.pt."
            )

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.vision_tower_name,
            local_files_only=True,
            trust_remote_code=True,
        )
        config = AutoConfig.from_pretrained(
            self.vision_tower_name,
            local_files_only=True,
            trust_remote_code=True,
        )
        target_size = int(self.input_image_size or getattr(config, "image_size", 512) or 512)
        config.image_size = target_size

        print(
            "[DINOv3PrivateSegVisionTower] building DINOv3 from config before "
            f"loading private segmentation checkpoint: base={self.vision_tower_name} "
            f"checkpoint={checkpoint}",
            flush=True,
        )
        self.vision_tower = AutoModel.from_config(config, trust_remote_code=True)
        selected_name = load_dinov3_vision_checkpoint(self.vision_tower, checkpoint)
        if selected_name == "scripted_dinov3_hf":
            apply_scripted_dinov3_processor_stats(self.image_processor)

        self.set_vision_tower_trainable(self.tune_vision_tower)
        self._target_size = target_size
        if hasattr(self.image_processor, "size"):
            self.image_processor.size = {"shortest_edge": target_size}
        if hasattr(self.image_processor, "crop_size"):
            self.image_processor.crop_size = {"height": target_size, "width": target_size}

        self.num_layers = self._infer_num_layers()
        self.num_register_tokens = self._infer_num_register_tokens()
        self.skip_tokens = 1 + self.num_register_tokens
        self._resolve_select_layer_index()
        self.cfg_only = self.vision_tower.config
        self.is_loaded = True
        print(
            "[DINOv3PrivateSegVisionTower] ready "
            f"hidden={self.hidden_size} layers={self.num_layers} "
            f"prefix_tokens={self.skip_tokens} input_size={self._target_size}",
            flush=True,
        )

    def _find_vision_blocks(self, raise_on_missing=True):
        candidates = (
            ("encoder", "layer"),
            ("encoder", "layers"),
            ("encoder", "blocks"),
            ("model", "layer"),
            ("model", "layers"),
            ("model", "blocks"),
            ("layer",),
            ("layers",),
            ("blocks",),
        )
        for path in candidates:
            obj = self.vision_tower
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        if raise_on_missing:
            raise ValueError("DINOv3 private segmentation tower blocks could not be found.")
        return None

    def _infer_num_layers(self):
        value = getattr(getattr(self.vision_tower, "config", None), "num_hidden_layers", None)
        if value is not None:
            return int(value)
        blocks = self._find_vision_blocks(raise_on_missing=False)
        return len(blocks) if blocks is not None else 0

    def _infer_num_register_tokens(self):
        config = getattr(self.vision_tower, "config", None)
        for name in ("num_register_tokens", "num_registers", "num_reg_tokens"):
            value = getattr(config, name, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        embeddings = getattr(self.vision_tower, "embeddings", None)
        reg_tokens = getattr(embeddings, "register_tokens", None)
        if torch.is_tensor(reg_tokens):
            if reg_tokens.ndim >= 2:
                return int(reg_tokens.shape[-2])
            if reg_tokens.ndim == 1:
                return 1
        return 0

    def set_vision_tower_trainable(self, trainable, last_n_blocks=None):
        if last_n_blocks is not None:
            self.unfreeze_last_n_blocks = int(last_n_blocks)
        trainable = bool(trainable)
        if not trainable:
            self.vision_tower.requires_grad_(False)
            self.tune_vision_tower = False
            self.trainable_vision_block_indices = []
            return

        blocks = self._find_vision_blocks(raise_on_missing=False)
        if self.unfreeze_last_n_blocks < 0:
            self.vision_tower.requires_grad_(True)
            self.tune_vision_tower = True
            self.trainable_vision_block_indices = list(range(len(blocks))) if blocks is not None else ["all"]
            return
        if blocks is None:
            raise ValueError("Cannot apply partial DINOv3 training because blocks were not found.")
        if self.unfreeze_last_n_blocks > len(blocks):
            raise ValueError(
                f"mm_vision_unfreeze_last_n_blocks={self.unfreeze_last_n_blocks} "
                f"exceeds DINOv3 depth {len(blocks)}."
            )
        self.vision_tower.requires_grad_(False)
        start = len(blocks) - self.unfreeze_last_n_blocks
        for block in blocks[start:]:
            block.requires_grad_(True)
        final_norm = getattr(self.vision_tower, "layernorm", None) or getattr(self.vision_tower, "norm", None)
        if self.unfreeze_last_n_blocks > 0 and final_norm is not None:
            final_norm.requires_grad_(True)
        self.trainable_vision_block_indices = list(range(start, len(blocks)))
        self.tune_vision_tower = bool(self.trainable_vision_block_indices)
        print(
            "DINOv3 private segmentation partial fine-tuning: "
            f"blocks={self.trainable_vision_block_indices}, "
            f"final_norm={self.unfreeze_last_n_blocks > 0}",
            flush=True,
        )

    def _resolve_select_layer_index(self):
        raw = int(self.select_layer)
        if raw >= 0:
            self.select_layer_idx = raw
            max_index = self.num_layers
        else:
            self.select_layer_idx = self.num_layers + raw
            max_index = self.num_layers - 1
        self.select_layer_idx = max(0, min(self.select_layer_idx, max_index))

    def set_llm_hidden_size(self, llm_hidden_size):
        return None

    def feature_select(self, image_forward_outs):
        hidden_states = image_forward_outs.hidden_states
        if not hidden_states:
            raise ValueError("DINOv3 private segmentation tower did not return hidden_states.")
        layer_idx = max(0, min(self.select_layer_idx, len(hidden_states) - 1))
        if layer_idx == len(hidden_states) - 1:
            features = image_forward_outs.last_hidden_state
        else:
            features = hidden_states[layer_idx]
        if self.select_feature == "patch":
            return features[:, self.skip_tokens:], None
        if self.select_feature == "cls_patch":
            return features, None
        raise ValueError(f"Unexpected select feature: {self.select_feature}")

    def forward(self, images):
        return self.forward_images(images, freeze_vision=not self.tune_vision_tower)

    def forward_images(self, images, freeze_vision=False):
        vision_context = torch.no_grad() if freeze_vision else nullcontext()
        if type(images) is list:
            main_features = []
            for image in images:
                with vision_context:
                    out = self.vision_tower(
                        pixel_values=image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                        output_hidden_states=True,
                    )
                mf, _ = self.feature_select(out)
                main_features.append(mf.to(image.dtype))
            return main_features[0] if len(main_features) == 1 else main_features, None

        with vision_context:
            out = self.vision_tower(
                pixel_values=images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
        return self.feature_select(out)

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
        return int(getattr(self.config, "hidden_size"))

    @property
    def num_patches_per_side(self):
        return int(self._target_size) // int(getattr(self.config, "patch_size", 16))

    @property
    def num_patches(self):
        return self.num_patches_per_side ** 2
