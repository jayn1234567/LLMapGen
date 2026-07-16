import os
from contextlib import nullcontext

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor, AutoModel
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from .dinov3_checkpoint import (
    apply_scripted_dinov3_processor_stats,
    load_dinov3_vision_checkpoint,
    _tensor_state_dict,
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
            "[DINOv3PrivateSegVisionTower] loading DINOv3 with the LLMapGen/SatelliteEncoder "
            f"from_pretrained path before private checkpoint: base={self.vision_tower_name} "
            f"checkpoint={checkpoint}",
            flush=True,
        )
        self.vision_tower = self._load_nonempty_base_tower(
            device_map=device_map,
            config=config,
            checkpoint=checkpoint,
            target_size=target_size,
        )
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

    @staticmethod
    def _state_nonempty_summary(module: nn.Module) -> tuple[int, int, list[str]]:
        nonempty_tensors = 0
        nonempty_params = 0
        examples = []
        for key, value in module.state_dict().items():
            if int(value.numel()) <= 0:
                continue
            nonempty_tensors += 1
            nonempty_params += int(value.numel())
            if len(examples) < 5:
                examples.append(f"{key}:{tuple(value.shape)}")
        return nonempty_tensors, nonempty_params, examples

    @staticmethod
    def _clone_dinov3_config(config) -> DINOv3ViTConfig:
        if config is None:
            return DINOv3ViTConfig()
        if isinstance(config, DINOv3ViTConfig):
            config_dict = config.to_dict()
        elif hasattr(config, "to_dict"):
            config_dict = config.to_dict()
        else:
            config_dict = dict(getattr(config, "__dict__", {}))
        config_dict.pop("_name_or_path", None)
        return DINOv3ViTConfig(**config_dict)

    @staticmethod
    def _infer_checkpoint_config_overrides(checkpoint: str) -> dict:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        raw_state = _tensor_state_dict(state)
        if not raw_state:
            raise ValueError(f"Unable to inspect DINOv3 checkpoint tensors: {checkpoint}")

        cls_token = raw_state.get("encoder.cls_token")
        patch_weight = raw_state.get("encoder.patch_embed.proj.weight")
        fc1_weight = raw_state.get("encoder.blocks.0.mlp.fc1.weight")
        storage_tokens = raw_state.get("encoder.storage_tokens")
        block_indexes = sorted(
            {
                int(str(key).split(".")[2])
                for key in raw_state
                if str(key).startswith("encoder.blocks.")
                and len(str(key).split(".")) > 3
                and str(key).split(".")[2].isdigit()
            }
        )

        overrides = {}
        if torch.is_tensor(cls_token) and cls_token.ndim >= 1:
            overrides["hidden_size"] = int(cls_token.shape[-1])
        if torch.is_tensor(patch_weight) and patch_weight.ndim == 4:
            overrides["patch_size"] = int(patch_weight.shape[-1])
            overrides["num_channels"] = int(patch_weight.shape[1])
            overrides.setdefault("hidden_size", int(patch_weight.shape[0]))
        if torch.is_tensor(fc1_weight) and fc1_weight.ndim == 2:
            overrides["intermediate_size"] = int(fc1_weight.shape[0])
            overrides.setdefault("hidden_size", int(fc1_weight.shape[1]))
        if torch.is_tensor(storage_tokens) and storage_tokens.ndim >= 2:
            overrides["num_register_tokens"] = int(storage_tokens.shape[-2])
        if block_indexes:
            overrides["num_hidden_layers"] = int(max(block_indexes) + 1)
        hidden_size = int(overrides.get("hidden_size", 1024))
        overrides["num_attention_heads"] = int(
            os.environ.get("DINOV3_PRIVATE_SEG_NUM_HEADS", max(1, hidden_size // 64))
        )
        return overrides

    def _build_tower_from_config(self, config, target_size: int) -> nn.Module:
        cfg = self._clone_dinov3_config(config)
        cfg.image_size = int(target_size)
        return DINOv3ViTModel(cfg)

    def _build_tower_from_checkpoint_config(self, config, checkpoint: str, target_size: int) -> nn.Module:
        cfg = self._clone_dinov3_config(config)
        overrides = self._infer_checkpoint_config_overrides(checkpoint)
        for key, value in overrides.items():
            setattr(cfg, key, value)
        cfg.image_size = int(target_size)
        print(
            "[DINOv3PrivateSegVisionTower] checkpoint-derived config "
            f"hidden={getattr(cfg, 'hidden_size', None)} "
            f"layers={getattr(cfg, 'num_hidden_layers', None)} "
            f"heads={getattr(cfg, 'num_attention_heads', None)} "
            f"intermediate={getattr(cfg, 'intermediate_size', None)} "
            f"registers={getattr(cfg, 'num_register_tokens', None)} "
            f"patch={getattr(cfg, 'patch_size', None)} image_size={cfg.image_size}",
            flush=True,
        )
        return DINOv3ViTModel(cfg)

    def _load_nonempty_base_tower(self, device_map=None, config=None, checkpoint=None, target_size=512) -> nn.Module:
        load_errors = []
        loaders = (
            (
                "AutoModel.from_pretrained",
                lambda: AutoModel.from_pretrained(
                    self.vision_tower_name,
                    device_map=device_map,
                    local_files_only=True,
                    trust_remote_code=True,
                    torch_dtype="auto",
                    low_cpu_mem_usage=False,
                ),
            ),
            (
                "DINOv3ViTModel.from_pretrained",
                lambda: DINOv3ViTModel.from_pretrained(
                    self.vision_tower_name,
                    device_map=device_map,
                    local_files_only=True,
                    low_cpu_mem_usage=False,
                ),
            ),
            (
                "DINOv3ViTModel.from_config",
                lambda: self._build_tower_from_config(config, int(target_size or 512)),
            ),
            (
                "DINOv3ViTModel.from_checkpoint_config",
                lambda: self._build_tower_from_checkpoint_config(
                    config,
                    str(checkpoint),
                    int(target_size or 512),
                ),
            ),
        )
        for name, loader in loaders:
            try:
                tower = loader()
            except Exception as exc:
                load_errors.append(f"{name}: {exc!r}")
                continue
            tensors, params, examples = self._state_nonempty_summary(tower)
            print(
                "[DINOv3PrivateSegVisionTower] "
                f"{name} nonempty_tensors={tensors} nonempty_params={params} "
                f"examples={examples}",
                flush=True,
            )
            if tensors >= 300 and params >= 100_000_000:
                return tower
            load_errors.append(
                f"{name}: nonempty_tensors={tensors} nonempty_params={params} examples={examples}"
            )
            del tower
        raise ValueError(
            "Unable to build a non-empty DINOv3 base tower before loading dinov3_lora.pt. "
            "This means the base model directory/model class is still producing empty "
            f"placeholder parameters. Tried: {load_errors}"
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
