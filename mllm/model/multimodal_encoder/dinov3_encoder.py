import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from transformers import AutoImageProcessor
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from .deepstack import build_deepstack_mergers
from .visual_layer_fusion import VisualLayerFusion


def _tensor_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if hasattr(payload, "state_dict") and not isinstance(payload, dict):
        try:
            module_state = payload.state_dict()
        except Exception:
            module_state = None
        if isinstance(module_state, dict):
            return {str(key): value for key, value in module_state.items() if torch.is_tensor(value)}
    if not isinstance(payload, dict):
        return {}
    direct = {str(key): value for key, value in payload.items() if torch.is_tensor(value)}
    if direct:
        return direct
    for key in ("vision_tower", "vision_encoder", "encoder", "backbone", "model", "state_dict", "module", "net"):
        nested_state = _tensor_state_dict(payload.get(key))
        if nested_state:
            return nested_state
    return {}


def _strip_common_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = (
        "module.",
        "_orig_mod.",
        "base_model.model.",
        "base_model.",
        "vision_tower.",
        "vision_encoder.",
        "encoder.",
        "backbone.",
        "net.",
    )
    state = dict(state_dict)
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if state and all(str(key).startswith(prefix) for key in state):
                state = {str(key)[len(prefix):]: value for key, value in state.items()}
                changed = True
                break
    return state


def _find_target_key(target_state: Dict[str, torch.Tensor], value: torch.Tensor, suffixes: Sequence[str]) -> str | None:
    value_shape = tuple(value.shape)
    for suffix in suffixes:
        matches = [
            key
            for key, target_value in target_state.items()
            if str(key).endswith(str(suffix)) and _logical_shape(target_value) == value_shape
        ]
        if matches:
            return sorted(matches, key=len)[0]
    return None


def _logical_shape(value: torch.Tensor) -> tuple[int, ...]:
    ds_shape = getattr(value, "ds_shape", None)
    if ds_shape is not None:
        return tuple(int(dim) for dim in ds_shape)
    return tuple(value.shape)


def _module_target_tensors(module: nn.Module) -> Dict[str, torch.Tensor]:
    targets: Dict[str, torch.Tensor] = {}
    targets.update({name: param for name, param in module.named_parameters()})
    targets.update({name: buffer for name, buffer in module.named_buffers()})
    if targets:
        return targets
    return module.state_dict()


def _distributed_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


def _copy_checkpoint_tensor(target: torch.Tensor, value: torch.Tensor, key: str) -> None:
    if _logical_shape(target) != tuple(value.shape):
        raise RuntimeError(
            f"Shape mismatch for {key}: checkpoint={tuple(value.shape)} "
            f"target_logical={_logical_shape(target)} target_local={tuple(target.shape)}"
        )

    def _copy_into_target() -> None:
        source = value.detach().to(device=target.device, dtype=target.dtype)
        target.data.copy_(source)

    if isinstance(target, nn.Parameter) and hasattr(target, "ds_id"):
        try:
            import deepspeed

            with deepspeed.zero.GatheredParameters([target], modifier_rank=0):
                if _distributed_rank() == 0:
                    _copy_into_target()
            return
        except Exception as exc:
            if tuple(target.shape) != tuple(value.shape):
                raise RuntimeError(
                    f"Failed to load ZeRO-partitioned DINOv3 parameter {key}. "
                    "This usually means DeepSpeed is active but the parameter could not be gathered."
                ) from exc

    if tuple(target.shape) != tuple(value.shape):
        raise RuntimeError(
            f"Cannot load {key}: checkpoint={tuple(value.shape)} "
            f"target_local={tuple(target.shape)} target_logical={_logical_shape(target)}. "
            "If this is a DeepSpeed ZeRO-3 run, install/import deepspeed before loading the vision checkpoint."
        )
    _copy_into_target()


def _load_state_dict_zero_aware(
    module: nn.Module,
    filtered: Dict[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    target_state = _module_target_tensors(module)
    loaded_keys = set()
    with torch.no_grad():
        for key, value in filtered.items():
            target = target_state.get(key)
            if target is None:
                continue
            _copy_checkpoint_tensor(target, value, key)
            loaded_keys.add(key)

    missing = [key for key in target_state if key not in loaded_keys]
    unexpected = [key for key in filtered if key not in target_state]
    return missing, unexpected


def _assign_by_suffix(
    mapped: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
    value: Optional[torch.Tensor],
    suffixes: Sequence[str],
) -> bool:
    if value is None or not torch.is_tensor(value):
        return False
    target_key = _find_target_key(target_state, value, suffixes)
    if target_key is None:
        return False
    mapped[target_key] = value
    return True


def _assign_with_optional_unsqueeze(
    mapped: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
    value: Optional[torch.Tensor],
    suffixes: Sequence[str],
) -> bool:
    if value is None or not torch.is_tensor(value):
        return False
    if _assign_by_suffix(mapped, target_state, value, suffixes):
        return True
    if value.ndim == 2:
        return _assign_by_suffix(mapped, target_state, value.unsqueeze(1), suffixes)
    return False


def _suffix_aligned_state(module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    target_state = _module_target_tensors(module)
    source_by_suffix: Dict[str, tuple[str, torch.Tensor]] = {}
    for source_key, value in state_dict.items():
        parts = str(source_key).split(".")
        for start in range(len(parts)):
            suffix = ".".join(parts[start:])
            target = target_state.get(suffix)
            if target is not None and _logical_shape(target) == tuple(value.shape):
                existing = source_by_suffix.get(suffix)
                if existing is None or len(str(source_key)) < len(existing[0]):
                    source_by_suffix[suffix] = (str(source_key), value)
    return {target_key: value for target_key, (_, value) in source_by_suffix.items()}


def _scripted_dinov3_to_hf_state(
    vision_encoder: nn.Module,
    raw_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if not any(str(key).startswith("encoder.blocks.") for key in raw_state):
        return {}
    target_state = _module_target_tensors(vision_encoder)
    mapped: Dict[str, torch.Tensor] = {}

    _assign_by_suffix(mapped, target_state, raw_state.get("encoder.cls_token"), ("embeddings.cls_token", "cls_token"))
    _assign_by_suffix(
        mapped,
        target_state,
        raw_state.get("encoder.storage_tokens"),
        ("embeddings.register_tokens", "register_tokens", "storage_tokens"),
    )
    _assign_with_optional_unsqueeze(
        mapped,
        target_state,
        raw_state.get("encoder.mask_token"),
        ("embeddings.mask_token", "mask_token"),
    )
    _assign_by_suffix(
        mapped,
        target_state,
        raw_state.get("encoder.patch_embed.proj.weight"),
        (
            "embeddings.patch_embeddings.weight",
            "embeddings.patch_embeddings.projection.weight",
            "patch_embed.proj.weight",
        ),
    )
    _assign_by_suffix(
        mapped,
        target_state,
        raw_state.get("encoder.patch_embed.proj.bias"),
        (
            "embeddings.patch_embeddings.bias",
            "embeddings.patch_embeddings.projection.bias",
            "patch_embed.proj.bias",
        ),
    )

    block_indexes = sorted(
        {
            int(str(key).split(".")[2])
            for key in raw_state
            if str(key).startswith("encoder.blocks.")
            and len(str(key).split(".")) > 3
            and str(key).split(".")[2].isdigit()
        }
    )
    for idx in block_indexes:
        src = f"encoder.blocks.{idx}"
        layer_suffixes = (
            f"layer.{idx}",
            f"model.layer.{idx}",
            f"encoder.layer.{idx}",
            f"encoder.layers.{idx}",
            f"encoder.blocks.{idx}",
            f"blocks.{idx}",
            f"layers.{idx}",
        )

        for norm_name in ("norm1", "norm2"):
            for attr in ("weight", "bias"):
                _assign_by_suffix(
                    mapped,
                    target_state,
                    raw_state.get(f"{src}.{norm_name}.{attr}"),
                    tuple(f"{layer}.{norm_name}.{attr}" for layer in layer_suffixes),
                )
        for mlp_name in ("fc1", "fc2"):
            target_mlp_names = ("up_proj", "fc1") if mlp_name == "fc1" else ("down_proj", "fc2")
            for attr in ("weight", "bias"):
                _assign_by_suffix(
                    mapped,
                    target_state,
                    raw_state.get(f"{src}.mlp.{mlp_name}.{attr}"),
                    tuple(
                        f"{layer}.mlp.{target_name}.{attr}"
                        for layer in layer_suffixes
                        for target_name in target_mlp_names
                    ),
                )
        for source_name, target_names in (
            ("ls1.gamma", ("layer_scale1.lambda1", "layer_scale1.gamma", "ls1.gamma")),
            ("ls2.gamma", ("layer_scale2.lambda1", "layer_scale2.gamma", "ls2.gamma")),
        ):
            _assign_by_suffix(
                mapped,
                target_state,
                raw_state.get(f"{src}.{source_name}"),
                tuple(f"{layer}.{target}" for layer in layer_suffixes for target in target_names),
            )
        for attr in ("weight", "bias"):
            _assign_by_suffix(
                mapped,
                target_state,
                raw_state.get(f"{src}.attn.proj.{attr}"),
                tuple(
                    f"{layer}.{target}.{attr}"
                    for layer in layer_suffixes
                    for target in ("attention.o_proj", "attention.output.dense", "attention.output.projection", "attn.proj")
                ),
            )

        qkv_weight = raw_state.get(f"{src}.attn.qkv.qkv.weight")
        qkv_bias = raw_state.get(f"{src}.attn.qkv.qkv.bias")
        if torch.is_tensor(qkv_weight) and qkv_weight.ndim == 2 and qkv_weight.shape[0] % 3 == 0:
            q_weight, k_weight, v_weight = torch.chunk(qkv_weight, 3, dim=0)
            q_delta_a = raw_state.get(f"{src}.attn.qkv.linear_a_q.weight")
            q_delta_b = raw_state.get(f"{src}.attn.qkv.linear_b_q.weight")
            v_delta_a = raw_state.get(f"{src}.attn.qkv.linear_a_v.weight")
            v_delta_b = raw_state.get(f"{src}.attn.qkv.linear_b_v.weight")
            if torch.is_tensor(q_delta_a) and torch.is_tensor(q_delta_b):
                q_delta = torch.matmul(q_delta_b.to(dtype=q_weight.dtype), q_delta_a.to(dtype=q_weight.dtype))
                if tuple(q_delta.shape) == tuple(q_weight.shape):
                    q_weight = q_weight + q_delta
            if torch.is_tensor(v_delta_a) and torch.is_tensor(v_delta_b):
                v_delta = torch.matmul(v_delta_b.to(dtype=v_weight.dtype), v_delta_a.to(dtype=v_weight.dtype))
                if tuple(v_delta.shape) == tuple(v_weight.shape):
                    v_weight = v_weight + v_delta
            for name, value in (("query", q_weight), ("key", k_weight), ("value", v_weight)):
                short_name = {"query": "q_proj", "key": "k_proj", "value": "v_proj"}[name]
                _assign_by_suffix(
                    mapped,
                    target_state,
                    value,
                    tuple(
                        f"{layer}.{target}.{name}.weight"
                        for layer in layer_suffixes
                        for target in ("attention.attention", "attention")
                    )
                    + tuple(f"{layer}.attention.{short_name}.weight" for layer in layer_suffixes)
                    + tuple(f"{layer}.attn.{name}.weight" for layer in layer_suffixes),
                )
        if torch.is_tensor(qkv_bias) and qkv_bias.ndim == 1 and qkv_bias.shape[0] % 3 == 0:
            q_bias, k_bias, v_bias = torch.chunk(qkv_bias, 3, dim=0)
            for name, value in (("query", q_bias), ("key", k_bias), ("value", v_bias)):
                short_name = {"query": "q_proj", "key": "k_proj", "value": "v_proj"}[name]
                _assign_by_suffix(
                    mapped,
                    target_state,
                    value,
                    tuple(
                        f"{layer}.{target}.{name}.bias"
                        for layer in layer_suffixes
                        for target in ("attention.attention", "attention")
                    )
                    + tuple(f"{layer}.attention.{short_name}.bias" for layer in layer_suffixes)
                    + tuple(f"{layer}.attn.{name}.bias" for layer in layer_suffixes),
                )

    for source_key, suffixes in (
        ("encoder.norm.weight", ("layernorm.weight", "norm.weight")),
        ("encoder.norm.bias", ("layernorm.bias", "norm.bias")),
    ):
        _assign_by_suffix(mapped, target_state, raw_state.get(source_key), suffixes)
    return mapped


def _shape_match_score(module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> tuple[int, int]:
    target_state = _module_target_tensors(module)
    matched = {
        key: value
        for key, value in state_dict.items()
        if key in target_state and torch.is_tensor(value) and _logical_shape(target_state[key]) == tuple(value.shape)
    }
    return len(matched), sum(int(value.numel()) for value in matched.values())


def _select_checkpoint_state(module: nn.Module, raw_state: Dict[str, torch.Tensor]) -> tuple[str, Dict[str, torch.Tensor]]:
    stripped = _strip_common_prefixes(raw_state)
    candidates = [
        ("raw", raw_state),
        ("stripped", stripped),
        ("suffix_aligned", _suffix_aligned_state(module, raw_state)),
        ("scripted_dinov3_hf", _scripted_dinov3_to_hf_state(module, raw_state)),
    ]
    best_name = ""
    best_state: Dict[str, torch.Tensor] = {}
    best_score = (0, 0)
    seen = set()
    for name, state in candidates:
        if not state:
            continue
        signature = tuple(sorted(state.keys()))
        if signature in seen:
            continue
        seen.add(signature)
        score = _shape_match_score(module, state)
        if score > best_score:
            best_name = name
            best_state = state
            best_score = score
    if best_score[0] <= 0:
        lora_keys = [key for key in raw_state if "lora" in str(key).lower()]
        hint = ""
        if lora_keys:
            hint = (
                " The checkpoint looks like a LoRA-only adapter. Merge it into the base "
                "DINOv3 weights first, or provide a checkpoint with full encoder weights."
            )
        target_keys = list(module.state_dict().keys())[:12]
        raw_keys = list(raw_state.keys())[:12]
        scripted_count = len(_scripted_dinov3_to_hf_state(module, raw_state))
        raise ValueError(
            "Unable to match any DINOv3 checkpoint weights by shape. "
            f"raw_keys_sample={raw_keys}; target_keys_sample={target_keys}; "
            f"scripted_dinov3_candidate_tensors={scripted_count}.{hint}"
        )
    return best_name, best_state


def _load_external_dinov3_checkpoint(module: nn.Module, checkpoint_path: str) -> str:
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {ckpt_path}")
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    raw_state = _tensor_state_dict(payload)
    if not raw_state:
        raise ValueError(f"Unable to extract tensor state_dict from DINOv3 checkpoint: {ckpt_path}")
    selected_name, selected_state = _select_checkpoint_state(module, raw_state)
    target_state = _module_target_tensors(module)
    filtered = {
        key: value
        for key, value in selected_state.items()
        if key in target_state and torch.is_tensor(value) and _logical_shape(target_state[key]) == tuple(value.shape)
    }
    missing, unexpected = _load_state_dict_zero_aware(module, filtered)
    print(
        "[DINOv3 checkpoint] "
        f"selected_state={selected_name} loaded={len(filtered)} "
        f"missing={len(missing)} unexpected={len(unexpected)} path={ckpt_path}",
        flush=True,
    )
    return selected_name


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
        self.vision_tower_checkpoint_path = getattr(args, 'vision_tower_checkpoint_path', None)
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
        self._load_external_checkpoint_if_configured()
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
        self._load_external_checkpoint_if_configured()
        self._keep_stable_dtype()

    def _load_external_checkpoint_if_configured(self):
        checkpoint_path = str(self.vision_tower_checkpoint_path or "").strip()
        if not checkpoint_path:
            return
        selected_name = _load_external_dinov3_checkpoint(self.vision_tower, checkpoint_path)
        if selected_name == "scripted_dinov3_hf":
            if hasattr(self.image_processor, "image_mean"):
                self.image_processor.image_mean = [0.5, 0.5, 0.5]
            if hasattr(self.image_processor, "image_std"):
                self.image_processor.image_std = [1.0, 1.0, 1.0]
            print(
                "[DINOv3 checkpoint] using scripted segmentation normalization: mean=0.5 std=1.0",
                flush=True,
            )

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
