import math
from copy import copy
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _split_csv(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _split_optional_ints(value) -> List[Optional[int]]:
    items = _split_csv(value)
    result = []
    for item in items:
        if item.lower() in {"none", "null", "-"}:
            result.append(None)
        else:
            result.append(int(item))
    return result


def _canonical_fusion_type(value: str) -> str:
    value = str(value or "softmax_router").lower()
    if value in {"softmax_router", "router", "token_router", "moe", "gating", "gated_moe"}:
        return "softmax_router"
    if value in {"concat_projector", "concat_mlp", "static_concat", "prismatic_concat", "concat"}:
        return "concat_projector"
    raise ValueError(f"Unsupported multi_vision_fusion: {value}")


class MultiVisionMoEVisionTower(nn.Module):
    """Generic multi-vision tower fusion.

    Each expert is an ordinary project vision tower. Features are aligned to a
    shared square token grid, fused, and returned through the same interface as a
    normal vision tower, so the existing mm_projector and LLM path stay unchanged.
    """

    mm_vision_tower_type = "multi_moe"

    def __init__(self, args, single_tower_builder: Callable, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.single_tower_builder = single_tower_builder
        requested_type = str(getattr(args, "mm_vision_tower_type", "multi_moe") or "multi_moe").lower()
        self.tune_vision_tower = getattr(args, "unfreeze_mm_vision_tower", False)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        self.input_image_size = getattr(args, "input_image_size", None)
        self.primary_index = int(getattr(args, "multi_vision_primary_index", 0) or 0)
        self.target_grid = getattr(args, "multi_vision_target_grid", None)
        self.requested_hidden_size = getattr(args, "multi_vision_hidden_size", None)
        self.router_temperature = float(getattr(args, "multi_vision_router_temperature", 1.0) or 1.0)
        self.router_use_diff = bool(getattr(args, "multi_vision_router_use_diff", True))
        self.router_hidden_ratio = float(getattr(args, "multi_vision_router_hidden_ratio", 0.25) or 0.25)
        self.dropout_p = float(getattr(args, "multi_vision_dropout", 0.0) or 0.0)
        fusion_type = getattr(args, "multi_vision_fusion", "softmax_router") or "softmax_router"
        if requested_type in {
            "multi_concat",
            "multi_vision_concat",
            "dual_vision_concat",
            "prismatic_concat",
            "dino_siglip_concat",
            "dinov2_siglip_concat",
            "dinov3_siglip_concat",
        } and str(fusion_type).lower() == "softmax_router":
            fusion_type = "concat_projector"
        self.fusion_type = _canonical_fusion_type(fusion_type)
        self.mm_vision_tower_type = "multi_concat" if self.fusion_type == "concat_projector" else "multi_moe"
        self.deepstack_visual_indexes = None
        self.deepstack_mergers = None
        self.last_router_weights = None

        towers = _split_csv(getattr(args, "multi_vision_towers", None))
        if not towers:
            towers = _split_csv(getattr(args, "mm_vision_tower", getattr(args, "vision_tower", None)))
        if len(towers) < 2:
            raise ValueError(
                "MultiVisionMoE needs at least two vision towers. "
                "Pass --mm_vision_tower_type multi_moe and either "
                "--vision_tower path1,path2 or --multi_vision_towers path1,path2."
            )

        tower_types = _split_csv(getattr(args, "multi_vision_tower_types", None))
        input_sizes = _split_optional_ints(getattr(args, "multi_vision_input_image_sizes", None))
        if tower_types and len(tower_types) != len(towers):
            raise ValueError("multi_vision_tower_types must have the same length as multi_vision_towers.")
        if input_sizes and len(input_sizes) != len(towers):
            raise ValueError("multi_vision_input_image_sizes must have the same length as multi_vision_towers.")
        if not 0 <= self.primary_index < len(towers):
            raise ValueError(f"multi_vision_primary_index={self.primary_index} is out of range for {len(towers)} towers.")

        self.vision_tower_names = towers
        self.vision_tower_name = ",".join(towers)
        self.vision_tower_types = tower_types
        self.multi_vision_towers = self.vision_tower_name
        self.multi_vision_tower_types = ",".join(tower_types) if tower_types else None
        self.multi_vision_input_image_sizes = ",".join(str(item) if item is not None else "none" for item in input_sizes) if input_sizes else None
        self.multi_vision_primary_index = self.primary_index
        self.multi_vision_hidden_size = self.requested_hidden_size
        self.multi_vision_target_grid = self.target_grid
        self.multi_vision_fusion = self.fusion_type
        self.multi_vision_router_temperature = self.router_temperature
        self.multi_vision_router_hidden_ratio = self.router_hidden_ratio
        self.multi_vision_router_use_diff = self.router_use_diff
        self.multi_vision_dropout = self.dropout_p

        expert_modules = []
        for idx, tower_path in enumerate(towers):
            expert_args = copy(args)
            expert_args.vision_tower = tower_path
            expert_args.mm_vision_tower = tower_path
            if tower_types:
                expert_args.mm_vision_tower_type = tower_types[idx]
            else:
                expert_args.mm_vision_tower_type = None
            if input_sizes:
                expert_args.input_image_size = input_sizes[idx]
            else:
                expert_args.input_image_size = self.input_image_size
            expert_args.deepstack_visual_indexes = None
            expert_args.disable_deepstack = True
            expert_modules.append(single_tower_builder(expert_args, delay_load=delay_load))

        self.vision_towers = nn.ModuleList(expert_modules)
        self.num_experts = len(self.vision_towers)
        self.image_processor = getattr(self.vision_towers[self.primary_index], "image_processor", None)
        self._target_size = (
            getattr(self.vision_towers[self.primary_index], "_target_size", None)
            or getattr(self.vision_towers[self.primary_index], "input_image_size", None)
            or self.input_image_size
        )

        self._build_fusion_layers()
        if not delay_load:
            self.is_loaded = all(getattr(tower, "is_loaded", False) for tower in self.vision_towers)

    def _expert_hidden_sizes(self) -> List[int]:
        return [int(tower.hidden_size) for tower in self.vision_towers]

    def _build_fusion_layers(self):
        expert_hidden_sizes = self._expert_hidden_sizes()
        hidden_size = int(self.requested_hidden_size or max(expert_hidden_sizes))
        self._hidden_size = hidden_size

        self.expert_adapters = nn.ModuleList()
        self.router = nn.Identity()
        self.post_fusion = nn.Identity()
        self.concat_projector = nn.Identity()
        self.out_norm = nn.LayerNorm(hidden_size)

        if self.fusion_type == "softmax_router":
            self.expert_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, hidden_size),
                )
                for in_dim in expert_hidden_sizes
            ])
            router_in_dim = hidden_size * self.num_experts
            if self.router_use_diff:
                router_in_dim *= 2
            router_hidden = max(64, int(hidden_size * self.router_hidden_ratio))
            self.router = nn.Sequential(
                nn.LayerNorm(router_in_dim),
                nn.Linear(router_in_dim, router_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(router_hidden, self.num_experts),
            )
            self.post_fusion = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(hidden_size * 4, hidden_size),
            )
            return

        if self.fusion_type == "concat_projector":
            concat_in_dim = sum(expert_hidden_sizes)
            concat_hidden = max(hidden_size * 4, concat_in_dim)
            self.concat_projector = nn.Sequential(
                nn.LayerNorm(concat_in_dim),
                nn.Linear(concat_in_dim, concat_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(concat_hidden, hidden_size),
            )
            return

        raise ValueError(f"Unsupported multi_vision_fusion: {self.fusion_type}")

    def load_model(self, device_map=None):
        for tower in self.vision_towers:
            if not getattr(tower, "is_loaded", False):
                tower.load_model(device_map=device_map)
        self.image_processor = self.vision_towers[self.primary_index].image_processor
        self._target_size = (
            getattr(self.vision_towers[self.primary_index], "_target_size", None)
            or getattr(self.vision_towers[self.primary_index], "input_image_size", None)
            or self.input_image_size
        )
        if self.target_grid is None:
            grids = [getattr(tower, "num_patches_per_side", None) for tower in self.vision_towers]
            grids = [int(grid) for grid in grids if grid is not None]
            self.target_grid = min(grids) if grids else None
        self.multi_vision_target_grid = self.target_grid
        self.multi_vision_hidden_size = self._hidden_size
        self.is_loaded = True

    def set_vision_tower_trainable(self, trainable: bool):
        self.tune_vision_tower = bool(trainable)
        for tower in self.vision_towers:
            tower.tune_vision_tower = bool(trainable)
            if hasattr(tower, "vision_tower"):
                tower.vision_tower.requires_grad_(bool(trainable))
            else:
                tower.requires_grad_(bool(trainable))
        for module in (self.expert_adapters, self.router, self.post_fusion, self.concat_projector, self.out_norm):
            module.requires_grad_(True)

    def set_llm_hidden_size(self, llm_hidden_size):
        # Main-path fusion outputs vision hidden-size tokens and still uses the
        # existing mm_projector. DeepStack fusion can be added here later without
        # changing the public interface.
        return

    @staticmethod
    def _normalize_output(output):
        if isinstance(output, (tuple, list)) and len(output) == 2:
            return output[0], output[1]
        return output, None

    @staticmethod
    def _as_batch_tensor(features):
        if isinstance(features, list):
            return torch.cat(features, dim=0)
        return features

    @staticmethod
    def _infer_square_grid(num_tokens: int) -> int:
        grid = int(math.sqrt(num_tokens))
        if grid * grid != num_tokens:
            raise ValueError(
                f"MultiVisionMoE expects square patch-token grids, got {num_tokens} tokens. "
                "Use mm_vision_select_feature=patch and square image preprocessing."
            )
        return grid

    def _resize_tokens(self, tokens, target_grid: int):
        current_grid = self._infer_square_grid(tokens.shape[1])
        if current_grid == target_grid:
            return tokens
        bsz, _, channels = tokens.shape
        tokens_2d = tokens.transpose(1, 2).reshape(bsz, channels, current_grid, current_grid)
        resized = F.interpolate(tokens_2d, size=(target_grid, target_grid), mode="bicubic", align_corners=False)
        return resized.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _processor_stats(processor, device, dtype):
        if processor is None:
            return None
        mean = getattr(processor, "image_mean", None)
        std = getattr(processor, "image_std", None)
        if mean is None or std is None:
            return None
        if not isinstance(mean, (list, tuple)):
            mean = [float(mean)]
        if not isinstance(std, (list, tuple)):
            std = [float(std)]
        if len(mean) == 1:
            mean = list(mean) * 3
        if len(std) == 1:
            std = list(std) * 3
        mean_t = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
        std_t = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)
        return mean_t, std_t

    @staticmethod
    def _tower_target_size(tower):
        target_size = (
            getattr(tower, "_target_size", None)
            or getattr(tower, "input_image_size", None)
            or getattr(getattr(tower, "config", None), "image_size", None)
        )
        return int(target_size) if target_size is not None else None

    def _prepare_tensor_for_tower(self, images: torch.Tensor, tower_idx: int):
        if images.dim() not in (3, 4):
            return images

        tower = self.vision_towers[tower_idx]
        device = images.device
        dtype = images.dtype
        primary_stats = self._processor_stats(self.image_processor, device=device, dtype=dtype)
        expert_stats = self._processor_stats(getattr(tower, "image_processor", None), device=device, dtype=dtype)

        batched = images.dim() == 4
        work = images if batched else images.unsqueeze(0)
        if primary_stats is not None:
            primary_mean, primary_std = primary_stats
            work = work * primary_std + primary_mean

        target_size = self._tower_target_size(tower)
        if target_size is not None and (work.shape[-2] != target_size or work.shape[-1] != target_size):
            work = F.interpolate(work, size=(target_size, target_size), mode="bilinear", align_corners=False)

        if expert_stats is not None:
            expert_mean, expert_std = expert_stats
            work = (work - expert_mean) / expert_std
        return work if batched else work.squeeze(0)

    def _prepare_images_for_tower(self, images, tower_idx: int):
        if isinstance(images, list):
            return [self._prepare_tensor_for_tower(image, tower_idx) for image in images]
        if torch.is_tensor(images):
            return self._prepare_tensor_for_tower(images, tower_idx)
        return images

    def _align_raw_features(self, expert_features):
        aligned = []
        grids = []
        target_param = next(self.out_norm.parameters())
        for features in expert_features:
            features = self._as_batch_tensor(features)
            features = features.to(device=target_param.device, dtype=target_param.dtype)
            aligned.append(features)
            grids.append(self._infer_square_grid(features.shape[1]))

        target_grid = int(self.target_grid or min(grids))
        return [self._resize_tokens(features, target_grid) for features in aligned]

    def _fuse_softmax_router(self, expert_features):
        adapted = []
        grids = []
        for idx, features in enumerate(expert_features):
            features = self._as_batch_tensor(features)
            adapter_param = next(self.expert_adapters[idx].parameters())
            features = features.to(device=adapter_param.device, dtype=adapter_param.dtype)
            features = self.expert_adapters[idx](features)
            adapted.append(features)
            grids.append(self._infer_square_grid(features.shape[1]))

        target_grid = int(self.target_grid or min(grids))
        aligned = [self._resize_tokens(features, target_grid) for features in adapted]
        stacked = torch.stack(aligned, dim=2)  # B x T x E x D
        router_parts = list(aligned)
        if self.router_use_diff:
            mean_feature = stacked.mean(dim=2)
            router_parts.extend([torch.abs(features - mean_feature) for features in aligned])
        router_input = torch.cat(router_parts, dim=-1)
        logits = self.router(router_input)
        weights = torch.softmax(logits / max(self.router_temperature, 1e-6), dim=-1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=2)
        fused = fused + self.post_fusion(fused)
        return self.out_norm(fused), weights

    def _fuse_concat_projector(self, expert_features):
        aligned = self._align_raw_features(expert_features)
        concat_features = torch.cat(aligned, dim=-1)
        fused = self.concat_projector(concat_features)
        return self.out_norm(fused), None

    def forward(self, images):
        expert_features = []
        for idx, tower in enumerate(self.vision_towers):
            tower_images = self._prepare_images_for_tower(images, idx)
            main_features, _ = self._normalize_output(tower(tower_images))
            expert_features.append(main_features)

        if self.fusion_type == "softmax_router":
            fused, router_weights = self._fuse_softmax_router(expert_features)
        elif self.fusion_type == "concat_projector":
            fused, router_weights = self._fuse_concat_projector(expert_features)
        else:
            raise ValueError(f"Unsupported multi_vision_fusion: {self.fusion_type}")
        self.last_router_weights = router_weights.detach() if router_weights is not None else None
        return fused, None

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def config(self):
        patch_size = None
        if self._target_size:
            patch_size = self._target_size // self.num_patches_per_side
        return {
            "image_size": self._target_size,
            "hidden_size": self.hidden_size,
            "patch_size": patch_size,
            "vision_tower_type": self.mm_vision_tower_type,
            "vision_towers": self.vision_tower_names,
            "multi_vision_fusion": self.fusion_type,
        }

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_patches_per_side(self):
        if self.target_grid is not None:
            return int(self.target_grid)
        grids = [getattr(tower, "num_patches_per_side", None) for tower in self.vision_towers]
        grids = [int(grid) for grid in grids if grid is not None]
        if grids:
            return min(grids)
        raise AttributeError("target grid is unknown before expert configs are loaded.")

    @property
    def num_patches(self):
        return self.num_patches_per_side ** 2
