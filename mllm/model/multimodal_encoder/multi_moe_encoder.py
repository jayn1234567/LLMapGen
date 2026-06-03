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


class MultiVisionMoEVisionTower(nn.Module):
    """Generic token-level MoE router for multiple vision towers.

    Each expert is an ordinary project vision tower. Expert features are adapted
    to a shared hidden size, spatially resized to a shared square token grid, and
    fused by a per-token softmax router. The fused tokens keep the same interface
    as a normal vision tower, so the existing mm_projector and LLM injection path
    can stay unchanged.
    """

    mm_vision_tower_type = "multi_moe"

    def __init__(self, args, single_tower_builder: Callable, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.single_tower_builder = single_tower_builder
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
        self.fusion_type = str(getattr(args, "multi_vision_fusion", "softmax_router") or "softmax_router")
        self.deepstack_visual_indexes = None
        self.deepstack_mergers = None

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
        self.out_norm = nn.LayerNorm(hidden_size)

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
        self.is_loaded = True

    def set_vision_tower_trainable(self, trainable: bool):
        self.tune_vision_tower = bool(trainable)
        for tower in self.vision_towers:
            tower.tune_vision_tower = bool(trainable)
            if hasattr(tower, "vision_tower"):
                tower.vision_tower.requires_grad_(bool(trainable))
            else:
                tower.requires_grad_(bool(trainable))
        for module in (self.expert_adapters, self.router, self.post_fusion, self.out_norm):
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

    def _fuse(self, expert_features):
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

    def forward(self, images):
        if self.fusion_type != "softmax_router":
            raise ValueError(f"Unsupported multi_vision_fusion: {self.fusion_type}")
        expert_features = []
        for tower in self.vision_towers:
            main_features, _ = self._normalize_output(tower(images))
            expert_features.append(main_features)
        fused, router_weights = self._fuse(expert_features)
        self.last_router_weights = router_weights.detach()
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
