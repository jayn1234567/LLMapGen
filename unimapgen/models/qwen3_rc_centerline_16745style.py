from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from unimapgen.models.encoders.satellite_encoder import SatelliteEncoder
from unimapgen.models.hf_utils import resolve_hf_snapshot_path

try:
    from peft import LoraConfig, PeftModel, get_peft_model

    _PEFT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    LoraConfig = None
    PeftModel = None
    get_peft_model = None
    _PEFT_IMPORT_ERROR = exc

try:
    from transformers import AutoModelForCausalLM

    _TRANSFORMERS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    AutoModelForCausalLM = None
    _TRANSFORMERS_IMPORT_ERROR = exc

try:
    from safetensors.torch import load_file as load_safetensors_file

    _SAFETENSORS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    load_safetensors_file = None
    _SAFETENSORS_IMPORT_ERROR = exc


def build_1d_sincos_embedding(positions: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    freq_idx = torch.arange(half_dim, dtype=torch.float32, device=positions.device)
    denom = torch.pow(10000.0, freq_idx / max(half_dim, 1))
    phase = positions.unsqueeze(-1) / denom.unsqueeze(0)
    emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.cat([emb, emb.new_zeros((*emb.shape[:-1], dim - emb.shape[-1]))], dim=-1)
    return emb[..., :dim]


def build_2d_sincos_embedding(height: int, width: int, dim: int) -> torch.Tensor:
    if int(dim) % 2 != 0:
        raise ValueError(f"2D sin-cos embedding requires even dim, got {dim}")
    y = torch.arange(int(height), dtype=torch.float32)
    x = torch.arange(int(width), dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    half = dim // 2
    emb_y = build_1d_sincos_embedding(yy.reshape(-1), half)
    emb_x = build_1d_sincos_embedding(xx.reshape(-1), dim - half)
    return torch.cat([emb_y, emb_x], dim=-1)


def resolve_torch_dtype(dtype_name: str) -> torch.dtype | str | None:
    name = str(dtype_name).strip().lower()
    if not name or name == "none":
        return None
    if name == "auto":
        return "auto"
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    resolved = mapping.get(name)
    if resolved is None:
        raise ValueError(f"Unsupported torch dtype spec: {dtype_name}")
    return resolved


def initialize_new_token_embeddings(model: nn.Module, old_vocab_size: int) -> None:
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        return
    new_vocab_size = int(input_embeddings.weight.shape[0])
    if new_vocab_size <= int(old_vocab_size):
        return
    with torch.no_grad():
        avg_input = input_embeddings.weight[: int(old_vocab_size)].mean(dim=0, keepdim=True)
        input_embeddings.weight[int(old_vocab_size) : new_vocab_size] = avg_input
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and int(output_embeddings.weight.shape[0]) == new_vocab_size:
            avg_output = output_embeddings.weight[: int(old_vocab_size)].mean(dim=0, keepdim=True)
            output_embeddings.weight[int(old_vocab_size) : new_vocab_size] = avg_output


class SelectiveTrainableTokenAdapter(nn.Module):
    def __init__(
        self,
        input_weight: torch.Tensor,
        token_ids: Sequence[int],
        output_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        ids = [int(token_id) for token_id in token_ids]
        hidden_size = int(input_weight.shape[1])
        if ids:
            input_init = input_weight[ids].detach().clone()
        else:
            input_init = input_weight.new_zeros((0, hidden_size))
        self.register_buffer("token_ids", torch.tensor(ids, dtype=torch.long), persistent=True)
        self.register_buffer("base_input_weight", input_weight.detach().clone(), persistent=False)
        self.trainable_input_rows = nn.Parameter(input_init)

        tied_output = bool(output_weight is None or output_weight.data_ptr() == input_weight.data_ptr())
        self.output_tied = tied_output
        if tied_output:
            self.register_buffer("base_output_weight", torch.empty(0, dtype=input_weight.dtype), persistent=False)
            self.trainable_output_rows = None
        else:
            output_init = output_weight[ids].detach().clone() if ids else output_weight.new_zeros((0, hidden_size))
            self.register_buffer("base_output_weight", output_weight.detach().clone(), persistent=False)
            self.trainable_output_rows = nn.Parameter(output_init)

    @property
    def has_trainable_tokens(self) -> bool:
        return int(self.token_ids.numel()) > 0

    def set_trainable(self, enabled: bool) -> None:
        self.trainable_input_rows.requires_grad_(bool(enabled))
        if self.trainable_output_rows is not None:
            self.trainable_output_rows.requires_grad_(bool(enabled))

    def _scatter_rows(self, base_weight: torch.Tensor, rows: torch.Tensor | None) -> torch.Tensor:
        if rows is None or not self.has_trainable_tokens:
            return base_weight
        token_ids = self.token_ids.to(device=base_weight.device)
        scatter_index = token_ids.view(-1, 1).expand(-1, int(base_weight.shape[1]))
        return base_weight.scatter(0, scatter_index, rows.to(device=base_weight.device, dtype=base_weight.dtype))

    def composed_input_weight(self) -> torch.Tensor:
        return self._scatter_rows(self.base_input_weight, self.trainable_input_rows)

    def composed_output_weight(self) -> torch.Tensor:
        if self.output_tied:
            return self.composed_input_weight()
        return self._scatter_rows(self.base_output_weight, self.trainable_output_rows)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = F.embedding(input_ids, self.base_input_weight)
        if not self.has_trainable_tokens:
            return embeddings
        for row_idx, token_id in enumerate(self.token_ids.tolist()):
            mask = input_ids.eq(int(token_id))
            if not bool(mask.any()):
                continue
            replacement = self.trainable_input_rows[row_idx].to(device=embeddings.device, dtype=embeddings.dtype)
            view_shape = [1] * embeddings.ndim
            view_shape[-1] = int(replacement.shape[0])
            embeddings = torch.where(mask.unsqueeze(-1), replacement.view(*view_shape), embeddings)
        return embeddings


class SelectiveTokenEmbedding(nn.Module):
    def __init__(self, adapter: SelectiveTrainableTokenAdapter) -> None:
        super().__init__()
        self.adapter = adapter
        self.num_embeddings = int(adapter.base_input_weight.shape[0])
        self.embedding_dim = int(adapter.base_input_weight.shape[1])

    @property
    def weight(self) -> torch.Tensor:
        return self.adapter.composed_input_weight()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.adapter.embed(input_ids)


class SelectiveTokenLMHead(nn.Module):
    def __init__(self, adapter: SelectiveTrainableTokenAdapter, bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self.adapter = adapter
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", nn.Parameter(bias.detach().clone(), requires_grad=False))

    @property
    def weight(self) -> torch.Tensor:
        return self.adapter.composed_output_weight()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.adapter.composed_output_weight().to(device=hidden_states.device, dtype=hidden_states.dtype)
        bias = None if self.bias is None else self.bias.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return F.linear(hidden_states, weight, bias)


def unwrap_model(model: nn.Module) -> nn.Module:
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return unwrapped


def load_optional_state_dict(module: nn.Module, state_dict: Dict[str, torch.Tensor] | None, name: str) -> None:
    if not state_dict:
        return
    target_state = module.state_dict()
    filtered_state, skipped_missing, skipped_mismatch = filter_state_dict_by_shape(target_state, state_dict)
    missing, unexpected = module.load_state_dict(filtered_state, strict=False)
    missing = list(missing)
    unexpected = list(unexpected)
    print(
        (
            f"[qwen3-rc] loaded {name}: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"skipped_missing={len(skipped_missing)} skipped_mismatch={len(skipped_mismatch)}"
        ),
        flush=True,
    )


def load_adapter_state_dict(adapter_dir: Path) -> Dict[str, torch.Tensor]:
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.is_file():
        if load_safetensors_file is None:
            raise RuntimeError(
                "Adapter checkpoint requires safetensors, but import failed: "
                f"{_SAFETENSORS_IMPORT_ERROR!r}"
            )
        return load_safetensors_file(str(safetensors_path))
    bin_path = adapter_dir / "adapter_model.bin"
    if bin_path.is_file():
        state = torch.load(str(bin_path), map_location="cpu")
        if not isinstance(state, dict):
            raise TypeError(f"Unexpected adapter checkpoint type: {type(state)!r}")
        return state
    raise FileNotFoundError(f"No adapter model file found under: {adapter_dir}")


def filter_state_dict_by_shape(
    target_state: Dict[str, torch.Tensor],
    source_state: Dict[str, torch.Tensor],
) -> tuple[Dict[str, torch.Tensor], list[str], list[str]]:
    kept: Dict[str, torch.Tensor] = {}
    skipped_missing: list[str] = []
    skipped_mismatch: list[str] = []
    for key, value in source_state.items():
        target_value = target_state.get(key)
        if target_value is None:
            skipped_missing.append(str(key))
            continue
        if tuple(target_value.shape) != tuple(value.shape):
            skipped_mismatch.append(str(key))
            continue
        kept[str(key)] = value
    return kept, skipped_missing, skipped_mismatch


def choose_group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        groups = choose_group_count(int(out_channels))
        self.block = nn.Sequential(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(groups, int(out_channels)),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RoadFeatureResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = choose_group_count(int(channels))
        self.conv1 = nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, int(channels))
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, int(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.act(x + residual)


class RoadFeatureNeck(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_blocks: int) -> None:
        super().__init__()
        self.in_proj = (
            ConvNormAct(int(in_channels), int(hidden_channels), kernel_size=1, padding=0)
            if int(in_channels) != int(hidden_channels)
            else ConvNormAct(int(in_channels), int(hidden_channels))
        )
        self.blocks = nn.Sequential(
            *[RoadFeatureResidualBlock(int(hidden_channels)) for _ in range(max(1, int(num_blocks)))]
        )
        self.out_proj = (
            ConvNormAct(int(hidden_channels), int(in_channels), kernel_size=1, padding=0)
            if int(hidden_channels) != int(in_channels)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.out_proj(x)


class HRMapNetStyleSegHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample_scale: int = 1) -> None:
        super().__init__()
        self.upsample_scale = max(1, int(upsample_scale))
        self.pre = nn.Sequential(
            nn.Conv2d(int(in_channels), int(in_channels), kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.post = nn.Sequential(
            nn.Conv2d(int(in_channels), int(in_channels), kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        if int(self.upsample_scale) > 1:
            x = F.interpolate(
                x,
                scale_factor=float(self.upsample_scale),
                mode="bilinear",
                align_corners=False,
            )
        return self.post(x)


class QueryResamplerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(int(hidden_dim))
        self.self_attn = nn.MultiheadAttention(
            embed_dim=int(hidden_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(int(hidden_dim))
        self.memory_norm = nn.LayerNorm(int(hidden_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=int(hidden_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(int(hidden_dim))
        mlp_hidden_dim = max(int(hidden_dim), int(round(float(hidden_dim) * float(mlp_ratio))))
        self.ffn = nn.Sequential(
            nn.Linear(int(hidden_dim), int(mlp_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(mlp_hidden_dim), int(hidden_dim)),
        )

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
    ) -> torch.Tensor:
        norm_query = self.self_attn_norm(query)
        self_attn_out, _ = self.self_attn(
            norm_query + query_pos,
            norm_query + query_pos,
            norm_query,
            need_weights=False,
        )
        query = query + self_attn_out

        norm_query = self.cross_attn_norm(query)
        norm_memory = self.memory_norm(memory)
        cross_attn_out, _ = self.cross_attn(
            norm_query + query_pos,
            norm_memory + memory_pos,
            norm_memory,
            need_weights=False,
        )
        query = query + cross_attn_out
        query = query + self.ffn(self.ffn_norm(query))
        return query


class LearnedQueryResampler(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
        query_grid_size: int,
        num_layers: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.query_grid_size = int(query_grid_size)
        if self.query_grid_size * self.query_grid_size != self.num_queries:
            raise ValueError(
                f"Query resampler expects square query grid, got num_queries={num_queries} "
                f"query_grid_size={query_grid_size}"
            )
        self.query_content = nn.Parameter(torch.randn(1, self.num_queries, self.hidden_dim) * 0.02)
        query_pos = build_2d_sincos_embedding(self.query_grid_size, self.query_grid_size, self.hidden_dim)
        self.register_buffer("query_position_embeddings", query_pos.unsqueeze(0), persistent=False)
        self.layers = nn.ModuleList(
            [
                QueryResamplerBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self._dense_pos_cache: dict[tuple[int, int, str, str], torch.Tensor] = {}

    def _dense_position_embeddings(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (int(height), int(width), str(device), str(dtype))
        cached = self._dense_pos_cache.get(key)
        if cached is None or cached.device != device or cached.dtype != dtype:
            pos = build_2d_sincos_embedding(int(height), int(width), self.hidden_dim)
            cached = pos.unsqueeze(0).to(device=device, dtype=dtype)
            self._dense_pos_cache = {key: cached}
        return cached

    def forward(self, dense_features: torch.Tensor) -> torch.Tensor:
        if dense_features.ndim != 4:
            raise ValueError(
                f"LearnedQueryResampler expects dense features of shape [B, C, H, W], got {tuple(dense_features.shape)}"
            )
        batch_size, channels, height, width = dense_features.shape
        if int(channels) != self.hidden_dim:
            raise ValueError(
                f"LearnedQueryResampler hidden dim mismatch: features={channels} expected={self.hidden_dim}"
            )
        memory = dense_features.flatten(2).transpose(1, 2).contiguous()
        memory_pos = self._dense_position_embeddings(
            int(height),
            int(width),
            device=memory.device,
            dtype=memory.dtype,
        )
        query = self.query_content.expand(int(batch_size), -1, -1).to(device=memory.device, dtype=memory.dtype)
        query_pos = self.query_position_embeddings.to(device=memory.device, dtype=memory.dtype)
        for layer in self.layers:
            query = layer(query=query, query_pos=query_pos, memory=memory, memory_pos=memory_pos)
        return self.output_norm(query)


class Qwen3RCCenterlineModel(nn.Module):
    supports_gradient_checkpointing = True

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Any,
        num_visual_tokens: int = 64,
        visual_grid_size: int = 8,
        visual_hidden_dim: int = 256,
        visual_projector_hidden_dim: int = 1024,
        coord_head_hidden_dim: int = 1024,
        language_model_dtype: str = "auto",
        visual_backbone: str = "resnet50_fpn",
        visual_backbone_pretrained: bool = False,
        visual_backbone_weights_path: str = "",
        road_neck_hidden_dim: int = 0,
        road_neck_num_blocks: int = 2,
        query_resampler_depth: int = 3,
        query_resampler_heads: int = 8,
        query_resampler_mlp_ratio: float = 4.0,
        seg_num_classes: int = 1,
        local_files_only: bool = False,
        freeze_language_model: bool = False,
        freeze_vision_encoder: bool = False,
        freeze_vision_backbone_only: bool = False,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = False,
        coord_use_sigmoid: bool = True,
        train_new_token_embeddings_only: bool = True,
    ) -> None:
        super().__init__()
        if AutoModelForCausalLM is None:
            raise RuntimeError(
                "Qwen3RCCenterlineModel failed to import transformers.AutoModelForCausalLM. "
                f"Original import error: {_TRANSFORMERS_IMPORT_ERROR!r}"
            )

        resolved_model_path = resolve_hf_snapshot_path(str(model_name_or_path))
        modules_state_path = Path(resolved_model_path) / "rc_cnn_prefix_modules.pt"
        adapter_config_path = Path(resolved_model_path) / "adapter_config.json"
        language_model_source = resolved_model_path
        adapter_checkpoint_dir: Path | None = None
        if adapter_config_path.is_file():
            adapter_checkpoint_dir = Path(resolved_model_path)
            adapter_cfg = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            base_model_name_or_path = str(adapter_cfg.get("base_model_name_or_path", "")).strip()
            if not base_model_name_or_path:
                raise ValueError(f"adapter_config.json missing base_model_name_or_path: {adapter_config_path}")
            language_model_source = resolve_hf_snapshot_path(base_model_name_or_path)
            print(
                f"[qwen3-rc] adapter checkpoint detected: base_model={language_model_source} adapter={adapter_checkpoint_dir}",
                flush=True,
            )
        self.num_visual_tokens = int(num_visual_tokens)
        self.visual_grid_size = int(visual_grid_size)
        self.coord_use_sigmoid = bool(coord_use_sigmoid)
        if int(self.visual_grid_size) * int(self.visual_grid_size) != int(self.num_visual_tokens):
            raise ValueError(
                f"visual_grid_size={self.visual_grid_size} does not match num_visual_tokens={self.num_visual_tokens}"
            )

        lm_torch_dtype = resolve_torch_dtype(str(language_model_dtype))
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_source,
            local_files_only=bool(local_files_only),
            trust_remote_code=True,
            torch_dtype=lm_torch_dtype,
            low_cpu_mem_usage=True,
        )
        old_vocab_size = int(self.language_model.get_input_embeddings().weight.shape[0])
        new_vocab_size = int(len(tokenizer))
        if new_vocab_size != old_vocab_size:
            self.language_model.resize_token_embeddings(new_vocab_size)
            initialize_new_token_embeddings(self.language_model, old_vocab_size=old_vocab_size)
        self.new_token_start = int(old_vocab_size)
        self.new_token_end = int(new_vocab_size)
        self.train_new_token_embeddings_only = bool(train_new_token_embeddings_only)
        self.train_full_token_embeddings = not bool(self.train_new_token_embeddings_only)
        self.special_token_adapter: SelectiveTrainableTokenAdapter | None = None
        if self.train_new_token_embeddings_only and self.new_token_end > self.new_token_start:
            new_token_ids = list(range(self.new_token_start, self.new_token_end))
            self._enable_selective_trainable_tokens(new_token_ids)
            print(
                (
                    "[qwen3-rc] selective token tuning enabled for newly added tokens: "
                    f"count={len(new_token_ids)} ids={new_token_ids[0]}..{new_token_ids[-1]}"
                ),
                flush=True,
            )

        self.hidden_size = int(self.language_model.get_input_embeddings().weight.shape[1])
        self.seg_num_classes = max(1, int(seg_num_classes))
        self.vis_patch_token_id = int(tokenizer.convert_tokens_to_ids("<vis_patch>"))
        self.coord_token_id = int(tokenizer.convert_tokens_to_ids("<coord_pt>"))
        if self.vis_patch_token_id < 0 or self.coord_token_id < 0:
            raise ValueError("Tokenizer is missing required special tokens.")

        self.vision_encoder = SatelliteEncoder(
            use_fallback=True,
            fallback_backbone=str(visual_backbone),
            fallback_pretrained=bool(visual_backbone_pretrained),
            fallback_weights_path=str(visual_backbone_weights_path),
            fallback_channels=(64, 128, 256),
            fallback_hw=(self.visual_grid_size, self.visual_grid_size),
            fallback_dim=int(visual_hidden_dim),
            out_hw=(self.visual_grid_size, self.visual_grid_size),
        )
        self.visual_projector = nn.Sequential(
            nn.Linear(int(self.vision_encoder.hidden_size), int(visual_projector_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(visual_projector_hidden_dim), self.hidden_size),
        )
        neck_hidden = int(road_neck_hidden_dim) if int(road_neck_hidden_dim) > 0 else int(self.vision_encoder.hidden_size)
        self.road_neck = RoadFeatureNeck(
            in_channels=int(self.vision_encoder.hidden_size),
            hidden_channels=neck_hidden,
            num_blocks=int(road_neck_num_blocks),
        )
        self.query_resampler = LearnedQueryResampler(
            hidden_dim=int(self.vision_encoder.hidden_size),
            num_queries=self.num_visual_tokens,
            query_grid_size=self.visual_grid_size,
            num_layers=int(query_resampler_depth),
            num_heads=int(query_resampler_heads),
            mlp_ratio=float(query_resampler_mlp_ratio),
        )
        self.seg_head = HRMapNetStyleSegHead(
            in_channels=int(self.vision_encoder.hidden_size),
            out_channels=self.seg_num_classes,
            upsample_scale=2,
        )
        self.centerline_heatmap_head = HRMapNetStyleSegHead(
            in_channels=int(self.vision_encoder.hidden_size),
            out_channels=1,
            upsample_scale=2,
        )
        self.coord_head = nn.Sequential(
            nn.Linear(self.hidden_size, int(coord_head_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(coord_head_hidden_dim), 2),
        )
        self.visual_type_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        pos = build_2d_sincos_embedding(self.visual_grid_size, self.visual_grid_size, self.hidden_size)
        self.register_buffer("visual_position_embeddings", pos.unsqueeze(0), persistent=False)

        modules_state: Dict[str, Any] | None = None
        if modules_state_path.is_file():
            modules_state = torch.load(str(modules_state_path), map_location="cpu")
            print(f"[qwen3-rc] found custom module weights at {modules_state_path}", flush=True)

        lm_has_peft = bool(getattr(self.language_model, "peft_config", None))
        if bool(use_lora) and adapter_checkpoint_dir is not None:
            if PeftModel is None:
                raise RuntimeError(
                    "Qwen3RCCenterlineModel needs peft.PeftModel to restore adapter checkpoints, "
                    f"but peft import failed: {_PEFT_IMPORT_ERROR!r}"
                )
            self.language_model = PeftModel.from_pretrained(
                self.language_model,
                str(adapter_checkpoint_dir),
                is_trainable=True,
            )
            self._set_selective_token_trainable(bool(self.train_new_token_embeddings_only))
            print(
                f"[qwen3-rc] restored adapter checkpoint via PeftModel.from_pretrained: {adapter_checkpoint_dir}",
                flush=True,
            )
            try:
                self.language_model.print_trainable_parameters()
            except Exception:
                pass
        elif bool(use_lora) and not lm_has_peft:
            if LoraConfig is None or get_peft_model is None:
                raise RuntimeError(
                    "Qwen3RCCenterlineModel requested LoRA, but peft could not be imported. "
                    f"Original import error: {_PEFT_IMPORT_ERROR!r}"
                )
            modules_to_save = ["embed_tokens", "lm_head"] if self.train_full_token_embeddings else None
            peft_cfg = LoraConfig(
                r=int(lora_rank),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                modules_to_save=modules_to_save,
            )
            self.language_model = get_peft_model(self.language_model, peft_cfg)
            self._set_selective_token_trainable(bool(self.train_new_token_embeddings_only))
            if self.train_full_token_embeddings:
                print(
                    "[qwen3-rc] full embedding tuning enabled: train embed_tokens/lm_head together with LoRA",
                    flush=True,
                )
            try:
                self.language_model.print_trainable_parameters()
            except Exception:
                pass
        elif bool(freeze_language_model):
            for param in self.language_model.parameters():
                param.requires_grad = False
            self._set_selective_token_trainable(False)
            self._set_full_embedding_trainable(False)

        if isinstance(modules_state, dict):
            load_optional_state_dict(self.vision_encoder, modules_state.get("vision_encoder"), "vision_encoder")
            load_optional_state_dict(self.road_neck, modules_state.get("road_neck"), "road_neck")
            load_optional_state_dict(self.query_resampler, modules_state.get("query_resampler"), "query_resampler")
            load_optional_state_dict(self.visual_projector, modules_state.get("visual_projector"), "visual_projector")
            load_optional_state_dict(self.seg_head, modules_state.get("seg_head"), "seg_head")
            load_optional_state_dict(
                self.centerline_heatmap_head,
                modules_state.get("centerline_heatmap_head"),
                "centerline_heatmap_head",
            )
            load_optional_state_dict(
                self.coord_head,
                modules_state.get("coord_head", modules_state.get("coord_head_l4")),
                "coord_head",
            )
            load_optional_state_dict(
                self.special_token_adapter,
                modules_state.get("special_token_adapter"),
                "special_token_adapter",
            )
            visual_type_embedding = modules_state.get("visual_type_embedding")
            if isinstance(visual_type_embedding, torch.Tensor):
                with torch.no_grad():
                    self.visual_type_embedding.copy_(visual_type_embedding.to(dtype=self.visual_type_embedding.dtype))
                print("[qwen3-rc] loaded visual_type_embedding", flush=True)

        if bool(freeze_vision_encoder):
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        elif bool(freeze_vision_backbone_only):
            if hasattr(self.vision_encoder, "freeze_backbone_only"):
                self.vision_encoder.freeze_backbone_only()
            else:
                for param in self.vision_encoder.parameters():
                    param.requires_grad = False

        if bool(gradient_checkpointing) and hasattr(self.language_model, "gradient_checkpointing_enable"):
            try:
                self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.language_model.gradient_checkpointing_enable()
            if hasattr(self.language_model.config, "use_cache"):
                self.language_model.config.use_cache = False

    def _enable_selective_trainable_tokens(self, token_ids: Sequence[int]) -> None:
        input_embeddings = self.language_model.get_input_embeddings()
        output_embeddings = self.language_model.get_output_embeddings()
        if input_embeddings is None:
            raise ValueError("Language model does not expose input embeddings for selective token tuning.")
        input_weight = input_embeddings.weight.detach()
        output_weight = None if output_embeddings is None else output_embeddings.weight.detach()
        self.special_token_adapter = SelectiveTrainableTokenAdapter(
            input_weight=input_weight,
            token_ids=token_ids,
            output_weight=output_weight,
        )
        self.language_model.set_input_embeddings(SelectiveTokenEmbedding(self.special_token_adapter))
        if output_embeddings is not None:
            output_bias = getattr(output_embeddings, "bias", None)
            self.language_model.set_output_embeddings(SelectiveTokenLMHead(self.special_token_adapter, bias=output_bias))
        self._set_selective_token_trainable(True)

    def _set_selective_token_trainable(self, enabled: bool) -> None:
        if self.special_token_adapter is not None:
            self.special_token_adapter.set_trainable(bool(enabled))

    def _set_full_embedding_trainable(self, enabled: bool) -> None:
        if self.special_token_adapter is not None:
            return
        input_embeddings = self.language_model.get_input_embeddings()
        if input_embeddings is not None:
            for param in input_embeddings.parameters():
                param.requires_grad_(bool(enabled))
        output_embeddings = self.language_model.get_output_embeddings()
        if output_embeddings is not None and output_embeddings is not input_embeddings:
            for param in output_embeddings.parameters():
                param.requires_grad_(bool(enabled))

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
        kwargs = gradient_checkpointing_kwargs or {}
        if hasattr(self.language_model, "gradient_checkpointing_enable"):
            try:
                self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
            except TypeError:
                self.language_model.gradient_checkpointing_enable()
        if hasattr(self.language_model, "config") and hasattr(self.language_model.config, "use_cache"):
            self.language_model.config.use_cache = False

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.language_model, "gradient_checkpointing_disable"):
            self.language_model.gradient_checkpointing_disable()
        if hasattr(self.language_model, "config") and hasattr(self.language_model.config, "use_cache"):
            self.language_model.config.use_cache = True

    def build_visual_embeddings(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vision_outputs = self.vision_encoder.forward_features(pixel_values)
        dense_features = vision_outputs["dense_features"]
        if dense_features.ndim != 4:
            raise ValueError(f"Expected dense visual features to be 4D, got shape={tuple(dense_features.shape)}")
        adapted_dense = self.road_neck(dense_features)
        visual_tokens = self.query_resampler(adapted_dense)
        if int(visual_tokens.shape[1]) != self.num_visual_tokens:
            raise ValueError(
                f"Visual encoder produced {visual_tokens.shape[1]} tokens, expected {self.num_visual_tokens}"
            )
        projected = self.visual_projector(visual_tokens)
        pos = self.visual_position_embeddings.to(device=projected.device, dtype=projected.dtype)
        return (
            projected + pos + self.visual_type_embedding.to(device=projected.device, dtype=projected.dtype),
            adapted_dense,
            visual_tokens,
        )

    def query_tokens_to_map(self, query_tokens: torch.Tensor) -> torch.Tensor:
        if query_tokens.ndim != 3:
            raise ValueError(f"Expected query tokens to be [B, N, C], got shape={tuple(query_tokens.shape)}")
        batch_size, num_tokens, hidden_dim = query_tokens.shape
        if int(num_tokens) != int(self.num_visual_tokens):
            raise ValueError(f"Expected {self.num_visual_tokens} query tokens, got {num_tokens}")
        return query_tokens.transpose(1, 2).reshape(
            batch_size,
            hidden_dim,
            int(self.visual_grid_size),
            int(self.visual_grid_size),
        )

    def predict_coord_pyramid(self, hidden_states: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if len(hidden_states) < 1:
            raise ValueError(f"Need at least 1 hidden state tensor, got {len(hidden_states)}")
        coord_dtype = next(self.coord_head.parameters()).dtype
        hidden_last = hidden_states[-1].to(dtype=coord_dtype)
        coord_pred = self.coord_head(hidden_last)
        if bool(self.coord_use_sigmoid):
            coord_pred = torch.sigmoid(coord_pred)
        return {
            "coord_pred": coord_pred,
            "coord_pred_is_normalized": bool(self.coord_use_sigmoid),
        }

    def inject_visual_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        visual_embeddings: torch.Tensor,
        vis_patch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = vis_patch_mask if vis_patch_mask is not None else input_ids.eq(self.vis_patch_token_id)
        if mask.shape != input_ids.shape:
            raise ValueError("vis_patch_mask shape must match input_ids.")
        out = inputs_embeds.clone()
        for batch_idx in range(input_ids.shape[0]):
            positions = torch.nonzero(mask[batch_idx], as_tuple=False).flatten()
            if int(positions.numel()) != self.num_visual_tokens:
                raise ValueError(
                    f"Sample {batch_idx} has {int(positions.numel())} visual patch tokens, expected {self.num_visual_tokens}"
                )
            out[batch_idx, positions, :] = visual_embeddings[batch_idx]
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        vis_patch_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        del kwargs
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        visual_embeddings, adapted_dense, visual_tokens = self.build_visual_embeddings(pixel_values)
        inputs_embeds = self.inject_visual_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            visual_embeddings=visual_embeddings.to(dtype=inputs_embeds.dtype),
            vis_patch_mask=vis_patch_mask,
        )
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        coord_outputs = self.predict_coord_pyramid(outputs.hidden_states)
        seg_dtype = next(self.seg_head.parameters()).dtype
        heatmap_dtype = next(self.centerline_heatmap_head.parameters()).dtype
        seg_logits = self.seg_head(adapted_dense.to(dtype=seg_dtype))
        centerline_heatmap_logits = self.centerline_heatmap_head(adapted_dense.to(dtype=heatmap_dtype))
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            **coord_outputs,
            "seg_logits": seg_logits,
            "centerline_heatmap_logits": centerline_heatmap_logits,
        }


def save_qwen3_rc_centerline_modules(model: nn.Module, output_dir: str | Path, tokenizer: Any | None = None) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_for_save = unwrap_model(model)
    language_model = getattr(model_for_save, "language_model", None)
    if language_model is None:
        raise AttributeError("Wrapped model does not expose language_model.")
    language_model.save_pretrained(str(output_path))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_path))
    torch.save(
        {
            "vision_encoder": model_for_save.vision_encoder.state_dict(),
            "road_neck": model_for_save.road_neck.state_dict(),
            "visual_adapter": model_for_save.road_neck.state_dict(),
            "query_resampler": model_for_save.query_resampler.state_dict(),
            "visual_projector": model_for_save.visual_projector.state_dict(),
            "seg_head": model_for_save.seg_head.state_dict(),
            "centerline_heatmap_head": model_for_save.centerline_heatmap_head.state_dict(),
            "coord_head": model_for_save.coord_head.state_dict(),
            "special_token_adapter": (
                model_for_save.special_token_adapter.state_dict() if model_for_save.special_token_adapter is not None else None
            ),
            "visual_type_embedding": model_for_save.visual_type_embedding.detach().cpu(),
            "visual_grid_size": int(model_for_save.visual_grid_size),
            "num_visual_tokens": int(model_for_save.num_visual_tokens),
            "coord_use_sigmoid": bool(model_for_save.coord_use_sigmoid),
            "seg_num_classes": int(getattr(model_for_save, "seg_num_classes", 1)),
        },
        str(output_path / "rc_cnn_prefix_modules.pt"),
    )
