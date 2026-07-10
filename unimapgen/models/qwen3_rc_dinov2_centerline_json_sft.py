from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from unimapgen.models.encoders.satellite_encoder import SatelliteEncoder
from unimapgen.models.hf_utils import resolve_hf_snapshot_path
from unimapgen.models.qwen3_rc_centerline_16745style import (
    SelectiveTokenEmbedding,
    SelectiveTokenLMHead,
    SelectiveTrainableTokenAdapter,
    initialize_new_token_embeddings,
    load_optional_state_dict,
    resolve_torch_dtype,
    unwrap_model,
)
from unimapgen.models.qwen3_rc_dinov2_clip_align import (
    ResidualTokenMLPBlock,
    build_grid_centers,
    load_visual_encoder_checkpoint,
)

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


VISUAL_SPECIAL_TOKENS = ("<vis_start>", "<vis_patch>", "<vis_end>")
TRAINABLE_VISUAL_SPECIAL_TOKENS = ("<vis_start>", "<vis_end>")
MODULE_STATE_NAMES = (
    "vision_encoder",
    "visual_token_compressor",
    "visual_norm",
    "visual_projector",
    "geometric_position_mlp",
    "token_alignment",
    "view_type_embeddings",
    "special_token_adapter",
)


def load_visual_projector_with_bridge_v2_fallback(
    module: nn.Module,
    state_dict: Dict[str, torch.Tensor] | None,
    name: str,
) -> None:
    if not state_dict:
        return
    target_state = module.state_dict()

    direct_keys = {str(key) for key in state_dict.keys()}
    if any(key in target_state for key in direct_keys):
        load_optional_state_dict(module, state_dict, name)
        return

    # Bridge-v2 / Bridge-v2-segaux save their projector as fc1/fc2/fc3.
    # Stage-3 JSON SFT still uses the older 2-layer projector (0/2),
    # so we map the main projection path fc1 -> 0 and fc2 -> 2, while
    # intentionally dropping the residual refine fc3 branch.
    mapped: Dict[str, torch.Tensor] = {}
    bridge_v2_map = {
        "fc1.weight": "0.weight",
        "fc1.bias": "0.bias",
        "fc2.weight": "2.weight",
        "fc2.bias": "2.bias",
    }
    for source_key, target_key in bridge_v2_map.items():
        value = state_dict.get(source_key)
        target = target_state.get(target_key)
        if value is None or target is None:
            continue
        if tuple(value.shape) != tuple(target.shape):
            continue
        mapped[target_key] = value
    if mapped:
        load_optional_state_dict(module, mapped, f"{name}_bridgev2_mainpath")
        return

    load_optional_state_dict(module, state_dict, name)


def _resolve_modules_state_path(path_candidate: Path) -> Path | None:
    if path_candidate.is_file():
        return path_candidate
    if not path_candidate.is_dir():
        return None
    for filename in (
        "rc_dinov2_centerline_json_modules.pt",
        "rc_dinov2_centerline_json_modules.pth",
        "rc_dinov2_caption_modules.pt",
        "rc_dinov2_caption_modules.pth",
        "pytorch_model.bin",
        "model.safetensors",
    ):
        candidate = path_candidate / filename
        if candidate.is_file():
            return candidate
    return None


def _load_modules_state_file(path: Path) -> Dict[str, Any]:
    if path.suffix == ".safetensors":
        if load_safetensors_file is None:
            raise RuntimeError(
                f"modules_state requires safetensors but import failed: {_SAFETENSORS_IMPORT_ERROR!r}"
            )
        return dict(load_safetensors_file(str(path)))
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected modules_state type: {type(state)!r}")
    return state


def _unwrap_checkpoint_state(raw_state: Dict[str, Any]) -> Dict[str, Any]:
    if any(name in raw_state for name in MODULE_STATE_NAMES):
        return raw_state
    for wrapper_key in ("state_dict", "model", "module"):
        wrapped = raw_state.get(wrapper_key)
        if isinstance(wrapped, dict):
            return wrapped
    return raw_state


def _extract_prefixed_state_dict(state: Dict[str, Any], module_name: str) -> Dict[str, torch.Tensor]:
    direct = state.get(module_name)
    if isinstance(direct, dict):
        return {str(key): value for key, value in direct.items() if torch.is_tensor(value)}

    prefixes = (
        f"{module_name}.",
        f"module.{module_name}.",
        f"model.{module_name}.",
        f"module.model.{module_name}.",
        f"base_model.{module_name}.",
        f"base_model.model.{module_name}.",
        f"_fsdp_wrapped_module.{module_name}.",
        f"module._fsdp_wrapped_module.{module_name}.",
    )
    extracted: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        key_str = str(key)
        for prefix in prefixes:
            if key_str.startswith(prefix):
                extracted[key_str[len(prefix) :]] = value
                break
    return extracted


def _normalize_modules_state(raw_state: Dict[str, Any]) -> tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Any]]:
    state = _unwrap_checkpoint_state(raw_state)
    modules_state = {
        module_name: _extract_prefixed_state_dict(state, module_name)
        for module_name in MODULE_STATE_NAMES
    }
    summary = {
        module_name: len(module_state)
        for module_name, module_state in modules_state.items()
        if module_state
    }
    return modules_state, {"loaded_module_key_counts": summary}


def _module_at_path(root: nn.Module, path: str) -> nn.Module | None:
    module: Any = root
    for part in path.split("."):
        if not hasattr(module, part):
            return None
        module = getattr(module, part)
    return module if isinstance(module, nn.Module) else None


def _find_transformer_layers(root: nn.Module) -> tuple[str, Sequence[nn.Module]]:
    for path in (
        "model.encoder.layer",
        "model.encoder.layers",
        "model.encoder.blocks",
        "model.blocks",
        "model.layers",
        "encoder.layer",
        "encoder.layers",
        "encoder.blocks",
        "blocks",
        "layers",
    ):
        module = _module_at_path(root, path)
        if isinstance(module, (nn.ModuleList, nn.Sequential)) and len(module) > 0:
            return path, list(module)
    return "", []


def _set_module_trainable(module: nn.Module, enabled: bool) -> int:
    count = 0
    for param in module.parameters():
        param.requires_grad = bool(enabled)
        count += int(param.numel())
    return count


def _freeze_known_unused_vision_parameters(vision_encoder: nn.Module) -> list[str]:
    frozen_names: list[str] = []
    unused_name_fragments = ("mask_token", "pooler")
    for name, param in vision_encoder.named_parameters():
        lowered = str(name).lower()
        if any(fragment in lowered for fragment in unused_name_fragments):
            if param.requires_grad:
                param.requires_grad = False
                frozen_names.append(str(name))
    return frozen_names


def _set_vision_encoder_trainability(
    vision_encoder: nn.Module,
    *,
    freeze_vision_encoder: bool,
    train_last_n_layers: int,
) -> Dict[str, Any]:
    if bool(freeze_vision_encoder):
        for param in vision_encoder.parameters():
            param.requires_grad = False
    else:
        for param in vision_encoder.parameters():
            param.requires_grad = True

    frozen_unused_names = _freeze_known_unused_vision_parameters(vision_encoder)
    info: Dict[str, Any] = {
        "freeze_vision_encoder": bool(freeze_vision_encoder),
        "requested_last_n_layers": int(train_last_n_layers),
        "layer_path": "",
        "total_layers": 0,
        "unfrozen_layers": 0,
        "trainable_params": sum(int(p.numel()) for p in vision_encoder.parameters() if p.requires_grad),
        "frozen_known_unused_params": frozen_unused_names[:20],
        "frozen_known_unused_param_count": len(frozen_unused_names),
    }
    if not bool(freeze_vision_encoder) or int(train_last_n_layers) <= 0:
        return info

    layer_path, layers = _find_transformer_layers(vision_encoder)
    info["layer_path"] = layer_path
    info["total_layers"] = len(layers)
    if not layers:
        return info

    n_layers = min(int(train_last_n_layers), len(layers))
    for layer in layers[-n_layers:]:
        _set_module_trainable(layer, True)
    info["unfrozen_layers"] = n_layers

    # Keep the final ViT norm trainable with the last blocks when it exists.
    for norm_path in (
        "model.layernorm",
        "model.norm",
        "model.fc_norm",
        "model.encoder.layernorm",
        "model.encoder.norm",
    ):
        norm_module = _module_at_path(vision_encoder, norm_path)
        if norm_module is not None:
            _set_module_trainable(norm_module, True)
            info.setdefault("unfrozen_extra_modules", []).append(norm_path)

    frozen_unused_names.extend(_freeze_known_unused_vision_parameters(vision_encoder))
    info["frozen_known_unused_params"] = frozen_unused_names[:20]
    info["frozen_known_unused_param_count"] = len(frozen_unused_names)
    info["trainable_params"] = sum(int(p.numel()) for p in vision_encoder.parameters() if p.requires_grad)
    return info



class LearnedConvVisualTokenCompressor(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        input_grid_size: int,
        output_grid_size: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.input_grid_size = int(input_grid_size)
        self.output_grid_size = int(output_grid_size)
        self.hidden_dim = int(hidden_dim) if int(hidden_dim) > 0 else int(hidden_size)
        self.input_norm = nn.LayerNorm(int(hidden_size))
        self.input_proj = nn.Conv2d(int(hidden_size), int(self.hidden_dim), kernel_size=1)
        blocks: list[nn.Module] = []
        for _ in range(max(1, int(depth))):
            blocks.extend(
                [
                    nn.Conv2d(
                        int(self.hidden_dim),
                        int(self.hidden_dim),
                        kernel_size=3,
                        padding=1,
                        groups=int(self.hidden_dim),
                    ),
                    nn.GELU(),
                    nn.Conv2d(int(self.hidden_dim), int(self.hidden_dim), kernel_size=1),
                    nn.GELU(),
                    nn.Dropout2d(float(dropout)),
                ]
            )
        self.blocks = nn.Sequential(*blocks)
        self.output_proj = nn.Conv2d(int(self.hidden_dim), int(hidden_size), kernel_size=1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected 3D visual tokens, got shape={tuple(tokens.shape)}")
        batch_size, token_count, hidden_size = tokens.shape
        expected_tokens = int(self.input_grid_size) * int(self.input_grid_size)
        if int(token_count) != expected_tokens:
            raise ValueError(
                f"LearnedConvVisualTokenCompressor expected {expected_tokens} tokens "
                f"from {self.input_grid_size}x{self.input_grid_size}, got {int(token_count)}"
            )
        if int(hidden_size) != int(self.hidden_size):
            raise ValueError(f"Expected hidden_size={self.hidden_size}, got {int(hidden_size)}")

        feat = tokens.transpose(1, 2).reshape(
            int(batch_size),
            int(hidden_size),
            int(self.input_grid_size),
            int(self.input_grid_size),
        )
        base = F.adaptive_avg_pool2d(
            feat,
            output_size=(int(self.output_grid_size), int(self.output_grid_size)),
        )

        x = self.input_norm(tokens)
        x = x.transpose(1, 2).reshape_as(feat)
        x = self.input_proj(x)
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(
            x,
            output_size=(int(self.output_grid_size), int(self.output_grid_size)),
        )
        delta = self.output_proj(x)
        out = base + delta
        return out.flatten(2).transpose(1, 2).contiguous()


class Qwen3RCDinoCenterlineJSONSFTModel(nn.Module):
    supports_gradient_checkpointing = True

    def __init__(
        self,
        *,
        model_name_or_path: str,
        tokenizer: Any,
        dinov2_model_name_or_path: str,
        vision_model_name_or_path: str = "",
        vision_patch_size: int = 14,
        vision_num_prefix_tokens: int = -1,
        vision_layer_fusion_indexes: Sequence[int] | str | None = None,
        vision_layer_fusion_type: str = "mean",
        num_visual_tokens: int,
        visual_grid_size: int,
        encoder_visual_grid_size: int = 0,
        num_visual_views: int = 1,
        visual_projector_hidden_dim: int = 4096,
        geometric_mlp_hidden_dim: int = 512,
        token_alignment_hidden_dim: int = 4096,
        token_alignment_num_layers: int = 2,
        token_alignment_dropout: float = 0.0,
        visual_token_compressor: str = "none",
        visual_token_compressor_hidden_dim: int = 512,
        visual_token_compressor_depth: int = 2,
        visual_token_compressor_dropout: float = 0.0,
        use_view_type_embedding: bool = False,
        view_type_embedding_count: int = 2,
        view_type_embedding_init_std: float = 0.02,
        language_model_dtype: str = "auto",
        local_files_only: bool = True,
        freeze_language_model: bool = False,
        freeze_vision_encoder: bool = True,
        vision_train_last_n_layers: int = 0,
        encoder_input_pad_size: int = 0,
        encoder_input_pad_fill_rgb: Sequence[float] = (10.0 / 255.0, 12.0 / 255.0, 18.0 / 255.0),
        visual_encoder_checkpoint_path: str = "",
        modules_state_path: str = "",
        use_lora: bool = True,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if AutoModelForCausalLM is None:
            raise RuntimeError(
                "Qwen3RCDinoCenterlineJSONSFTModel failed to import AutoModelForCausalLM. "
                f"Original import error: {_TRANSFORMERS_IMPORT_ERROR!r}"
            )
        if len(tuple(encoder_input_pad_fill_rgb)) != 3:
            raise ValueError("encoder_input_pad_fill_rgb must contain exactly 3 values.")

        resolved_model_path = resolve_hf_snapshot_path(str(model_name_or_path))
        default_modules_state_path = Path(resolved_model_path) / "rc_dinov2_centerline_json_modules.pt"
        if not default_modules_state_path.is_file():
            pth_modules_state_path = Path(resolved_model_path) / "rc_dinov2_centerline_json_modules.pth"
            if pth_modules_state_path.is_file():
                default_modules_state_path = pth_modules_state_path
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
                f"[qwen3-rc-json] adapter checkpoint detected: base_model={language_model_source} adapter={adapter_checkpoint_dir}",
                flush=True,
            )

        self.num_visual_tokens = int(num_visual_tokens)
        self.visual_grid_size = int(visual_grid_size)
        self.encoder_visual_grid_size = int(encoder_visual_grid_size) if int(encoder_visual_grid_size) > 0 else int(self.visual_grid_size)
        self.encoder_tokens_per_view = int(self.encoder_visual_grid_size) * int(self.encoder_visual_grid_size)
        self.num_visual_views = max(1, int(num_visual_views))
        self.tokens_per_view = int(self.visual_grid_size) * int(self.visual_grid_size)
        self.use_view_type_embedding = bool(use_view_type_embedding)
        self.encoder_input_pad_size = max(0, int(encoder_input_pad_size))
        self.register_buffer(
            "encoder_input_pad_fill",
            torch.tensor(list(encoder_input_pad_fill_rgb), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        expected_visual_tokens = int(self.tokens_per_view) * int(self.num_visual_views)
        if expected_visual_tokens != int(self.num_visual_tokens):
            raise ValueError(
                f"visual_grid_size={self.visual_grid_size} and num_visual_views={self.num_visual_views} "
                f"produce {expected_visual_tokens} tokens, but num_visual_tokens={self.num_visual_tokens}"
            )
        if int(self.encoder_visual_grid_size) < int(self.visual_grid_size):
            raise ValueError(
                f"encoder_visual_grid_size={self.encoder_visual_grid_size} must be >= visual_grid_size={self.visual_grid_size}"
            )
        self.register_buffer(
            "grid_centers",
            build_grid_centers(self.visual_grid_size).unsqueeze(0),
            persistent=False,
        )

        lm_torch_dtype = resolve_torch_dtype(str(language_model_dtype))
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_source,
            local_files_only=bool(local_files_only),
            trust_remote_code=True,
            torch_dtype=lm_torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.config = self.language_model.config
        old_vocab_size = int(self.language_model.get_input_embeddings().weight.shape[0])
        new_vocab_size = int(len(tokenizer))
        if new_vocab_size != old_vocab_size:
            self.language_model.resize_token_embeddings(new_vocab_size)
            initialize_new_token_embeddings(self.language_model, old_vocab_size=old_vocab_size)
        self.special_token_adapter: SelectiveTrainableTokenAdapter | None = None
        self.visual_special_token_ids = tuple(
            int(tokenizer.convert_tokens_to_ids(token)) for token in VISUAL_SPECIAL_TOKENS
        )
        if any(int(token_id) < 0 for token_id in self.visual_special_token_ids):
            raise ValueError(
                f"Tokenizer is missing required visual special tokens: {VISUAL_SPECIAL_TOKENS}"
            )
        self.trainable_visual_special_token_ids = tuple(
            int(tokenizer.convert_tokens_to_ids(token)) for token in TRAINABLE_VISUAL_SPECIAL_TOKENS
        )
        if any(int(token_id) < 0 for token_id in self.trainable_visual_special_token_ids):
            raise ValueError(
                f"Tokenizer is missing required trainable visual special tokens: {TRAINABLE_VISUAL_SPECIAL_TOKENS}"
            )
        self._enable_selective_trainable_tokens(self.trainable_visual_special_token_ids)
        self._set_selective_token_trainable(True)

        self.hidden_size = int(self.language_model.get_input_embeddings().weight.shape[1])
        self.vis_patch_token_id = int(tokenizer.convert_tokens_to_ids("<vis_patch>"))
        if self.vis_patch_token_id < 0:
            raise ValueError("Tokenizer is missing required <vis_patch> token.")

        resolved_vision_model_name_or_path = str(vision_model_name_or_path).strip() or str(dinov2_model_name_or_path)
        prefix_tokens = None if int(vision_num_prefix_tokens) < 0 else int(vision_num_prefix_tokens)
        self.vision_encoder = SatelliteEncoder(
            model_name=resolved_vision_model_name_or_path,
            local_files_only=bool(local_files_only),
            use_fallback=False,
            out_hw=None,
            patch_size=int(vision_patch_size),
            drop_cls_token=True,
            num_prefix_tokens=prefix_tokens,
            normalize_input=True,
            vision_layer_fusion_indexes=vision_layer_fusion_indexes,
            vision_layer_fusion_type=str(vision_layer_fusion_type),
        )
        has_visual_encoder_checkpoint = bool(str(visual_encoder_checkpoint_path).strip())
        if has_visual_encoder_checkpoint:
            load_visual_encoder_checkpoint(
                self.vision_encoder,
                checkpoint_path=str(visual_encoder_checkpoint_path).strip(),
            )

        self.visual_norm = nn.LayerNorm(int(self.vision_encoder.hidden_size))
        compressor_name = str(visual_token_compressor).strip().lower()
        if compressor_name in {"", "none", "identity"}:
            if int(self.encoder_tokens_per_view) != int(self.tokens_per_view):
                raise ValueError(
                    "visual_token_compressor=none requires encoder_tokens_per_view "
                    f"({self.encoder_tokens_per_view}) == tokens_per_view ({self.tokens_per_view})"
                )
            self.visual_token_compressor: nn.Module = nn.Identity()
        elif compressor_name in {"learned_conv", "conv", "learned"}:
            self.visual_token_compressor = LearnedConvVisualTokenCompressor(
                hidden_size=int(self.vision_encoder.hidden_size),
                input_grid_size=int(self.encoder_visual_grid_size),
                output_grid_size=int(self.visual_grid_size),
                hidden_dim=int(visual_token_compressor_hidden_dim),
                depth=int(visual_token_compressor_depth),
                dropout=float(visual_token_compressor_dropout),
            )
        else:
            raise ValueError(f"Unsupported visual_token_compressor: {visual_token_compressor!r}")
        self.visual_token_compressor_name = compressor_name or "none"
        self.visual_projector = nn.Sequential(
            nn.Linear(int(self.vision_encoder.hidden_size), int(visual_projector_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(visual_projector_hidden_dim), self.hidden_size),
        )
        self.geometric_position_mlp = nn.Sequential(
            nn.Linear(2, int(geometric_mlp_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(geometric_mlp_hidden_dim), self.hidden_size),
        )
        self.token_alignment = nn.ModuleList(
            [
                ResidualTokenMLPBlock(
                    hidden_size=int(self.hidden_size),
                    mlp_hidden_dim=int(token_alignment_hidden_dim),
                    dropout=float(token_alignment_dropout),
                )
                for _ in range(max(1, int(token_alignment_num_layers)))
            ]
        )
        self.view_type_embeddings: nn.Embedding | None = None
        if self.use_view_type_embedding:
            embedding_count = max(int(view_type_embedding_count), int(self.num_visual_views))
            self.view_type_embeddings = nn.Embedding(embedding_count, int(self.hidden_size))
            nn.init.normal_(self.view_type_embeddings.weight, mean=0.0, std=float(view_type_embedding_init_std))

        modules_path_candidate = (
            Path(str(modules_state_path).strip()).expanduser()
            if str(modules_state_path).strip()
            else default_modules_state_path
        )
        modules_state_file = _resolve_modules_state_path(modules_path_candidate)
        if modules_state_file is not None:
            modules_state_raw = _load_modules_state_file(modules_state_file)
            modules_state, modules_state_summary = _normalize_modules_state(modules_state_raw)
            modules_state_summary["path"] = str(modules_state_file)
            print(
                f"[qwen3-rc-json] modules_state={json.dumps(modules_state_summary, ensure_ascii=False)}",
                flush=True,
            )
            if has_visual_encoder_checkpoint:
                if modules_state.get("vision_encoder"):
                    print(
                        "[qwen3-rc-json] skip modules_state vision_encoder because visual_encoder_checkpoint_path is set",
                        flush=True,
                    )
            else:
                load_optional_state_dict(self.vision_encoder, modules_state.get("vision_encoder"), "vision_encoder")
            load_optional_state_dict(
                self.visual_token_compressor,
                modules_state.get("visual_token_compressor"),
                "visual_token_compressor",
            )
            load_optional_state_dict(self.visual_norm, modules_state.get("visual_norm"), "visual_norm")
            load_visual_projector_with_bridge_v2_fallback(
                self.visual_projector,
                modules_state.get("visual_projector"),
                "visual_projector",
            )
            load_optional_state_dict(
                self.geometric_position_mlp,
                modules_state.get("geometric_position_mlp"),
                "geometric_position_mlp",
            )
            load_optional_state_dict(self.token_alignment, modules_state.get("token_alignment"), "token_alignment")
            if self.view_type_embeddings is not None:
                load_optional_state_dict(
                    self.view_type_embeddings,
                    modules_state.get("view_type_embeddings"),
                    "view_type_embeddings",
                )
            if self.special_token_adapter is not None:
                load_optional_state_dict(
                    self.special_token_adapter,
                    modules_state.get("special_token_adapter"),
                    "special_token_adapter",
                )

        if bool(use_lora) and adapter_checkpoint_dir is not None:
            if PeftModel is None:
                raise RuntimeError(
                    "Qwen3RCDinoCenterlineJSONSFTModel needs peft.PeftModel to restore adapter checkpoints, "
                    f"but peft import failed: {_PEFT_IMPORT_ERROR!r}"
                )
            self.language_model = PeftModel.from_pretrained(
                self.language_model,
                str(adapter_checkpoint_dir),
                is_trainable=True,
            )
            self._set_selective_token_trainable(True)
            print(
                f"[qwen3-rc-json] restored adapter checkpoint via PeftModel.from_pretrained: {adapter_checkpoint_dir}",
                flush=True,
            )
        elif bool(use_lora):
            if LoraConfig is None or get_peft_model is None:
                raise RuntimeError(
                    "Qwen3RCDinoCenterlineJSONSFTModel requested LoRA, but peft could not be imported. "
                    f"Original import error: {_PEFT_IMPORT_ERROR!r}"
                )
            peft_cfg = LoraConfig(
                r=int(lora_rank),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            self.language_model = get_peft_model(self.language_model, peft_cfg)
            self._set_selective_token_trainable(True)
            try:
                self.language_model.print_trainable_parameters()
            except Exception:
                pass
        elif bool(freeze_language_model):
            for param in self.language_model.parameters():
                param.requires_grad = False
            self._set_selective_token_trainable(True)

        vision_trainability = _set_vision_encoder_trainability(
            self.vision_encoder,
            freeze_vision_encoder=bool(freeze_vision_encoder),
            train_last_n_layers=int(vision_train_last_n_layers),
        )
        print(f"[qwen3-rc-json] vision_trainability={json.dumps(vision_trainability, ensure_ascii=False)}", flush=True)
        compressor_summary = {
            "name": self.visual_token_compressor_name,
            "encoder_visual_grid_size": int(self.encoder_visual_grid_size),
            "encoder_tokens_per_view": int(self.encoder_tokens_per_view),
            "visual_grid_size": int(self.visual_grid_size),
            "tokens_per_view": int(self.tokens_per_view),
            "num_visual_views": int(self.num_visual_views),
            "num_visual_tokens": int(self.num_visual_tokens),
        }
        print(
            f"[qwen3-rc-json] visual_token_compressor={json.dumps(compressor_summary, ensure_ascii=False)}",
            flush=True,
        )

        if bool(gradient_checkpointing) and hasattr(self.language_model, "gradient_checkpointing_enable"):
            try:
                self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.language_model.gradient_checkpointing_enable()
            if hasattr(self.language_model, "config") and hasattr(self.language_model.config, "use_cache"):
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

    def _set_selective_token_trainable(self, enabled: bool) -> None:
        if self.special_token_adapter is not None:
            self.special_token_adapter.set_trainable(bool(enabled))

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

    def _maybe_center_pad_image(self, image: torch.Tensor) -> torch.Tensor:
        target_size = max(int(self.encoder_input_pad_size), int(image.shape[-2]), int(image.shape[-1]))
        if target_size == int(image.shape[-2]) and target_size == int(image.shape[-1]):
            return image
        batch, channels, height, width = image.shape
        top = max(0, (target_size - int(height)) // 2)
        left = max(0, (target_size - int(width)) // 2)
        padded = self.encoder_input_pad_fill.to(dtype=image.dtype, device=image.device).expand(
            int(batch),
            int(channels),
            int(target_size),
            int(target_size),
        ).clone()
        padded[:, :, top : top + int(height), left : left + int(width)] = image
        return padded

    def build_visual_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim == 5:
            batch, views, channels, height, width = pixel_values.shape
            if int(views) != int(self.num_visual_views):
                raise ValueError(
                    f"Expected {self.num_visual_views} visual views, got pixel_values shape={tuple(pixel_values.shape)}"
                )
            flat_pixels = pixel_values.reshape(int(batch) * int(views), int(channels), int(height), int(width))
            projected = self._build_single_view_visual_embeddings(flat_pixels)
            projected = projected.reshape(int(batch), int(views), int(self.tokens_per_view), int(projected.shape[-1]))
            if self.view_type_embeddings is not None:
                view_ids = torch.arange(int(views), device=projected.device, dtype=torch.long)
                max_id = int(self.view_type_embeddings.num_embeddings) - 1
                view_ids = view_ids.clamp(max=max_id)
                view_bias = self.view_type_embeddings(view_ids).to(dtype=projected.dtype).view(1, int(views), 1, -1)
                projected = projected + view_bias
            projected = projected.reshape(int(batch), int(views) * int(self.tokens_per_view), int(projected.shape[-1]))
            for block in self.token_alignment:
                projected = block(projected)
            return projected
        if pixel_values.ndim != 4:
            raise ValueError(f"Expected pixel_values to be 4D or 5D, got shape={tuple(pixel_values.shape)}")
        projected = self._build_single_view_visual_embeddings(pixel_values)
        if self.view_type_embeddings is not None:
            view_ids = torch.zeros((1,), device=projected.device, dtype=torch.long)
            projected = projected + self.view_type_embeddings(view_ids).to(dtype=projected.dtype).view(1, 1, -1)
        for block in self.token_alignment:
            projected = block(projected)
        return projected

    def _build_single_view_visual_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        encoder_input = self._maybe_center_pad_image(pixel_values)
        vision_outputs = self.vision_encoder.forward_features(encoder_input)
        visual_tokens = vision_outputs["tokens"]
        if visual_tokens.ndim != 3:
            raise ValueError(f"Expected visual tokens to be 3D, got shape={tuple(visual_tokens.shape)}")
        if int(visual_tokens.shape[1]) != self.encoder_tokens_per_view:
            raise ValueError(
                f"Visual encoder produced {int(visual_tokens.shape[1])} tokens, expected {self.encoder_tokens_per_view}"
            )
        visual_tokens = self.visual_token_compressor(visual_tokens)
        if int(visual_tokens.shape[1]) != self.tokens_per_view:
            raise ValueError(
                f"Visual token compressor produced {int(visual_tokens.shape[1])} tokens, expected {self.tokens_per_view}"
            )
        projected = self.visual_projector(self.visual_norm(visual_tokens))
        pos = self.grid_centers.to(device=projected.device, dtype=projected.dtype)
        projected = projected + self.geometric_position_mlp(pos.expand(int(projected.shape[0]), -1, -1))
        return projected

    def inject_visual_embeddings(
        self,
        *,
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
        visual_embeddings = self.build_visual_embeddings(pixel_values)
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
            use_cache=False,
            return_dict=True,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def save_qwen3_rc_dinov2_centerline_json_modules(
    model: nn.Module,
    output_dir: str | Path,
    tokenizer: Any | None = None,
) -> None:
    if _distributed_rank() != 0:
        _distributed_barrier()
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_for_save = unwrap_model(model)
    language_model = getattr(model_for_save, "language_model", None)
    if language_model is None:
        raise AttributeError("Wrapped model does not expose language_model.")
    language_model.save_pretrained(str(output_path), safe_serialization=False)
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_path))
    torch.save(
        {
            "vision_encoder": model_for_save.vision_encoder.state_dict(),
            "visual_token_compressor": model_for_save.visual_token_compressor.state_dict(),
            "visual_norm": model_for_save.visual_norm.state_dict(),
            "visual_projector": model_for_save.visual_projector.state_dict(),
            "geometric_position_mlp": model_for_save.geometric_position_mlp.state_dict(),
            "token_alignment": model_for_save.token_alignment.state_dict(),
            "view_type_embeddings": (
                model_for_save.view_type_embeddings.state_dict()
                if getattr(model_for_save, "view_type_embeddings", None) is not None
                else None
            ),
            "special_token_adapter": (
                model_for_save.special_token_adapter.state_dict()
                if model_for_save.special_token_adapter is not None
                else None
            ),
            "encoder_input_pad_size": int(model_for_save.encoder_input_pad_size),
            "encoder_visual_grid_size": int(model_for_save.encoder_visual_grid_size),
            "encoder_tokens_per_view": int(model_for_save.encoder_tokens_per_view),
            "visual_grid_size": int(model_for_save.visual_grid_size),
            "num_visual_tokens": int(model_for_save.num_visual_tokens),
            "tokens_per_view": int(getattr(model_for_save, "tokens_per_view", model_for_save.num_visual_tokens)),
            "num_visual_views": int(getattr(model_for_save, "num_visual_views", 1)),
            "use_view_type_embedding": bool(getattr(model_for_save, "use_view_type_embedding", False)),
        },
        str(output_path / "rc_dinov2_centerline_json_modules.pt"),
    )
    _distributed_barrier()
