from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

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


VISUAL_SPECIAL_TOKENS = ("<vis_start>", "<vis_patch>", "<vis_end>")
TRAINABLE_VISUAL_SPECIAL_TOKENS = ("<vis_start>", "<vis_end>")


class Qwen3RCDinoCenterlineJSONSFTModel(nn.Module):
    supports_gradient_checkpointing = True

    def __init__(
        self,
        *,
        model_name_or_path: str,
        tokenizer: Any,
        dinov2_model_name_or_path: str,
        num_visual_tokens: int,
        visual_grid_size: int,
        visual_projector_hidden_dim: int = 4096,
        geometric_mlp_hidden_dim: int = 512,
        token_alignment_hidden_dim: int = 4096,
        token_alignment_num_layers: int = 2,
        token_alignment_dropout: float = 0.0,
        language_model_dtype: str = "auto",
        local_files_only: bool = True,
        freeze_language_model: bool = False,
        freeze_vision_encoder: bool = True,
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
        self.encoder_input_pad_size = max(0, int(encoder_input_pad_size))
        self.register_buffer(
            "encoder_input_pad_fill",
            torch.tensor(list(encoder_input_pad_fill_rgb), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        if int(self.visual_grid_size) * int(self.visual_grid_size) != int(self.num_visual_tokens):
            raise ValueError(
                f"visual_grid_size={self.visual_grid_size} does not match num_visual_tokens={self.num_visual_tokens}"
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
        old_vocab_size = int(self.language_model.get_input_embeddings().weight.shape[0])
        new_vocab_size = int(len(tokenizer))
        if new_vocab_size != old_vocab_size:
            self.language_model.resize_token_embeddings(new_vocab_size)
            initialize_new_token_embeddings(self.language_model, old_vocab_size=old_vocab_size)
        # SFT v1 依旧不训练 <vis_patch> 本身，只让两侧边界 token 学到视觉块边界语义。
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

        self.vision_encoder = SatelliteEncoder(
            model_name=str(dinov2_model_name_or_path),
            local_files_only=bool(local_files_only),
            use_fallback=False,
            out_hw=None,
            patch_size=14,
            drop_cls_token=True,
            normalize_input=True,
        )
        if str(visual_encoder_checkpoint_path).strip():
            load_visual_encoder_checkpoint(
                self.vision_encoder,
                checkpoint_path=str(visual_encoder_checkpoint_path).strip(),
            )

        self.visual_norm = nn.LayerNorm(int(self.vision_encoder.hidden_size))
        # 这里复用 Stage 2 的视觉桥，SFT 只是继续在 bridge 之上学习 JSON 生成。
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

        modules_path_candidate = (
            Path(str(modules_state_path).strip()).expanduser()
            if str(modules_state_path).strip()
            else default_modules_state_path
        )
        if modules_path_candidate.is_file():
            modules_state = torch.load(str(modules_path_candidate), map_location="cpu", weights_only=False)
            if not isinstance(modules_state, dict):
                raise TypeError(f"Unexpected modules_state type: {type(modules_state)!r}")
            load_optional_state_dict(self.vision_encoder, modules_state.get("vision_encoder"), "vision_encoder")
            load_optional_state_dict(self.visual_norm, modules_state.get("visual_norm"), "visual_norm")
            load_optional_state_dict(self.visual_projector, modules_state.get("visual_projector"), "visual_projector")
            load_optional_state_dict(
                self.geometric_position_mlp,
                modules_state.get("geometric_position_mlp"),
                "geometric_position_mlp",
            )
            load_optional_state_dict(self.token_alignment, modules_state.get("token_alignment"), "token_alignment")
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
            # 这版只给 Qwen 打 LoRA，避免 full fine-tune 破坏已经稳定的语言能力。
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

        if bool(freeze_vision_encoder):
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

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
        encoder_input = self._maybe_center_pad_image(pixel_values)
        vision_outputs = self.vision_encoder.forward_features(encoder_input)
        visual_tokens = vision_outputs["tokens"]
        if visual_tokens.ndim != 3:
            raise ValueError(f"Expected visual tokens to be 3D, got shape={tuple(visual_tokens.shape)}")
        if int(visual_tokens.shape[1]) != self.num_visual_tokens:
            raise ValueError(
                f"Visual encoder produced {int(visual_tokens.shape[1])} tokens, expected {self.num_visual_tokens}"
            )
        projected = self.visual_projector(self.visual_norm(visual_tokens))
        pos = self.grid_centers.to(device=projected.device, dtype=projected.dtype)
        projected = projected + self.geometric_position_mlp(pos.expand(int(projected.shape[0]), -1, -1))
        for block in self.token_alignment:
            projected = block(projected)
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
        # 和 Stage 2 一样，视觉信息仍然是直接替换到 <vis_patch> 槽位中。
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


def save_qwen3_rc_dinov2_centerline_json_modules(
    model: nn.Module,
    output_dir: str | Path,
    tokenizer: Any | None = None,
) -> None:
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
            "visual_norm": model_for_save.visual_norm.state_dict(),
            "visual_projector": model_for_save.visual_projector.state_dict(),
            "geometric_position_mlp": model_for_save.geometric_position_mlp.state_dict(),
            "token_alignment": model_for_save.token_alignment.state_dict(),
            "special_token_adapter": (
                model_for_save.special_token_adapter.state_dict()
                if model_for_save.special_token_adapter is not None
                else None
            ),
            "encoder_input_pad_size": int(model_for_save.encoder_input_pad_size),
            "visual_grid_size": int(model_for_save.visual_grid_size),
            "num_visual_tokens": int(model_for_save.num_visual_tokens),
        },
        str(output_path / "rc_dinov2_centerline_json_modules.pt"),
    )
