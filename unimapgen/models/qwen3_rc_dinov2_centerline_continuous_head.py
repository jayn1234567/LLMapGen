from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from unimapgen.models.hf_utils import resolve_hf_snapshot_path
from unimapgen.models.qwen3_rc_centerline_16745style import (
    load_optional_state_dict,
    unwrap_model,
)
from unimapgen.models.qwen3_rc_dinov2_centerline_json_sft import (
    Qwen3RCDinoCenterlineJSONSFTModel,
)


class Qwen3RCDinoCenterlineContinuousHeadModel(Qwen3RCDinoCenterlineJSONSFTModel):
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
        coord_head_hidden_dim: int = 1024,
        coord_use_sigmoid: bool = True,
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
        resolved_model_path = resolve_hf_snapshot_path(str(model_name_or_path))
        default_modules_state_path = Path(resolved_model_path) / "rc_dinov2_centerline_continuous_head_modules.pt"
        modules_state_source = str(modules_state_path).strip()
        if not modules_state_source and default_modules_state_path.is_file():
            modules_state_source = str(default_modules_state_path)

        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            dinov2_model_name_or_path=dinov2_model_name_or_path,
            num_visual_tokens=num_visual_tokens,
            visual_grid_size=visual_grid_size,
            visual_projector_hidden_dim=visual_projector_hidden_dim,
            geometric_mlp_hidden_dim=geometric_mlp_hidden_dim,
            token_alignment_hidden_dim=token_alignment_hidden_dim,
            token_alignment_num_layers=token_alignment_num_layers,
            token_alignment_dropout=token_alignment_dropout,
            language_model_dtype=language_model_dtype,
            local_files_only=local_files_only,
            freeze_language_model=freeze_language_model,
            freeze_vision_encoder=freeze_vision_encoder,
            encoder_input_pad_size=encoder_input_pad_size,
            encoder_input_pad_fill_rgb=encoder_input_pad_fill_rgb,
            visual_encoder_checkpoint_path=visual_encoder_checkpoint_path,
            modules_state_path=modules_state_source,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.coord_use_sigmoid = bool(coord_use_sigmoid)
        self.coord_head = nn.Sequential(
            nn.Linear(self.hidden_size, int(coord_head_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(coord_head_hidden_dim), 2),
        )

        if modules_state_source:
            modules_path = Path(modules_state_source).expanduser()
            if modules_path.is_file():
                modules_state = torch.load(str(modules_path), map_location="cpu", weights_only=False)
                if isinstance(modules_state, dict):
                    load_optional_state_dict(self.coord_head, modules_state.get("coord_head"), "coord_head")

    def predict_coordinates(self, hidden_states: torch.Tensor) -> torch.Tensor:
        coord_dtype = next(self.coord_head.parameters()).dtype
        coord_pred = self.coord_head(hidden_states.to(dtype=coord_dtype))
        if bool(self.coord_use_sigmoid):
            coord_pred = torch.sigmoid(coord_pred)
        return coord_pred

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        vis_patch_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
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
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        coord_pred = self.predict_coordinates(outputs.hidden_states[-1])
        result: dict[str, torch.Tensor] = {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "coord_pred": coord_pred,
        }
        if bool(output_hidden_states):
            result["hidden_states"] = outputs.hidden_states
        return result


def save_qwen3_rc_dinov2_centerline_continuous_head_modules(
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
            "coord_head": model_for_save.coord_head.state_dict(),
            "special_token_adapter": (
                model_for_save.special_token_adapter.state_dict()
                if model_for_save.special_token_adapter is not None
                else None
            ),
            "encoder_input_pad_size": int(model_for_save.encoder_input_pad_size),
            "visual_grid_size": int(model_for_save.visual_grid_size),
            "num_visual_tokens": int(model_for_save.num_visual_tokens),
            "coord_use_sigmoid": bool(model_for_save.coord_use_sigmoid),
        },
        str(output_path / "rc_dinov2_centerline_continuous_head_modules.pt"),
    )


__all__ = [
    "Qwen3RCDinoCenterlineContinuousHeadModel",
    "save_qwen3_rc_dinov2_centerline_continuous_head_modules",
]
