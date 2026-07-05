from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from unimapgen.models.encoders.satellite_encoder import SatelliteEncoder
from unimapgen.models.hf_utils import resolve_hf_snapshot_path
from unimapgen.models.qwen3_rc_centerline_16745style import (
    filter_state_dict_by_shape,
    load_optional_state_dict,
    resolve_torch_dtype,
    unwrap_model,
)

try:
    from torch.distributed.nn.functional import all_gather as dist_all_gather_with_grad

    _DIST_NN_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    dist_all_gather_with_grad = None
    _DIST_NN_IMPORT_ERROR = exc

try:
    from transformers import AutoModelForCausalLM

    _TRANSFORMERS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    AutoModelForCausalLM = None
    _TRANSFORMERS_IMPORT_ERROR = exc


def extract_prefixed_state_dict(state_dict: Dict[str, torch.Tensor], prefixes: Sequence[str]) -> Dict[str, torch.Tensor]:
    extracted: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        key_str = str(key)
        for prefix in prefixes:
            if key_str.startswith(prefix):
                extracted[key_str[len(prefix) :]] = value
                break
    return extracted


def _tensor_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        return {}
    direct = {str(key): value for key, value in payload.items() if torch.is_tensor(value)}
    if direct:
        return direct
    for key in (
        "vision_encoder",
        "encoder",
        "backbone",
        "model",
        "state_dict",
        "module",
        "net",
    ):
        nested = payload.get(key)
        nested_state = _tensor_state_dict(nested)
        if nested_state:
            return nested_state
    return {}


def _strip_common_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = (
        "module.",
        "_orig_mod.",
        "base_model.model.",
        "base_model.",
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
                state = {str(key)[len(prefix) :]: value for key, value in state.items()}
                changed = True
                break
    return state


def _candidate_encoder_states(raw_state: Dict[str, torch.Tensor]) -> list[tuple[str, Dict[str, torch.Tensor]]]:
    stripped = _strip_common_prefixes(raw_state)
    candidates: list[tuple[str, Dict[str, torch.Tensor]]] = [
        ("raw", raw_state),
        ("stripped", stripped),
    ]
    if stripped and not all(str(key).startswith("model.") for key in stripped):
        candidates.append(("stripped_plus_model", {f"model.{key}": value for key, value in stripped.items()}))
    if raw_state and not all(str(key).startswith("model.") for key in raw_state):
        candidates.append(("raw_plus_model", {f"model.{key}": value for key, value in raw_state.items()}))
    deduped: list[tuple[str, Dict[str, torch.Tensor]]] = []
    seen = set()
    for name, state in candidates:
        signature = tuple(sorted(state.keys())[:20])
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append((name, state))
    return deduped


def _shape_match_score(module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> tuple[int, int]:
    target_state = module.state_dict()
    filtered, _, _ = filter_state_dict_by_shape(target_state, state_dict)
    matched_tensors = len(filtered)
    matched_numel = sum(int(value.numel()) for value in filtered.values())
    return matched_tensors, matched_numel


def _select_encoder_state(vision_encoder: nn.Module, raw_state: Dict[str, torch.Tensor]) -> tuple[str, Dict[str, torch.Tensor]]:
    best_name = ""
    best_state: Dict[str, torch.Tensor] = {}
    best_score = (0, 0)
    for name, candidate in _candidate_encoder_states(raw_state):
        score = _shape_match_score(vision_encoder, candidate)
        if score > best_score:
            best_name = name
            best_state = candidate
            best_score = score
    if best_score[0] <= 0:
        lora_keys = [key for key in raw_state if "lora" in str(key).lower()]
        hint = ""
        if lora_keys:
            hint = (
                " The checkpoint looks like a LoRA-only adapter because it contains LoRA keys. "
                "Merge the LoRA adapter into the base DINOv3 weights first, or provide a checkpoint "
                "that contains full vision encoder weights."
            )
        raise ValueError(f"Unable to match any visual encoder weights by shape.{hint}")
    return best_name, best_state


def load_visual_encoder_checkpoint(
    vision_encoder: nn.Module,
    checkpoint_path: str | Path,
) -> None:
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Visual encoder checkpoint not found: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    encoder_state = _tensor_state_dict(state)
    if not encoder_state:
        raise ValueError(f"Unable to extract encoder weights from checkpoint: {ckpt_path}")
    selected_name, encoder_state = _select_encoder_state(vision_encoder, encoder_state)
    print(
        f"[visual-checkpoint] selected_state={selected_name} tensors={len(encoder_state)} path={ckpt_path}",
        flush=True,
    )
    load_optional_state_dict(vision_encoder, encoder_state, f"visual_encoder_checkpoint[{ckpt_path}]")


def build_grid_centers(grid_size: int) -> torch.Tensor:
    size = int(grid_size)
    coords = []
    for row in range(size):
        for col in range(size):
            cx = 2.0 * (float(col) + 0.5) / float(size) - 1.0
            cy = 2.0 * (float(row) + 0.5) / float(size) - 1.0
            coords.append([cx, cy])
    return torch.tensor(coords, dtype=torch.float32)


class ResidualTokenMLPBlock(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(hidden_size))
        self.fc1 = nn.Linear(int(hidden_size), int(mlp_hidden_dim))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(mlp_hidden_dim), int(hidden_size))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class Qwen3RCDinoClipAlignModel(nn.Module):
    def __init__(
        self,
        *,
        model_name_or_path: str,
        dinov2_model_name_or_path: str,
        num_visual_tokens: int,
        visual_grid_size: int,
        contrastive_dim: int = 1024,
        visual_projector_hidden_dim: int = 4096,
        geometric_mlp_hidden_dim: int = 512,
        token_alignment_hidden_dim: int = 4096,
        token_alignment_num_layers: int = 2,
        token_alignment_dropout: float = 0.0,
        language_model_dtype: str = "auto",
        local_files_only: bool = True,
        freeze_language_model: bool = True,
        freeze_vision_encoder: bool = True,
        encoder_input_pad_size: int = 0,
        encoder_input_pad_fill_rgb: Sequence[float] = (10.0 / 255.0, 12.0 / 255.0, 18.0 / 255.0),
        visual_encoder_checkpoint_path: str = "",
        modules_state_path: str = "",
    ) -> None:
        super().__init__()
        if AutoModelForCausalLM is None:
            raise RuntimeError(
                "Qwen3RCDinoClipAlignModel failed to import AutoModelForCausalLM. "
                f"Original import error: {_TRANSFORMERS_IMPORT_ERROR!r}"
            )
        if len(tuple(encoder_input_pad_fill_rgb)) != 3:
            raise ValueError("encoder_input_pad_fill_rgb must contain exactly 3 values.")
        if int(visual_grid_size) * int(visual_grid_size) != int(num_visual_tokens):
            raise ValueError(
                f"visual_grid_size={visual_grid_size} does not match num_visual_tokens={num_visual_tokens}"
            )

        self.num_visual_tokens = int(num_visual_tokens)
        self.visual_grid_size = int(visual_grid_size)
        self.encoder_input_pad_size = max(0, int(encoder_input_pad_size))
        self.contrastive_dim = int(contrastive_dim)
        self.register_buffer(
            "encoder_input_pad_fill",
            torch.tensor(list(encoder_input_pad_fill_rgb), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "grid_centers",
            build_grid_centers(self.visual_grid_size).unsqueeze(0),
            persistent=False,
        )

        resolved_model_path = resolve_hf_snapshot_path(str(model_name_or_path))
        lm_torch_dtype = resolve_torch_dtype(str(language_model_dtype))
        self.language_model = AutoModelForCausalLM.from_pretrained(
            resolved_model_path,
            local_files_only=bool(local_files_only),
            trust_remote_code=True,
            torch_dtype=lm_torch_dtype,
            low_cpu_mem_usage=True,
        )
        if bool(freeze_language_model):
            for param in self.language_model.parameters():
                param.requires_grad = False

        self.hidden_size = int(self.language_model.get_input_embeddings().weight.shape[1])
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
        if bool(freeze_vision_encoder):
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        self.visual_norm = nn.LayerNorm(int(self.vision_encoder.hidden_size))
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
        self.readout_norm = nn.LayerNorm(self.hidden_size)
        self.image_projection = nn.Linear(self.hidden_size, self.contrastive_dim)
        self.text_projection = nn.Linear(self.hidden_size, self.contrastive_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32))

        modules_path = Path(str(modules_state_path).strip()).expanduser() if str(modules_state_path).strip() else None
        if modules_path is not None and modules_path.is_file():
            modules_state = torch.load(str(modules_path), map_location="cpu", weights_only=False)
            if not isinstance(modules_state, dict):
                raise TypeError(f"Unexpected modules_state type: {type(modules_state)!r}")
            load_optional_state_dict(self.visual_norm, modules_state.get("visual_norm"), "visual_norm")
            load_optional_state_dict(self.visual_projector, modules_state.get("visual_projector"), "visual_projector")
            load_optional_state_dict(
                self.geometric_position_mlp,
                modules_state.get("geometric_position_mlp"),
                "geometric_position_mlp",
            )
            load_optional_state_dict(self.token_alignment, modules_state.get("token_alignment"), "token_alignment")
            load_optional_state_dict(self.readout_norm, modules_state.get("readout_norm"), "readout_norm")
            load_optional_state_dict(self.image_projection, modules_state.get("image_projection"), "image_projection")
            load_optional_state_dict(self.text_projection, modules_state.get("text_projection"), "text_projection")
            logit_scale = modules_state.get("logit_scale")
            if isinstance(logit_scale, torch.Tensor):
                with torch.no_grad():
                    self.logit_scale.copy_(logit_scale.to(dtype=self.logit_scale.dtype))

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

    def build_aligned_visual_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
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

    def encode_image(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        aligned_tokens = self.build_aligned_visual_tokens(pixel_values)
        pooled = aligned_tokens.mean(dim=1)
        pooled = self.readout_norm(pooled)
        image_embeddings = F.normalize(self.image_projection(pooled), dim=-1)
        return image_embeddings, aligned_tokens

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        if not outputs.hidden_states:
            raise ValueError("Language model did not return hidden_states.")
        hidden_states = outputs.hidden_states[-1]
        lengths = attention_mask.to(dtype=torch.long).sum(dim=1).clamp(min=1) - 1
        batch_index = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        last_hidden = hidden_states[batch_index, lengths, :]
        return F.normalize(self.text_projection(last_hidden), dim=-1)

    def _gather_with_grad(self, tensor: torch.Tensor) -> torch.Tensor:
        if (
            not dist.is_available()
            or not dist.is_initialized()
            or int(dist.get_world_size()) <= 1
            or dist_all_gather_with_grad is None
        ):
            return tensor
        gathered = dist_all_gather_with_grad(tensor)
        return torch.cat(list(gathered), dim=0)

    def _gather_no_grad(self, tensor: torch.Tensor) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized() or int(dist.get_world_size()) <= 1:
            return tensor
        gathered = [torch.empty_like(tensor) for _ in range(int(dist.get_world_size()))]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    def grouped_clip_loss(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_image_embeddings = self._gather_with_grad(image_embeddings)
        all_text_embeddings = self._gather_with_grad(text_embeddings)
        all_group_ids = self._gather_no_grad(group_ids.to(dtype=torch.long))
        logits = self.logit_scale.exp().clamp(max=100.0) * all_image_embeddings @ all_text_embeddings.t()
        positive_mask = all_group_ids.view(-1, 1).eq(all_group_ids.view(1, -1))
        positive_float = positive_mask.to(dtype=logits.dtype)

        image_targets = positive_float / positive_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        text_targets = positive_float.t() / positive_float.t().sum(dim=1, keepdim=True).clamp_min(1.0)
        loss_i2t = -(image_targets * logits.log_softmax(dim=1)).sum(dim=1).mean()
        loss_t2i = -(text_targets * logits.t().log_softmax(dim=1)).sum(dim=1).mean()
        return 0.5 * (loss_i2t + loss_t2i), logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        group_ids: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        del kwargs
        image_embeddings, aligned_tokens = self.encode_image(pixel_values)
        text_embeddings = self.encode_text(input_ids, attention_mask)
        loss, logits = self.grouped_clip_loss(image_embeddings, text_embeddings, group_ids)
        return {
            "loss": loss,
            "logits": logits,
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
            "aligned_visual_tokens": aligned_tokens,
        }


def save_qwen3_rc_dinov2_clip_align_modules(
    model: nn.Module,
    output_dir: str | Path,
    tokenizer: Any | None = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_for_save = unwrap_model(model)
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_path))
    torch.save(
        {
            "vision_encoder": model_for_save.vision_encoder.state_dict(),
            "visual_norm": model_for_save.visual_norm.state_dict(),
            "visual_projector": model_for_save.visual_projector.state_dict(),
            "geometric_position_mlp": model_for_save.geometric_position_mlp.state_dict(),
            "token_alignment": model_for_save.token_alignment.state_dict(),
            "readout_norm": model_for_save.readout_norm.state_dict(),
            "image_projection": model_for_save.image_projection.state_dict(),
            "text_projection": model_for_save.text_projection.state_dict(),
            "logit_scale": model_for_save.logit_scale.detach().cpu(),
            "encoder_input_pad_size": int(model_for_save.encoder_input_pad_size),
            "visual_grid_size": int(model_for_save.visual_grid_size),
            "num_visual_tokens": int(model_for_save.num_visual_tokens),
            "contrastive_dim": int(model_for_save.contrastive_dim),
        },
        str(output_path / "rc_dinov2_clip_align_modules.pt"),
    )
