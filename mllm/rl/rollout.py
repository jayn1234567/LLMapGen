from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Any

import torch


def _patch_vllm_transformers_aimv2_registration() -> None:
    """Allow vLLM 0.9.x to import with Transformers versions that already ship AIMv2.

    vLLM 0.9.x registers its own AIMv2 config at import time. Transformers 4.56
    also contains an AIMv2 entry, so the duplicate registration raises before
    vLLM can even start. We only relax this one known duplicate and leave all
    other config registrations unchanged.
    """

    try:
        from transformers.models.auto import configuration_auto
    except Exception:
        return

    mapping = configuration_auto.CONFIG_MAPPING
    if getattr(mapping, "_mllm_vllm_aimv2_patch", False):
        return

    original_register = mapping.register

    def register(model_type, config, exist_ok=False):
        if model_type == "aimv2":
            exist_ok = True
        return original_register(model_type, config, exist_ok=exist_ok)

    mapping.register = register
    setattr(mapping, "_mllm_vllm_aimv2_patch", True)


def _install_vllm_subprocess_sitecustomize_patch() -> None:
    """Propagate the AIMv2 registration patch into vLLM inspection subprocesses."""

    patch_dir = Path(os.environ.get("MLLM_VLLM_PATCH_DIR", "/tmp/mllm_vllm_sitecustomize"))
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_file = patch_dir / "sitecustomize.py"
    patch_file.write_text(
        """
try:
    from transformers.models.auto import configuration_auto
    _mapping = configuration_auto.CONFIG_MAPPING
    if not getattr(_mapping, "_mllm_vllm_aimv2_patch", False):
        _original_register = _mapping.register
        def _mllm_register(model_type, config, exist_ok=False):
            if model_type == "aimv2":
                exist_ok = True
            return _original_register(model_type, config, exist_ok=exist_ok)
        _mapping.register = _mllm_register
        setattr(_mapping, "_mllm_vllm_aimv2_patch", True)
except Exception:
    pass
""".lstrip(),
        encoding="utf-8",
    )
    pythonpath = os.environ.get("PYTHONPATH", "")
    parts = [part for part in pythonpath.split(os.pathsep) if part]
    patch_dir_text = str(patch_dir)
    if patch_dir_text not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([patch_dir_text, *parts])


@dataclass
class RolloutPrompt:
    group_index: int
    sample_id: str
    prompt_ids: list[int]
    prompt_embeds: torch.Tensor
    ground_truth: str
    coord_config: dict[str, Any]
    image_index: int


@dataclass
class RolloutCompletion:
    group_index: int
    sample_id: str
    completion_index: int
    token_ids: list[int]
    text: str
    finish_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutBatch:
    rollout_id: str
    prompts: list[RolloutPrompt]
    lora_adapter_path: str | None = None
    lora_int_id: int | None = None


@dataclass
class RolloutSample:
    group_index: int
    sample_id: str
    prompt_ids: torch.Tensor
    completion_ids: torch.Tensor
    text: str
    ground_truth: str
    coord_config: dict[str, Any]
    image_index: int


def normalize_completion_text(text: str) -> str:
    cleaned = str(text or "").strip()
    for token in ("<|im_end|>", "<|endoftext|>", "</s>"):
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned.strip()


class VLLMPromptEmbedRolloutWorker:
    """vLLM rollout worker using externally computed multimodal prompt embeds.

    The actor still owns the project-specific DINO/projector forward pass. This
    worker only runs the Qwen text decoder through vLLM. That keeps rollout on
    the formal vLLM path without requiring a full custom vLLM DINO model plugin.
    """

    def __init__(
        self,
        model_path: str,
        num_generations: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.70,
        dtype: str = "auto",
        max_model_len: int | None = None,
        enable_lora: bool = True,
        max_lora_rank: int = 8,
        enforce_eager: bool = False,
        trust_remote_code: bool = True,
    ):
        if num_generations < 2:
            raise ValueError("GRPO requires num_generations >= 2.")
        if temperature <= 0:
            raise ValueError("vLLM GRPO rollout requires temperature > 0 to produce grouped samples.")

        _patch_vllm_transformers_aimv2_registration()
        _install_vllm_subprocess_sitecustomize_patch()
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise ImportError(
                "vLLM is required for formal GRPO rollout. Install the project RL extra "
                "or install vllm in the active environment."
            ) from exc

        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "enable_lora": enable_lora,
            "max_lora_rank": max_lora_rank,
            "enforce_eager": enforce_eager,
        }
        if max_model_len:
            llm_kwargs["max_model_len"] = max_model_len
        # Prompt embedding support is intentionally enabled explicitly. vLLM
        # versions that do not support it should fail here rather than silently
        # falling back to text-only prompts.
        llm_kwargs["enable_prompt_embeds"] = True

        self.llm = LLM(**llm_kwargs)
        self.SamplingParams = SamplingParams
        self.num_generations = num_generations
        self.sampling_params = SamplingParams(
            n=num_generations,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            skip_special_tokens=True,
        )

    def _lora_request(self, adapter_path: str | None, lora_int_id: int | None):
        if not adapter_path:
            return None
        try:
            from vllm.lora.request import LoRARequest
        except ImportError as exc:  # pragma: no cover
            raise ImportError("The installed vLLM does not expose LoRARequest.") from exc
        return LoRARequest(
            lora_name=f"actor_step_{lora_int_id or int(time.time())}",
            lora_int_id=int(lora_int_id or 1),
            lora_path=adapter_path,
        )

    def generate(self, batch: RolloutBatch) -> list[RolloutCompletion]:
        requests = [
            {"prompt_embeds": prompt.prompt_embeds.detach().cpu()}
            for prompt in batch.prompts
        ]
        outputs = self.llm.generate(
            requests,
            sampling_params=self.sampling_params,
            lora_request=self._lora_request(batch.lora_adapter_path, batch.lora_int_id),
        )
        completions: list[RolloutCompletion] = []
        for prompt, request_output in zip(batch.prompts, outputs):
            for completion_index, output in enumerate(request_output.outputs):
                token_ids = list(getattr(output, "token_ids", []) or [])
                text = normalize_completion_text(getattr(output, "text", ""))
                completions.append(
                    RolloutCompletion(
                        group_index=prompt.group_index,
                        sample_id=prompt.sample_id,
                        completion_index=completion_index,
                        token_ids=[int(token) for token in token_ids],
                        text=text,
                        finish_reason=getattr(output, "finish_reason", None),
                    )
                )
        return completions
