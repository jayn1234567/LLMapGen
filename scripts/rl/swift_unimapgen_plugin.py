"""ms-swift adapter for the UniMapGen DINOv2 + CapRL text model.

This file is loaded by ms-swift through ``--external_plugins``.  It keeps the
Swift integration at the boundary of the project: model construction and
image preprocessing are delegated to the existing UniMapGen builder, while
the reward remains the dependency-light map reward in ``mllm.reward``.

The plugin deliberately does not register a native Qwen-VL vision tower.  The
model used by the main route is a text-only CapRL-Qwen3-derived decoder with
the project's DINOv2 tower and ``mlp2x_gelu`` projector attached.
"""

from __future__ import annotations

import os
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig

from mllm.constants import DEFAULT_IMAGE_PATCH_TOKEN, IMAGE_TOKEN_INDEX
from mllm.mm_utils import process_images
from mllm.model.builder import load_pretrained_model
from mllm.model.language_model.qwen_family import as_qwen_multimodal_config
from mllm.reward.swift_map_reward import SwiftMapRewardConfig, compute_swift_map_rewards

from swift.model import ModelLoader, ModelMeta, MultiModelKeys, register_model, register_model_arch
from swift.rewards import ORM, orms
from swift.template import StdTemplateInputs, Template, TemplateMeta, register_template
from swift.utils import Processor


MODEL_TYPE = "unimapgen_qwen3_dinov2"
ARCH_TYPE = "unimapgen_qwen3_dinov2"
TEMPLATE_TYPE = "unimapgen_qwen3_dinov2"

PROJECT_SYSTEM_PROMPT = (
    "You are a road-map reconstruction assistant designed to process BEV "
    "(Bird's Eye View) images generated from LiDAR data.\n"
    "Predict the complete road map from the current patch in the BEV image.\n"
    "Return only valid JSON in the required schema.\n"
    "Do not output markdown fences or extra explanation.\n"
    "Keep all coordinates in the patch-local coordinate system."
)


def _env_path(name: str, *, required: bool = False) -> Optional[str]:
    value = os.environ.get(name, "").strip()
    if value:
        path = str(Path(value).expanduser())
        if required and not Path(path).exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
        return path
    if required:
        raise RuntimeError(f"{name} must be set for the UniMapGen Swift plugin")
    return None


def _local_device(model_kwargs: Dict[str, Any]) -> str:
    """Resolve the per-process device without assuming CUDA is available."""
    explicit = os.environ.get("UNIMAPGEN_LOAD_DEVICE", "").strip()
    if explicit:
        return explicit

    device_map = model_kwargs.get("device_map")
    if isinstance(device_map, dict):
        mapped = device_map.get("")
        if mapped is not None:
            return str(mapped)
    if isinstance(device_map, str) and device_map not in {"auto", "balanced", "balanced_low_0"}:
        return device_map

    npu = getattr(torch, "npu", None)
    if npu is not None and bool(getattr(npu, "is_available", lambda: False)()):
        return f"npu:{int(os.environ.get('LOCAL_RANK', '0'))}"
    if torch.cuda.is_available():
        return f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
    return "cpu"


def _torch_dtype(value: Any) -> Optional[torch.dtype]:
    if isinstance(value, torch.dtype):
        return value
    text = str(value or "").strip().lower()
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(text)


def _model_has_weights(path: str) -> bool:
    root = Path(path)
    if (root / "adapter_config.json").is_file():
        return any((root / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))
    return any(
        (root / name).is_file()
        for name in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
    )


def _is_lora_checkpoint(path: str) -> bool:
    root = Path(path)
    return (root / "adapter_config.json").is_file() and any(
        (root / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def _is_full_checkpoint(path: str) -> bool:
    root = Path(path)
    return (root / "config.json").is_file() and _model_has_weights(path) and not _is_lora_checkpoint(path)


def _resolve_load_spec(model_dir: str) -> tuple[str, Optional[str], str]:
    """Resolve the model/base/start-checkpoint relationship used by the builder.

    Swift's ``--model`` remains the base model so that its config and tokenizer
    discovery work normally.  ``UNIMAPGEN_START_CHECKPOINT`` is an explicit
    optional SFT result layered on top of that base before Swift adds its GRPO
    adapter.  Ambiguous checkpoint directories fail closed instead of silently
    training from the base model.
    """
    model_dir = str(Path(model_dir).resolve())
    base_dir = _env_path("UNIMAPGEN_MODEL_BASE", required=False) or model_dir
    start_dir = _env_path("UNIMAPGEN_START_CHECKPOINT", required=False)
    if start_dir:
        start_dir = str(Path(start_dir).resolve())
        if not Path(start_dir).is_dir():
            raise FileNotFoundError(f"UNIMAPGEN_START_CHECKPOINT is not a directory: {start_dir}")
        if _is_lora_checkpoint(start_dir):
            if Path(base_dir).resolve() == Path(start_dir).resolve():
                raise RuntimeError("UNIMAPGEN_START_CHECKPOINT must differ from UNIMAPGEN_MODEL_BASE")
            # The project builder's native LoRA path needs both files.  The
            # manual fallback below handles older exports without config.json.
            if not (Path(start_dir) / "non_lora_trainables.bin").is_file():
                raise RuntimeError(
                    "UniMapGen LoRA start checkpoint is missing non_lora_trainables.bin: "
                    f"{start_dir}"
                )
            return start_dir, base_dir, "unimapgen_mllm_lora"
        if _is_full_checkpoint(start_dir):
            return start_dir, None, "unimapgen_mllm"
        raise RuntimeError(
            "UNIMAPGEN_START_CHECKPOINT is neither a supported full checkpoint "
            f"nor a LoRA adapter directory: {start_dir}"
        )

    if _is_lora_checkpoint(model_dir):
        if Path(base_dir).resolve() == Path(model_dir).resolve():
            raise RuntimeError(
                "A UniMapGen LoRA checkpoint requires UNIMAPGEN_MODEL_BASE "
                "to point to the CapRL-Qwen3-derived base model."
            )
        return model_dir, base_dir, "unimapgen_mllm_lora"
    if _is_full_checkpoint(model_dir) and Path(model_dir).resolve() != Path(base_dir).resolve():
        return model_dir, None, "unimapgen_mllm"
    return base_dir, None, "unimapgen_mllm"


def _load_legacy_lora_without_config(
    checkpoint_dir: str,
    base_dir: str,
    *,
    device: str,
    builder_kwargs: Dict[str, Any],
) -> Any:
    """Load older project LoRA exports that omitted the multimodal config.

    Some historical SFT outputs contain ``adapter_config.json`` and
    ``non_lora_trainables.bin`` but no ``config.json``.  PEFT can still attach
    the adapter to a fully initialized project model; this path preserves the
    trained projector/vision tensors and then merges the old adapter before
    Swift adds a fresh GRPO adapter.
    """
    non_lora_path = Path(checkpoint_dir) / "non_lora_trainables.bin"
    if not non_lora_path.is_file():
        raise RuntimeError(
            "Legacy LoRA checkpoint has no non_lora_trainables.bin, so its "
            "trained vision/projector weights cannot be reconstructed: "
            f"{checkpoint_dir}"
        )
    _, model, _, _ = load_pretrained_model(
        base_dir,
        None,
        "unimapgen_mllm",
        device_map="auto",
        device=device,
        model_config_overrides=_vision_overrides(),
        tokenizer_use_fast=False,
        **builder_kwargs,
    )
    state = torch.load(non_lora_path, map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError(f"Invalid non_lora_trainables.bin: {non_lora_path}")
    state = {(key[11:] if key.startswith("base_model.") else key): value for key, value in state.items()}
    if any(key.startswith("model.model.") for key in state):
        state = {(key[6:] if key.startswith("model.") else key): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        "[unimapgen-swift] legacy LoRA non-LoRA weights loaded "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model = model.merge_and_unload()
    return model


def _vision_overrides() -> Dict[str, Any]:
    tower = _env_path("UNIMAPGEN_VISION_TOWER", required=True)
    input_size = int(os.environ.get("UNIMAPGEN_INPUT_IMAGE_SIZE", "518"))
    select_layer = int(os.environ.get("UNIMAPGEN_VISION_SELECT_LAYER", "-2"))
    return {
        "mm_vision_tower": tower,
        "vision_tower": tower,
        "mm_vision_tower_type": "dinov2",
        "mm_hidden_size": int(os.environ.get("UNIMAPGEN_VISION_HIDDEN_SIZE", "1024")),
        "mm_projector_type": os.environ.get("UNIMAPGEN_PROJECTOR_TYPE", "mlp2x_gelu"),
        "mm_vision_select_layer": select_layer,
        "mm_vision_select_feature": os.environ.get("UNIMAPGEN_VISION_SELECT_FEATURE", "patch"),
        "mm_patch_merge_type": "flat",
        "input_image_size": input_size,
        "disable_deepstack": True,
        "deepstack_visual_indexes": None,
        "mm_use_im_patch_token": True,
        "mm_use_im_start_end": False,
        "use_mm_proj": True,
        "image_aspect_ratio": "square",
    }


def _adapt_qwen_config(config: PretrainedConfig) -> PretrainedConfig:
    family = os.environ.get("UNIMAPGEN_QWEN_FAMILY", "qwen3").strip() or "qwen3"
    adapted = as_qwen_multimodal_config(config, family=family)
    for key, value in _vision_overrides().items():
        setattr(adapted, key, value)
    return adapted


def _tokenizer_source(model_dir: str) -> str:
    configured = _env_path("UNIMAPGEN_TOKENIZER", required=False)
    if configured:
        return configured
    base = _env_path("UNIMAPGEN_MODEL_BASE", required=False)
    if base:
        return base
    return model_dir


class UniMapGenLoader(ModelLoader):
    """Load a project multimodal model through the existing builder."""

    def get_config(self, model_dir: str) -> PretrainedConfig:
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        return _adapt_qwen_config(config)

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        source = _tokenizer_source(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=False)
        if DEFAULT_IMAGE_PATCH_TOKEN not in tokenizer.get_vocab():
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        # These attributes are consumed by the custom Template and mirror the
        # project tokenizer setup in LlavaMetaForCausalLM.
        tokenizer.image_token = "<image>"
        tokenizer.image_token_id = IMAGE_TOKEN_INDEX
        tokenizer.model_max_length = int(
            os.environ.get("UNIMAPGEN_MODEL_MAX_LENGTH", str(getattr(tokenizer, "model_max_length", 4096)))
        )
        return tokenizer

    def get_model(
        self,
        model_dir: str,
        config: PretrainedConfig,
        processor: Processor,
        model_kwargs: Dict[str, Any],
    ):
        del processor  # The project builder creates and synchronizes its own tokenizer.
        model_dir = str(Path(model_dir).resolve())
        load_path, load_base, model_name = _resolve_load_spec(model_dir)

        device = _local_device(model_kwargs)
        dtype = _torch_dtype(self.torch_dtype)
        builder_kwargs: Dict[str, Any] = {}
        if dtype is not None:
            builder_kwargs["torch_dtype"] = dtype

        print(
            "[unimapgen-swift] loading "
            f"checkpoint={load_path} base={load_base or '<same>'} "
            f"vision={_env_path('UNIMAPGEN_VISION_TOWER', required=True)} "
            f"device={device} dtype={dtype or 'builder-default'}",
            flush=True,
        )
        if model_name == "unimapgen_mllm_lora" and not (Path(load_path) / "config.json").is_file():
            print(
                "[unimapgen-swift] using legacy LoRA loader because config.json is absent",
                flush=True,
            )
            model = _load_legacy_lora_without_config(
                load_path,
                str(load_base),
                device=device,
                builder_kwargs=builder_kwargs,
            )
        else:
            _, model, _, _ = load_pretrained_model(
                load_path,
                load_base,
                model_name,
                device_map="auto",
                device=device,
                model_config_overrides=_vision_overrides(),
                tokenizer_use_fast=False,
                **builder_kwargs,
            )
        _patch_project_call_signatures(model)
        return model


def _patch_project_call_signatures(model: Any) -> None:
    """Bridge Swift's standard ``input_ids=`` call to the project API.

    The project model intentionally names its generation input ``inputs`` for
    compatibility with the original inference code.  Transformers training
    uses ``input_ids``.  Keeping this adapter on the plugin avoids changing the
    stable project model API.
    """
    if getattr(model, "_swift_unimapgen_patched", False):
        return

    original_generate = model.generate

    @wraps(original_generate)
    def generate_adapter(
        input_ids=None,
        inputs=None,
        images=None,
        pixel_values=None,
        image_sizes=None,
        **kwargs,
    ):
        if inputs is None:
            inputs = input_ids
        if images is None:
            images = pixel_values
        return original_generate(
            inputs=inputs,
            images=images,
            image_sizes=image_sizes,
            **kwargs,
        )

    model.generate = generate_adapter

    original_forward = model.forward

    @wraps(original_forward)
    def forward_adapter(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        image_sizes=None,
        return_dict=None,
        cache_position=None,
        visual_pos_mask=None,
        deepstack_visual_embeds=None,
        inputs=None,
        pixel_values=None,
        num_items_in_batch=None,
        loss_scale=None,
        **kwargs,
    ):
        del num_items_in_batch, loss_scale, kwargs
        if input_ids is None:
            input_ids = inputs
        if images is None:
            images = pixel_values
        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            image_sizes=image_sizes,
            return_dict=return_dict,
            cache_position=cache_position,
            visual_pos_mask=visual_pos_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

    model.forward = forward_adapter
    model._swift_unimapgen_patched = True


class UniMapGenTemplate(Template):
    """ChatML template carrying project-style negative image placeholders."""

    placeholder_tokens = [IMAGE_TOKEN_INDEX]
    skip_prompt = False
    use_model = True
    load_images = True
    support_padding_free = False

    def replace_tag(
        self,
        media_type: Literal["image", "video", "audio"],
        index: int,
        inputs: StdTemplateInputs,
    ) -> List[Any]:
        del index, inputs
        if media_type != "image":
            raise ValueError(f"UniMapGen only supports image inputs, got {media_type!r}")
        return [[IMAGE_TOKEN_INDEX], "\n"]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = list(inputs.images or [])
        if not images:
            return encoded

        model = self.model
        tower = model.get_vision_tower()
        processor = getattr(tower, "image_processor", None)
        if processor is None:
            raise RuntimeError("UniMapGen DINOv2 tower has no image_processor")
        image_tensor = process_images(images, processor, model.config)
        if not torch.is_tensor(image_tensor):
            image_tensor = torch.stack(list(image_tensor), dim=0)
        encoded["images"] = [image for image in image_tensor]
        encoded["image_sizes"] = [tuple(image.size) for image in images]
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = super()._data_collator_mm_data(batch)
        image_rows = [row.get("images") for row in batch if row.get("images") is not None]
        if image_rows:
            # Each Dataset V2 row has one image.  Flattening keeps the project
            # model's expected batch-major list of [C,H,W] tensors.
            result["images"] = [image for row in image_rows for image in row]
        size_rows = [row.get("image_sizes") for row in batch if row.get("image_sizes") is not None]
        if size_rows:
            result["image_sizes"] = [size for row in size_rows for size in row]
        return result


class UniMapGenMapORM(ORM):
    """Swift ORM wrapper around the project's line-F1 map reward."""

    def __init__(self, args=None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.reward_config = SwiftMapRewardConfig(
            map_task=os.environ.get("UNIMAPGEN_MAP_TASK", "lane_intersection"),
            default_patch_size=int(os.environ.get("UNIMAPGEN_PATCH_SIZE", "256")),
            default_coord_mode=os.environ.get("UNIMAPGEN_COORD_MODE", "norm1000"),
            default_coord_range=int(os.environ.get("UNIMAPGEN_COORD_RANGE", "1000")),
            meter_per_pixel=float(os.environ.get("UNIMAPGEN_METER_PER_PIXEL", "0.2")),
            buffer_size=float(os.environ.get("UNIMAPGEN_BUFFER_SIZE", "1.0")),
            match_threshold=float(os.environ.get("UNIMAPGEN_MATCH_THRESHOLD", "0.33")),
            coordinate_sigma_m=float(os.environ.get("UNIMAPGEN_COORDINATE_SIGMA_M", "0.75")),
        )

    def __call__(
        self,
        completions,
        solution=None,
        ground_truth=None,
        coord_config=None,
        map_task=None,
        **kwargs,
    ) -> List[float]:
        del kwargs
        target = solution if solution is not None else ground_truth
        return compute_swift_map_rewards(
            list(completions or []),
            solution=target,
            coord_config=coord_config,
            map_task=map_task,
            config=self.reward_config,
        )


register_model_arch(
    MultiModelKeys(
        ARCH_TYPE,
        language_model=["model.layers", "lm_head"],
        aligner=["model.mm_projector"],
        vision_tower=["model.vision_tower"],
    ),
    exist_ok=True,
)

register_template(
    TemplateMeta(
        template_type=TEMPLATE_TYPE,
        prefix=[],
        prompt=["<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n"],
        chat_sep=["<|im_end|>\n"],
        suffix=["<|im_end|>\n"],
        system_prefix=["<|im_start|>system\n{{SYSTEM}}<|im_end|>\n"],
        default_system=PROJECT_SYSTEM_PROMPT,
        template_cls=UniMapGenTemplate,
        auto_add_bos=False,
        stop_words=["<|im_end|>"],
    ),
    exist_ok=True,
)

register_model(
    ModelMeta(
        model_type=MODEL_TYPE,
        model_groups=[],
        loader=UniMapGenLoader,
        template=TEMPLATE_TYPE,
        model_arch=ARCH_TYPE,
        architectures=["Qwen3MultimodalForCausalLM"],
        is_multimodal=True,
    ),
    exist_ok=True,
)

orms["unimapgen_map"] = UniMapGenMapORM


__all__ = [
    "UniMapGenLoader",
    "UniMapGenTemplate",
    "UniMapGenMapORM",
    "compute_swift_map_rewards",
]
