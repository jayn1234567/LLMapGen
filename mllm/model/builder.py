#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import json
import warnings
import shutil
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from mllm.model import *
from mllm.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mllm.model.qwen3vl_extractor import is_qwen3vl_checkpoint, is_llava_checkpoint, ensure_extracted_llm_from_qwen3vl
from mllm.model.qwen_token_utils import qwen_tokenizer_kwargs, sync_qwen_token_config

try:
    from safetensors.torch import load_file as safe_load_file
except ImportError:  # pragma: no cover
    safe_load_file = None


def _load_checkpoint_file(path):
    if path.endswith(".safetensors"):
        if safe_load_file is None:
            raise ImportError("safetensors is required to load safetensors checkpoints")
        return safe_load_file(path, device="cpu")
    return torch.load(path, map_location="cpu")


def _iter_checkpoint_state_files(model_path, prefixes):
    safe_index = os.path.join(model_path, "model.safetensors.index.json")
    bin_index = os.path.join(model_path, "pytorch_model.bin.index.json")
    if os.path.isfile(safe_index) or os.path.isfile(bin_index):
        index_path = safe_index if os.path.isfile(safe_index) else bin_index
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        shard_names = sorted({
            shard for key, shard in weight_map.items()
            if key.startswith(prefixes)
        })
        for shard_name in shard_names:
            yield os.path.join(model_path, shard_name)
        return

    for filename in ("model.safetensors", "pytorch_model.bin"):
        path = os.path.join(model_path, filename)
        if os.path.isfile(path):
            yield path
            return
    for pattern in ("model-*-of-*.safetensors", "pytorch_model-*-of-*.bin"):
        for path in sorted(Path(model_path).glob(pattern)):
            yield str(path)


def _load_multimodal_weights_if_present(model, model_path):
    if not os.path.isdir(model_path):
        return 0

    prefixes = ("model.vision_tower.",)
    expected = model.state_dict()
    expected_runtime = dict(model.named_parameters())
    expected_runtime.update(dict(model.named_buffers()))
    has_deepspeed_partitioned_params = any(
        hasattr(param, "ds_id") or hasattr(param, "ds_shape")
        for param in expected_runtime.values()
    )
    if has_deepspeed_partitioned_params:
        print(
            "[WARN] Skipping post-init model.vision_tower.* checkpoint reload while "
            "parameters are DeepSpeed ZeRO-3 partitioned. The vision tower remains "
            "initialized from the requested vision_tower path."
        )
        return 0

    def expected_shape(key):
        tensor = expected_runtime.get(key)
        if tensor is not None:
            ds_shape = getattr(tensor, "ds_shape", None)
            if ds_shape is not None:
                return torch.Size(ds_shape)
            return tensor.shape
        tensor = expected.get(key)
        if tensor is not None:
            return tensor.shape
        return None

    loaded = {}
    skipped = []
    seen = 0
    for state_file in _iter_checkpoint_state_files(model_path, prefixes):
        state_dict = _load_checkpoint_file(state_file)
        for key, value in state_dict.items():
            if not key.startswith(prefixes):
                continue
            seen += 1
            shape = expected_shape(key)
            if shape is not None and shape == value.shape:
                loaded[key] = value
            else:
                skipped.append(key)
        del state_dict

    if not loaded:
        if seen:
            if has_deepspeed_partitioned_params:
                print(
                    f"[WARN] Found {seen} multimodal checkpoint tensors under {model_path}, "
                    "but none could be shape-matched while parameters are DeepSpeed-partitioned. "
                    "Continuing with the separately initialized vision tower."
                )
                return 0
            raise RuntimeError(
                f"Found {seen} multimodal checkpoint tensors under {model_path}, "
                "but none matched the initialized model. Check vision tower type, "
                "input_image_size, deepstack_visual_indexes, and LLM hidden size."
            )
        print(f"[WARN] No model.vision_tower.* tensors found in checkpoint: {model_path}")
        return 0

    missing, unexpected = model.load_state_dict(loaded, strict=False)
    print(
        f"Loaded {len(loaded)}/{seen} multimodal checkpoint tensors "
        "after vision tower init."
    )
    if unexpected:
        print(f"[WARN] Unexpected multimodal keys after reload: {unexpected[:20]}")
    if skipped:
        print(f"[WARN] Skipped multimodal keys with missing/mismatched shapes: {skipped[:20]}")
    return len(loaded)


def _apply_model_config_overrides(config, overrides=None):
    if not overrides:
        return config
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)
    return config


def _vision_tower_runtime_dtype(vision_tower, device):
    class_name = vision_tower.__class__.__name__.lower()
    tower_name = str(getattr(vision_tower, "vision_tower_name", "")).lower()
    if "dinov3" in class_name or "dinov3" in tower_name:
        if str(device).startswith(("cuda", "npu")):
            return torch.bfloat16
        return torch.float32
    return torch.float16


def _checkpoint_has_multimodal_config(model_path):
    config_path = os.path.join(str(model_path), "config.json")
    metadata_path = os.path.join(str(model_path), "qwen_multimodal_checkpoint.json")
    if os.path.exists(metadata_path):
        return True
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return False
    return any(cfg.get(key) is not None for key in ("mm_vision_tower", "vision_tower", "mm_projector_type"))


def _coerce_optional_bool(value, name):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean value for {name}: {value!r}")
    return bool(value)


def _resolve_tokenizer_use_fast(default_value, override_value):
    override_value = _coerce_optional_bool(override_value, "tokenizer_use_fast")
    if override_value is None:
        return bool(default_value)
    return override_value


def _has_tokenizer_files(path):
    if not path or not os.path.isdir(str(path)):
        return False
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    )
    return any(os.path.isfile(os.path.join(str(path), name)) for name in tokenizer_files)


def _has_slow_tokenizer_files(path):
    if not path or not os.path.isdir(str(path)):
        return False
    slow_tokenizer_files = ("tokenizer.model", "vocab.json", "merges.txt")
    return any(os.path.isfile(os.path.join(str(path), name)) for name in slow_tokenizer_files)


def _tokenizer_config_has_list_extra_special_tokens(path):
    config_path = os.path.join(str(path), "tokenizer_config.json")
    if not os.path.isfile(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return False
    return isinstance(cfg.get("extra_special_tokens"), list)


def _resolve_tokenizer_source(model_path):
    if _has_tokenizer_files(model_path):
        return model_path
    parent = os.path.dirname(os.path.abspath(str(model_path)))
    if parent and parent != os.path.abspath(str(model_path)) and _has_tokenizer_files(parent):
        warnings.warn(
            f"Tokenizer files were not found in {model_path}; loading tokenizer from parent directory {parent}."
        )
        return parent
    return model_path


def _rank0_print(message):
    if str(os.environ.get("RANK", "0")) in {"0", "-1"}:
        print(message)


def _load_tokenizer_with_fast_fallback(model_path, use_fast, **kwargs):
    use_fast = _resolve_tokenizer_use_fast(False, use_fast)
    tokenizer_source = _resolve_tokenizer_source(model_path)
    tokenizer_kwargs = dict(kwargs)
    if _tokenizer_config_has_list_extra_special_tokens(tokenizer_source):
        tokenizer_kwargs.setdefault("extra_special_tokens", {})
    if not use_fast and not _has_slow_tokenizer_files(tokenizer_source):
        warnings.warn(
            f"Slow tokenizer was requested for {tokenizer_source}, but no slow tokenizer "
            "vocab files were found. Retrying with use_fast=True."
        )
        use_fast = True
    _rank0_print(
        "[mllm] Loading tokenizer: "
        f"source={tokenizer_source}, requested_use_fast={use_fast}, "
        f"trust_remote_code={tokenizer_kwargs.get('trust_remote_code', False)}"
    )
    try:
        return AutoTokenizer.from_pretrained(tokenizer_source, use_fast=use_fast, **tokenizer_kwargs)
    except (ImportError, ValueError) as exc:
        message = str(exc)
        fast_backend_error = (
            "Couldn't instantiate the backend tokenizer" in message
            or "sentencepiece or tiktoken" in message
        )
        if use_fast and fast_backend_error:
            warnings.warn(
                f"Fast tokenizer failed for {tokenizer_source}; retrying with use_fast=False. "
                f"Original error: {exc}"
            )
            return AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False, **tokenizer_kwargs)
        raise


def _load_auto_config(model_path, **kwargs):
    kwargs.setdefault("trust_remote_code", True)
    try:
        return AutoConfig.from_pretrained(model_path, **kwargs)
    except ValueError as exc:
        message = str(exc)
        if "Should have a `model_type` key in its config.json" not in message:
            raise
        config_path = os.path.join(str(model_path), "config.json")
        if not os.path.isfile(config_path):
            raise
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        metadata_path = os.path.join(str(model_path), "qwen_multimodal_checkpoint.json")
        metadata = {}
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {}

        raw_model_type = (
            config_dict.get("model_type")
            or metadata.get("model_type")
            or ("qwen3" if "qwen3" in str(model_path).lower() else None)
            or ("qwen2" if "qwen2" in str(model_path).lower() else None)
        )
        if raw_model_type is None:
            architectures = config_dict.get("architectures") or metadata.get("architectures") or []
            joined_architectures = " ".join(str(item).lower() for item in architectures)
            if "qwen3" in joined_architectures:
                raw_model_type = "qwen3"
            elif "qwen2" in joined_architectures:
                raw_model_type = "qwen2"

        model_type_text = str(raw_model_type or "").lower()
        if "qwen3" in model_type_text:
            model_type = "qwen3"
        elif "qwen2" in model_type_text:
            model_type = "qwen2"
        else:
            model_type = None

        if model_type is None:
            raise ValueError(
                f"{model_path} config.json is missing model_type and qwen_multimodal_checkpoint.json "
                "does not provide one. Please use a valid checkpoint-* directory or repair config.json."
            ) from exc

        config_dict["model_type"] = model_type
        if model_type == "qwen3":
            config_dict.setdefault("architectures", ["Qwen3MultimodalForCausalLM"])
            config_class = Qwen3MultimodalConfig
        elif model_type == "qwen2":
            config_dict.setdefault("architectures", ["Qwen2MultimodalForCausalLM"])
            config_class = Qwen2MultimodalConfig
        else:
            raise ValueError(f"Unsupported inferred model_type for {model_path}: {model_type}") from exc

        for key in (
            "mm_vision_tower",
            "vision_tower",
            "mm_vision_tower_type",
            "input_image_size",
            "deepstack_visual_indexes",
            "disable_deepstack",
            "multi_vision_towers",
            "multi_vision_tower_types",
            "multi_vision_input_image_sizes",
            "multi_vision_primary_index",
            "multi_vision_hidden_size",
            "multi_vision_target_grid",
            "multi_vision_fusion",
            "multi_vision_router_temperature",
            "multi_vision_router_hidden_ratio",
            "multi_vision_router_use_diff",
            "multi_vision_dropout",
        ):
            if config_dict.get(key) is None and metadata.get(key) is not None:
                config_dict[key] = metadata[key]

        warnings.warn(
            f"{model_path}/config.json is missing model_type; inferred model_type={model_type} "
            "for multimodal checkpoint loading."
        )
        return config_class.from_dict(config_dict)


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, model_config_overrides=None, tokenizer_use_fast=None, **kwargs):
    kwargs = ({} if device_map is None else {"device_map": device_map}) | kwargs
    is_mllm_model = (
        any(key in model_name.lower() for key in ("mllm", "llava"))
        or _checkpoint_has_multimodal_config(model_path)
    )

    if is_qwen3vl_checkpoint(model_path) and not is_llava_checkpoint(model_path):
        print(f"Ensuring extracted LLM cache for Qwen3-VL checkpoint: {model_path}")
        cache_path = ensure_extracted_llm_from_qwen3vl(model_path)
        print(f"Using extracted LLM cache: {cache_path}")
        model_path = cache_path

    if device_map is None:
        resolved_device_map = None
    elif str(device).startswith("npu"):
        kwargs['device_map'] = {"": device}
        resolved_device_map = kwargs['device_map']
    elif str(device).startswith("cuda"):
        kwargs['device_map'] = {"": device}  # explicit mapping works for both cuda and cuda:N
        resolved_device_map = kwargs['device_map']
    else:
        kwargs['device_map'] = {"": device}
        resolved_device_map = kwargs['device_map']

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs.setdefault('torch_dtype', torch.float16)

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if is_mllm_model:
        # Load generic multimodal model. The "llava" branch name is kept for legacy checkpoints.
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from mllm.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path, trust_remote_code=True)
            _apply_model_config_overrides(lora_cfg_pretrained, model_config_overrides)
            tokenizer = _load_tokenizer_with_fast_fallback(
                model_base,
                use_fast=False,
                **qwen_tokenizer_kwargs(model_base, config=lora_cfg_pretrained),
            )
            print('Loading multimodal model from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional multimodal weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading multimodal model from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = _load_tokenizer_with_fast_fallback(
                    model_base,
                    use_fast=False,
                    **qwen_tokenizer_kwargs(model_base),
                )
                cfg_pretrained = _load_auto_config(model_path)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                cfg_pretrained = _load_auto_config(model_path)
                _apply_model_config_overrides(cfg_pretrained, model_config_overrides)
                tokenizer = _load_tokenizer_with_fast_fallback(
                    model_base,
                    use_fast=False,
                    **qwen_tokenizer_kwargs(model_base, config=cfg_pretrained),
                )
                model_type = getattr(cfg_pretrained, 'model_type', '')
                if 'qwen3' in model_type.lower():
                    model = LlavaQwen3ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                else:
                    model = LlavaQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = _load_tokenizer_with_fast_fallback(
                    model_path,
                    use_fast=False,
                    **qwen_tokenizer_kwargs(model_path),
                )
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = _load_tokenizer_with_fast_fallback(
                    model_path,
                    use_fast=False,
                    **qwen_tokenizer_kwargs(model_path),
                )
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            elif 'dclm' in model_name.lower():
                tokenizer = _load_tokenizer_with_fast_fallback(
                    model_path,
                    use_fast=False,
                    **qwen_tokenizer_kwargs(model_path),
                )
                model = LlavaOpenlmForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                cfg = _load_auto_config(model_path)
                _apply_model_config_overrides(cfg, model_config_overrides)
                tokenizer = _load_tokenizer_with_fast_fallback(
                    model_path,
                    use_fast=_resolve_tokenizer_use_fast(False, tokenizer_use_fast),
                    **qwen_tokenizer_kwargs(model_path, config=cfg),
                )
                model_type = getattr(cfg, 'model_type', '')
                if 'qwen3' in model_type.lower():
                    model = LlavaQwen3ForCausalLM.from_pretrained(
                        model_path,
                        config=cfg,
                        low_cpu_mem_usage=True,
                        **kwargs
                    )
                else:
                    model = LlavaQwen2ForCausalLM.from_pretrained(
                        model_path,
                        config=cfg,
                        low_cpu_mem_usage=True,
                        **kwargs
                    )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = _load_tokenizer_with_fast_fallback(
                model_base,
                use_fast=False,
                **qwen_tokenizer_kwargs(model_base),
            )
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = _load_tokenizer_with_fast_fallback(
                    model_path,
                    use_fast=False,
                    **qwen_tokenizer_kwargs(model_path),
                )
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                cfg = _load_auto_config(model_path)
                tokenizer = _load_tokenizer_with_fast_fallback(
                    model_path,
                    use_fast=False,
                    **qwen_tokenizer_kwargs(model_path, config=cfg),
                )
                model = AutoModelForCausalLM.from_pretrained(model_path, config=cfg, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if is_mllm_model:
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=resolved_device_map)
        if hasattr(vision_tower, 'set_llm_hidden_size'):
            vision_tower.set_llm_hidden_size(model.config.hidden_size)
            _load_multimodal_weights_if_present(model, model_path)
        vision_dtype = _vision_tower_runtime_dtype(vision_tower, device)
        if isinstance(resolved_device_map, dict):
            vision_tower.to(device=device, dtype=vision_dtype)
        elif resolved_device_map is None:
            vision_tower.to(device=device, dtype=vision_dtype)
        elif resolved_device_map != 'auto':
            vision_tower.to(device=resolved_device_map, dtype=vision_dtype)
        image_processor = vision_tower.image_processor

    sync_qwen_token_config(
        tokenizer=tokenizer,
        model=model,
        model_name_or_path=model_path,
    )

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
