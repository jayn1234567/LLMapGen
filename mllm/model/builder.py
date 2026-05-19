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


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, model_config_overrides=None, **kwargs):
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
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if is_mllm_model:
        # Load generic multimodal model. The "llava" branch name is kept for legacy checkpoints.
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from mllm.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            _apply_model_config_overrides(lora_cfg_pretrained, model_config_overrides)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, **qwen_tokenizer_kwargs(model_base))
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
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, **qwen_tokenizer_kwargs(model_base))
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, **qwen_tokenizer_kwargs(model_base))
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                _apply_model_config_overrides(cfg_pretrained, model_config_overrides)
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
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, **qwen_tokenizer_kwargs(model_path))
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, **qwen_tokenizer_kwargs(model_path))
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            elif 'dclm' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, **qwen_tokenizer_kwargs(model_path))
                model = LlavaOpenlmForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, **qwen_tokenizer_kwargs(model_path))
                cfg = AutoConfig.from_pretrained(model_path)
                _apply_model_config_overrides(cfg, model_config_overrides)
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
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, **qwen_tokenizer_kwargs(model_base))
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
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, **qwen_tokenizer_kwargs(model_path))
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, **qwen_tokenizer_kwargs(model_path))
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

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
