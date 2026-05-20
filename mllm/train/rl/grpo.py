from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import random
from typing import Any, Optional, Sequence

from PIL import Image
import torch
try:
    import torch_npu  # noqa: F401
except ModuleNotFoundError:
    torch_npu = None
from torch.utils.data import Dataset
import transformers
from transformers import Trainer

from mllm import conversation as conversation_lib
from mllm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from mllm.coord_utils import record_coord_config
from mllm.mm_utils import process_anyres_image, tokenizer_image_token
from mllm.model.builder import load_pretrained_model
from mllm.reward import MapRewardConfig, compute_map_reward
from mllm.train.checkpoint_metadata import sync_qwen_multimodal_config, write_qwen_multimodal_checkpoint_metadata
from mllm.train.train_qwen import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    resolve_lora_target_modules,
)


def _load_json_or_jsonl(file_path: str) -> list[dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class GRPOModelArguments:
    model_name_or_path: str = field(metadata={"help": "SFT checkpoint or base multimodal checkpoint."})
    model_base: Optional[str] = None
    version: str = "conv_qwen_3_Dinov2_huawei"
    vision_tower: Optional[str] = None
    input_image_size: Optional[int] = None
    deepstack_visual_indexes: Optional[list[int]] = None
    disable_deepstack: bool = False


@dataclass
class GRPODataArguments:
    data_path: list[str] = field(default_factory=list)
    image_folder: list[str] = field(default_factory=list)
    image_aspect_ratio: str = "pad"
    image_grid_pinpoints: Optional[str] = None
    train_sample_limit: Optional[int] = None
    sample_seed: int = 42
    training_branch: str = "auto"
    map_task: str = "lane"
    patch_size: int = 256
    coord_mode: str = "auto"
    coord_range: int = 1000


@dataclass
class GRPOTrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = False
    model_max_length: int = 4096
    grpo_backend: str = "custom"
    num_generations: int = 4
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    kl_beta: float = 0.02
    clip_range: float = 0.2
    lora_enable: bool = True
    lora_target_scope: str = "llm"
    lora_target_modules: Optional[str] = None
    lora_exclude_modules: Optional[str] = "lm_head,embed_tokens"
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    reward_format_weight: float = 0.08
    reward_centerline_instance_weight: float = 0.37
    reward_centerline_length_weight: float = 0.45
    reward_cut_type_weight: float = 0.05
    reward_cut_continuity_weight: float = 0.05
    reward_intersection_weight: float = 0.0
    npu_zero3_disable_synced_generation: bool = False


def _bind_npu_device_from_local_rank() -> Optional[int]:
    """Bind each torchrun worker to its visible NPU before distributed init."""
    if torch_npu is None or not hasattr(torch, "npu"):
        return None
    is_available = getattr(torch.npu, "is_available", None)
    if not callable(is_available) or not is_available():
        return None

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        local_rank = 0
    device_count = int(torch.npu.device_count())
    if local_rank >= device_count:
        raise ValueError(
            f"LOCAL_RANK={local_rank} but only {device_count} visible NPU device(s). "
            "Check ASCEND_RT_VISIBLE_DEVICES and torchrun --nproc_per_node."
        )
    torch.npu.set_device(local_rank)
    return local_rank


def _unwrap_module(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _generation_model(model):
    """Return the object that owns generate(); DDP keeps it on model.module."""
    if hasattr(model, "generate"):
        return model
    unwrapped = _unwrap_module(model)
    if hasattr(unwrapped, "generate"):
        return unwrapped
    raise AttributeError(f"{type(model).__name__} does not expose generate().")


@contextmanager
def _temporarily_eval(model):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


@contextmanager
def _disable_lora_adapter(model):
    unwrapped = _unwrap_module(model)
    disable_adapter = getattr(unwrapped, "disable_adapter", None)
    if not callable(disable_adapter):
        raise RuntimeError(
            "LoRA reference KL requested, but the wrapped model does not expose disable_adapter()."
        )
    with disable_adapter():
        yield


def _normalize_map_task(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {
        "lane": "lane",
        "lane_only": "lane",
        "centerline": "lane",
        "phase_a_lane": "lane",
        "phase_b_lane": "lane",
        "lane_intersection": "lane_intersection",
        "lane+intersection": "lane_intersection",
        "intersection_lane": "lane_intersection",
        "phase_a_lane_intersection": "lane_intersection",
        "phase_b_lane_intersection": "lane_intersection",
    }
    return aliases.get(text)


def _normalize_training_branch(branch: str, map_task: str) -> tuple[str, str, str | None]:
    # Phase and task are independent axes in the data pipeline:
    # phase_a/phase_b controls incoming state hints, while lane/lane_intersection
    # controls the target schema. Keep both axes explicit to avoid silent mixing.
    branch = str(branch or "auto").strip().lower()
    map_task = _normalize_map_task(map_task) or str(map_task).strip().lower()
    branch_map = {
        "auto": (f"auto_{map_task}", map_task, None),
        "auto_lane": ("auto_lane", "lane", None),
        "auto_lane_intersection": ("auto_lane_intersection", "lane_intersection", None),
        "phase_a_lane": ("phase_a_lane", "lane", "phase_a"),
        "a_lane": ("phase_a_lane", "lane", "phase_a"),
        "phase_b_lane": ("phase_b_lane", "lane", "phase_b"),
        "b_lane": ("phase_b_lane", "lane", "phase_b"),
        "phase_a_lane_intersection": ("phase_a_lane_intersection", "lane_intersection", "phase_a"),
        "a_lane_intersection": ("phase_a_lane_intersection", "lane_intersection", "phase_a"),
        "phase_b_lane_intersection": ("phase_b_lane_intersection", "lane_intersection", "phase_b"),
        "b_lane_intersection": ("phase_b_lane_intersection", "lane_intersection", "phase_b"),
    }
    if branch not in branch_map:
        raise ValueError(
            "Unsupported training_branch. Use auto_lane, auto_lane_intersection, "
            "phase_a_lane, phase_b_lane, phase_a_lane_intersection, or phase_b_lane_intersection."
        )
    resolved_branch, resolved_task, expected_phase = branch_map[branch]
    if branch != "auto" and map_task != resolved_task:
        raise ValueError(
            f"training_branch={resolved_branch} requires map_task={resolved_task}, got {map_task}."
        )
    return resolved_branch, resolved_task, expected_phase


def _record_phase(record: dict[str, Any]) -> str | None:
    # Data generated by different smoke/production builders stores phase in
    # slightly different places; normalize those variants before validation.
    phase = record.get("phase") or record.get("debug_phase")
    if phase is None and isinstance(record.get("meta"), dict):
        phase = record["meta"].get("phase")
    if phase is None:
        return None
    text = str(phase).strip().lower()
    if text in {"a", "phase_a"}:
        return "phase_a"
    if text in {"b", "phase_b"}:
        return "phase_b"
    return text


def _validate_record_branch(record: dict[str, Any], data_args: GRPODataArguments, expected_phase: str | None):
    # Fail fast when a lane-only run is given lane+intersection data, or when
    # a Phase B run is accidentally pointed at Phase A JSONL. Silent mixing would
    # make reward curves hard to interpret.
    record_task = _normalize_map_task(record.get("map_task") or record.get("task"))
    if record_task is not None and record_task != data_args.map_task:
        raise ValueError(
            f"Sample {record.get('id', record.get('record_id', '<unknown>'))} has map_task={record_task}, "
            f"but training branch expects {data_args.map_task}."
        )
    record_phase = _record_phase(record)
    if expected_phase is not None and record_phase is not None and record_phase != expected_phase:
        raise ValueError(
            f"Sample {record.get('id', record.get('record_id', '<unknown>'))} has phase={record_phase}, "
            f"but training branch expects {expected_phase}."
        )


def _extract_prompt_and_gt(sample: dict[str, Any]) -> tuple[str, str]:
    conversations = sample.get("conversations") or []
    prompt = sample.get("prompt")
    ground_truth = sample.get("ground_truth") or sample.get("labels")
    if prompt is None:
        for item in conversations:
            if item.get("from") in {"human", "user"}:
                prompt = item.get("value", "")
                break
    if ground_truth is None:
        for item in conversations:
            if item.get("from") in {"gpt", "assistant"}:
                ground_truth = item.get("value", "")
                break
    if prompt is None or ground_truth is None:
        raise ValueError("RL sample must contain a prompt/user turn and a ground-truth assistant JSON.")
    return str(prompt), str(ground_truth)


class MapRLDataset(Dataset):
    def __init__(self, tokenizer, data_args: GRPODataArguments, image_processor):
        records = []
        for idx, data_path in enumerate(data_args.data_path):
            loaded = _load_json_or_jsonl(data_path)
            records.extend({**entry, "img_path_idx": idx} for entry in loaded)
        _, _, expected_phase = _normalize_training_branch(data_args.training_branch, data_args.map_task)
        for record in records:
            _validate_record_branch(record, data_args, expected_phase)
        if data_args.train_sample_limit and len(records) > data_args.train_sample_limit:
            rng = random.Random(data_args.sample_seed)
            ids = sorted(rng.sample(range(len(records)), data_args.train_sample_limit))
            records = [records[i] for i in ids]
        self.records = records
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_processor = image_processor

    def __len__(self):
        return len(self.records)

    def _build_prompt(self, user_prompt: str) -> str:
        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _load_image(self, sample: dict[str, Any]):
        image_file = sample["image"]
        image_folder = self.data_args.image_folder[sample["img_path_idx"]]
        image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        image_size = image.size
        processor = self.image_processor
        if self.data_args.image_aspect_ratio == "pad":
            width, height = image.size
            if width != height:
                side = max(width, height)
                canvas = Image.new(image.mode, (side, side), tuple(int(x * 255) for x in processor.image_mean))
                canvas.paste(image, ((side - width) // 2, (side - height) // 2))
                image = canvas
            image_tensor = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        elif self.data_args.image_aspect_ratio == "anyres" or "anyres_max" in self.data_args.image_aspect_ratio:
            image_tensor = process_anyres_image(image, processor, self.data_args.image_grid_pinpoints)
        else:
            image_tensor = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image_tensor, image_size

    def __getitem__(self, index: int):
        sample = self.records[index]
        prompt_text, ground_truth = _extract_prompt_and_gt(sample)
        prompt = self._build_prompt(prompt_text)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        result = {
            "input_ids": input_ids,
            "ground_truth": ground_truth,
            "record_id": sample.get("id", sample.get("record_id", str(index))),
        }
        if "image" in sample:
            image, image_size = self._load_image(sample)
            result["image"] = image
            result["image_size"] = image_size
        return result


class MapRLCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[dict[str, Any]]) -> dict[str, Any]:
        input_ids = [item["input_ids"] for item in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        batch = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "ground_truth": [item["ground_truth"] for item in instances],
            "record_id": [item["record_id"] for item in instances],
        }
        if "image" in instances[0]:
            images = [item["image"] for item in instances]
            batch["image_sizes"] = [item["image_size"] for item in instances]
            if all(image.shape == images[0].shape for image in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images
        return batch


class MapGRPOTrainer(Trainer):
    def __init__(self, *args, ref_model=None, reward_config=None, tokenizer=None, use_lora_reference=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.reward_config = reward_config or MapRewardConfig()
        self.tokenizer = tokenizer or self.processing_class
        if self.processing_class is None and tokenizer is not None:
            self.processing_class = tokenizer
        self.use_lora_reference = use_lora_reference
        self._logged_npu_zero3_generation_workaround = False
        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad_(False)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        model = self.model
        if hasattr(model, "peft_config"):
            sync_qwen_multimodal_config(model)
            # ZeRO-3 parameter gather is collective. Gather on every rank,
            # then only world process zero writes the adapter and metadata.
            lora_state = get_peft_state_maybe_zero_3(model.named_parameters(), getattr(self.args, "lora_bias", "none"))
            non_lora_state = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
            if self.is_world_process_zero():
                os.makedirs(output_dir, exist_ok=True)
                model.config.save_pretrained(output_dir)
                model.save_pretrained(output_dir, state_dict=lora_state)
                tokenizer_to_save = self.processing_class or self.tokenizer
                if tokenizer_to_save is not None and hasattr(tokenizer_to_save, "save_pretrained"):
                    tokenizer_to_save.save_pretrained(output_dir)
                torch.save(non_lora_state, os.path.join(output_dir, "non_lora_trainables.bin"))
                write_qwen_multimodal_checkpoint_metadata(model, output_dir, self)
            return
        super().save_model(output_dir, _internal_call=_internal_call)
        write_qwen_multimodal_checkpoint_metadata(model, output_dir, self)

    def _sample_image(self, images, index: int):
        if images is None:
            return None
        if isinstance(images, torch.Tensor):
            return images[index].to(self.model.device)
        return images[index].to(self.model.device)

    def _completion_from_output(self, output_ids: torch.Tensor, prompt_ids: torch.Tensor):
        if output_ids.numel() >= prompt_ids.numel() and torch.equal(output_ids[:prompt_ids.numel()], prompt_ids):
            return output_ids[prompt_ids.numel():]
        return output_ids

    def _sequence_logprob(self, model, prompt_ids, completion_ids, image, image_size):
        if completion_ids.numel() == 0:
            completion_ids = torch.tensor([self.tokenizer.eos_token_id], device=prompt_ids.device, dtype=prompt_ids.dtype)
        full_ids = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0)
        labels = full_ids.clone()
        labels[:, :prompt_ids.numel()] = IGNORE_INDEX
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
        kwargs = {"input_ids": full_ids, "labels": labels}
        if image is not None:
            kwargs["images"] = image.unsqueeze(0)
            kwargs["image_sizes"] = [image_size]
        outputs = model(**kwargs)
        return -outputs.loss

    def _reference_logprob(self, model, prompt_ids, completion_ids, image, image_size):
        if self.ref_model is not None:
            with torch.no_grad():
                return self._sequence_logprob(self.ref_model, prompt_ids, completion_ids, image, image_size)
        if self.use_lora_reference:
            # In LoRA GRPO, the reference policy is the same SFT model with
            # adapters disabled. This keeps ZeRO-3 parameters inside the active
            # DeepSpeed engine instead of loading a second unwrapped model.
            with _disable_lora_adapter(model):
                with _temporarily_eval(model), torch.no_grad():
                    return self._sequence_logprob(model, prompt_ids, completion_ids, image, image_size)
        return None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        images = inputs.get("images")
        image_sizes = inputs.get("image_sizes")
        ground_truths = inputs["ground_truth"]
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        losses = []
        reward_values = []
        component_sums: dict[str, float] = {}

        for row_idx in range(input_ids.shape[0]):
            prompt_ids = input_ids[row_idx][attention_mask[row_idx].bool()].to(model.device)
            image = self._sample_image(images, row_idx)
            image_size = image_sizes[row_idx] if image_sizes is not None else None
            gen_images = image.unsqueeze(0).repeat(self.args.num_generations, 1, 1, 1) if image is not None else None
            gen_inputs = prompt_ids.unsqueeze(0).repeat(self.args.num_generations, 1)
            gen_attention = torch.ones_like(gen_inputs, device=model.device)
            generation_kwargs = {
                "attention_mask": gen_attention,
                "max_new_tokens": self.args.max_new_tokens,
                "use_cache": True,
                "do_sample": self.args.temperature > 0,
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "num_beams": 1,
                "pad_token_id": pad_token_id,
                "eos_token_id": eos_token_id,
            }
            if _should_disable_synced_generation_on_npu_zero3(self.args):
                # torch_npu can fail inside HF GenerationMixin's synced_gpus
                # scalar all-reduce path under ZeRO-3. Disable that path and
                # force fixed-length generation so every rank still executes
                # the same number of ZeRO-3 forward calls.
                generation_kwargs["synced_gpus"] = False
                generation_kwargs["eos_token_id"] = None
                if not self._logged_npu_zero3_generation_workaround and self.is_world_process_zero():
                    print(
                        "[mllm-grpo] NPU ZeRO3 generation: disabled HF synced_gpus "
                        "and EOS early-stop; generating fixed max_new_tokens per rank."
                    )
                    self._logged_npu_zero3_generation_workaround = True
            if gen_images is not None:
                generation_kwargs["images"] = gen_images
                generation_kwargs["image_sizes"] = [image_size] * self.args.num_generations
            with torch.no_grad():
                generated = _generation_model(model).generate(gen_inputs, **generation_kwargs)
            rewards = []
            completions = []
            for out in generated:
                completion = self._completion_from_output(out.to(model.device), prompt_ids)
                completions.append(completion)
                text = self.tokenizer.decode(completion, skip_special_tokens=False)
                reward_payload = compute_map_reward(text, ground_truths[row_idx], self.reward_config)
                rewards.append(float(reward_payload["reward"]))
                reward_values.append(float(reward_payload["reward"]))
                for key, value in reward_payload.get("components", {}).items():
                    component_sums[key] = component_sums.get(key, 0.0) + float(value)

            reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=model.device)
            advantages = reward_tensor - reward_tensor.mean()
            std = reward_tensor.std(unbiased=False)
            if std > 1e-6:
                advantages = advantages / (std + 1e-6)

            for completion, advantage in zip(completions, advantages):
                with torch.no_grad():
                    old_logp = self._sequence_logprob(model, prompt_ids, completion, image, image_size).detach()
                logp = self._sequence_logprob(model, prompt_ids, completion, image, image_size)
                ratio = torch.exp(logp - old_logp)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.args.clip_range, 1.0 + self.args.clip_range)
                pg_loss = -torch.minimum(ratio * advantage.detach(), clipped_ratio * advantage.detach())
                ref_logp = self._reference_logprob(model, prompt_ids, completion, image, image_size)
                if ref_logp is not None:
                    kl_loss = (logp - ref_logp.detach()).pow(2)
                else:
                    kl_loss = torch.zeros_like(pg_loss)
                losses.append(pg_loss + self.args.kl_beta * kl_loss)

        if not losses:
            loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        else:
            loss = torch.stack(losses).mean()
        if reward_values:
            metrics = {"grpo_reward": sum(reward_values) / len(reward_values)}
            for key, value in component_sums.items():
                metrics[f"reward_{key}"] = value / len(reward_values)
            self.log(metrics)
        return (loss, {}) if return_outputs else loss


def _model_config_overrides(model_args: GRPOModelArguments) -> dict[str, Any]:
    return {
        "mm_vision_tower": model_args.vision_tower,
        "vision_tower": model_args.vision_tower,
        "input_image_size": model_args.input_image_size,
        "disable_deepstack": model_args.disable_deepstack,
        "deepstack_visual_indexes": model_args.deepstack_visual_indexes,
    }


def _uses_deepspeed_zero3(training_args: GRPOTrainingArguments) -> bool:
    deepspeed_config = getattr(training_args, "deepspeed", None)
    if not deepspeed_config:
        return False
    if isinstance(deepspeed_config, (str, os.PathLike)):
        try:
            with open(deepspeed_config, "r", encoding="utf-8") as f:
                deepspeed_config = json.load(f)
        except Exception:
            return False
    if not isinstance(deepspeed_config, dict):
        return False
    zero_config = deepspeed_config.get("zero_optimization", {})
    return int(zero_config.get("stage", 0) or 0) == 3


def _is_npu_device(device) -> bool:
    return str(device or "").lower().startswith("npu")


def _should_disable_synced_generation_on_npu_zero3(training_args: GRPOTrainingArguments) -> bool:
    if not getattr(training_args, "npu_zero3_disable_synced_generation", True):
        return False
    return _uses_deepspeed_zero3(training_args) and _is_npu_device(getattr(training_args, "device", None))


def _load_policy_components(model_args: GRPOModelArguments, training_args: GRPOTrainingArguments):
    device_map = None if _uses_deepspeed_zero3(training_args) else {"": training_args.device}
    load_kwargs = {}
    if training_args.bf16:
        load_kwargs["torch_dtype"] = torch.bfloat16
    elif training_args.fp16:
        load_kwargs["torch_dtype"] = torch.float16
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_args.model_name_or_path,
        model_base=model_args.model_base,
        model_name=f"mllm_policy_{Path(model_args.model_name_or_path).name}",
        device_map=device_map,
        device=str(training_args.device),
        model_config_overrides=_model_config_overrides(model_args),
        # GRPO generation/reward follows the SFT slow-tokenizer path. Keep it
        # internal so cloud scripts stay compatible with older argument parsers.
        tokenizer_use_fast=False,
        **load_kwargs,
    )
    if device_map is None:
        model.to(training_args.device)
    tokenizer.model_max_length = training_args.model_max_length
    return tokenizer, model, image_processor


def _apply_lora(model, training_args: GRPOTrainingArguments):
    if not training_args.lora_enable:
        return model
    from peft import LoraConfig, get_peft_model

    targets = resolve_lora_target_modules(
        model,
        target_scope=training_args.lora_target_scope,
        target_modules=training_args.lora_target_modules,
        exclude_modules=training_args.lora_exclude_modules,
    )
    if not targets:
        raise ValueError("No LoRA target modules resolved for GRPO.")
    config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=targets,
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def train():
    bound_npu_device = _bind_npu_device_from_local_rank()
    parser = transformers.HfArgumentParser((GRPOModelArguments, GRPODataArguments, GRPOTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if str(os.environ.get("RANK", "0")) in {"0", "-1"}:
        import mllm.model.builder as builder_module

        print(
            "[mllm-grpo] startup: "
            f"grpo_file={__file__}, builder_file={builder_module.__file__}, "
            f"model={model_args.model_name_or_path}, vision={model_args.vision_tower}, "
            f"zero3={_uses_deepspeed_zero3(training_args)}, lora={training_args.lora_enable}, "
            f"kl_beta={training_args.kl_beta}, bf16={training_args.bf16}, fp16={training_args.fp16}, "
            f"bound_npu_device={bound_npu_device}, "
            f"npu_zero3_disable_synced_generation={training_args.npu_zero3_disable_synced_generation}"
        )
    resolved_branch, resolved_task, _ = _normalize_training_branch(data_args.training_branch, data_args.map_task)
    data_args.training_branch = resolved_branch
    data_args.map_task = resolved_task
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    if training_args.grpo_backend == "trl":
        try:
            import trl  # noqa: F401
            print("TRL is installed; using the MLLM VLM adapter for image-aware GRPO batches.")
        except ImportError as exc:
            raise ImportError("TRL backend requested but trl is not installed. Install project optional dependency: mllm[rl].") from exc
    elif training_args.grpo_backend != "custom":
        raise ValueError(f"Unsupported grpo_backend: {training_args.grpo_backend}")
    uses_zero3 = _uses_deepspeed_zero3(training_args)
    use_lora_reference = uses_zero3 and training_args.lora_enable and training_args.kl_beta > 0
    if uses_zero3 and training_args.kl_beta > 0 and not training_args.lora_enable:
        # Full-parameter ZeRO-3 cannot reuse the adapter-disabled trick because
        # there is no adapter boundary. A correct implementation needs a second
        # DeepSpeed-wrapped reference engine.
        raise ValueError(
            "Custom GRPO with DeepSpeed ZeRO-3 and --kl_beta > 0 currently requires LoRA. "
            "Full-parameter ZeRO-3 reference KL needs a separately wrapped reference DeepSpeed engine."
        )

    tokenizer, model, image_processor = _load_policy_components(model_args, training_args)
    ref_model = None
    if training_args.kl_beta > 0 and not use_lora_reference:
        ref_tokenizer, ref_model, _ = _load_policy_components(model_args, training_args)
        del ref_tokenizer
    model = _apply_lora(model, training_args)
    dataset = MapRLDataset(tokenizer=tokenizer, data_args=data_args, image_processor=image_processor)
    coord_mode = data_args.coord_mode
    coord_range = data_args.coord_range
    if coord_mode == "auto":
        coord_cfg = record_coord_config(
            dataset.records[0] if dataset.records else {},
            default_mode="pixel",
            default_patch_size=data_args.patch_size,
            default_coord_range=data_args.coord_range,
        )
        coord_mode = coord_cfg["coord_mode"]
        coord_range = coord_cfg["coord_range"]
    reward_config = MapRewardConfig(
        map_task=data_args.map_task,
        patch_size=data_args.patch_size,
        coord_mode=coord_mode,
        coord_range=coord_range,
        format_weight=training_args.reward_format_weight,
        centerline_instance_weight=training_args.reward_centerline_instance_weight,
        centerline_length_weight=training_args.reward_centerline_length_weight,
        cut_type_weight=training_args.reward_cut_type_weight,
        cut_continuity_weight=training_args.reward_cut_continuity_weight,
        intersection_weight=training_args.reward_intersection_weight,
    )
    collator = MapRLCollator(tokenizer)
    trainer = MapGRPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_config=reward_config,
        tokenizer=tokenizer,
        use_lora_reference=use_lora_reference,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
