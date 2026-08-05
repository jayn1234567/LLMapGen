from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import time
from typing import Optional

import torch
import transformers
from transformers import TrainerCallback

from mllm.torch_runtime import maybe_disable_cudnn_from_env

maybe_disable_cudnn_from_env(torch)

try:
    from transformers.training_args import ParallelismConfig  # type: ignore
except Exception:
    class ParallelismConfig:  # type: ignore
        pass
    try:
        transformers.training_args.ParallelismConfig = ParallelismConfig  # type: ignore[attr-defined]
    except Exception:
        pass

from mllm.train.swanlab_utils import build_swanlab_callback

from .data import NativeQwen3VLDataCollator, NativeQwen3VLDataset, normalize_path_list
from .modeling import load_native_model, load_processor, trainer_processor_kwarg


def rank0() -> bool:
    rank = os.environ.get("RANK")
    return rank is None or int(rank) == 0


def rank0_print(*args, **kwargs) -> None:
    if rank0():
        print(*args, **kwargs)


@dataclass
class NativeModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3-VL-8B-Instruct")
    model_base: Optional[str] = field(default=None)
    trust_remote_code: bool = field(default=True)
    attn_implementation: Optional[str] = field(default=None)
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    vision_lora_enable: bool = field(default=False)
    vision_lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,qkv,proj"
    )
    unfreeze_multimodal_merger: bool = field(default=False)


@dataclass
class NativeDataArguments:
    data_path: list[str] = field(default_factory=list)
    image_folder: list[str] = field(default_factory=list)
    eval_data_path: Optional[list[str]] = field(default=None)
    eval_image_folder: Optional[list[str]] = field(default=None)
    train_sample_limit: int = field(default=0)
    eval_sample_limit: int = field(default=0)
    sample_seed: int = field(default=42)
    system_prompt: Optional[str] = field(default=None)


@dataclass
class NativeTrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(default=4096)
    vision_lora_learning_rate: Optional[float] = field(default=None)
    merger_learning_rate: Optional[float] = field(default=None)
    save_best_train_loss: bool = field(default=False)
    best_train_loss_start_step: int = field(default=0)
    best_train_loss_dir: str = field(default="best")
    best_checkpoint_keep_limit: int = field(default=5)
    use_hf_progress_bar: bool = field(default=True)

    swanlab_enable: bool = field(default=False)
    swanlab_project: str = field(default="")
    swanlab_workspace: str = field(default="")
    swanlab_experiment_name: str = field(default="")
    swanlab_group: str = field(default="")
    swanlab_job_type: str = field(default="sft")
    swanlab_description: str = field(default="")
    swanlab_tags: str = field(default="")
    swanlab_mode: str = field(default="offline")
    swanlab_log_dir: str = field(default="")
    swanlab_api_host: str = field(default="")
    swanlab_web_host: str = field(default="")


class JsonlMetricLoggerCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.train_log_path = self.output_dir / "trainer_log.jsonl"
        self.eval_log_path = self.output_dir / "eval_log.jsonl"

    def _append(self, path: Path, payload: dict):
        if not rank0():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        payload = {"time": time.time(), "global_step": state.global_step, "epoch": state.epoch, **logs}
        if rank0():
            world_size = max(int(getattr(args, "world_size", 1) or 1), 1)
            throughput = float(logs.get("train_samples_per_second", 0.0) or 0.0) / world_size
            print(f"DI_throughput: {throughput:.2f} samples/s/npu", flush=True)
        if "eval_loss" in logs or any(str(key).startswith("eval_") for key in logs):
            self._append(self.eval_log_path, payload)
        else:
            self._append(self.train_log_path, payload)


class BestTrainLossCallback(TrainerCallback):
    def __init__(self):
        self.best_loss: float | None = None
        self.trainer = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not getattr(args, "save_best_train_loss", False) or not logs or "loss" not in logs:
            return
        if state.global_step < int(getattr(args, "best_train_loss_start_step", 0) or 0):
            return
        try:
            loss = float(logs["loss"])
        except (TypeError, ValueError):
            return
        if self.best_loss is not None and loss >= self.best_loss:
            return
        self.best_loss = loss
        if self.trainer is None or not rank0():
            return

        root = Path(args.best_train_loss_dir)
        if not root.is_absolute():
            root = Path(args.output_dir) / root
        root.mkdir(parents=True, exist_ok=True)
        ckpt_dir = root / f"loss_{loss:.6g}_step_{state.global_step}".replace("-", "m").replace(".", "p")
        self.trainer.save_model(str(ckpt_dir))
        processor = getattr(self.trainer, "processing_class", None) or getattr(self.trainer, "tokenizer", None)
        if processor is not None and hasattr(processor, "save_pretrained"):
            processor.save_pretrained(str(ckpt_dir))
        metadata = {
            "best_train_loss": loss,
            "best_train_loss_step": state.global_step,
            "checkpoint_dir": str(ckpt_dir),
            "time": time.time(),
        }
        (ckpt_dir / "best_train_loss.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        candidates = sorted(root.glob("loss_*_step_*"), key=lambda path: path.stat().st_mtime, reverse=True)
        keep = int(getattr(args, "best_checkpoint_keep_limit", 5) or 5)
        for stale in candidates[keep:]:
            shutil.rmtree(stale, ignore_errors=True)
        rank0_print(f"Saved native Qwen3-VL best train-loss checkpoint: {ckpt_dir}")


def _is_native_visual_name(name: str) -> bool:
    return ".visual." in f".{name}."


def _is_native_merger_name(name: str) -> bool:
    return _is_native_visual_name(name) and "merger" in name.lower()


class NativeMultimodalTrainer(transformers.Trainer):
    """Trainer with separate LoRA/merger LRs and complete PEFT checkpoints."""

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        grouped: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
        default_lr = float(self.args.learning_rate)
        vision_lr = float(self.args.vision_lora_learning_rate or default_lr)
        merger_lr = float(self.args.merger_learning_rate or default_lr)
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if _is_native_merger_name(name):
                lr = merger_lr
            elif _is_native_visual_name(name):
                lr = vision_lr
            else:
                lr = default_lr
            decay = 0.0 if name.endswith(".bias") or parameter.ndim <= 1 else float(self.args.weight_decay)
            grouped.setdefault((lr, decay), []).append(parameter)

        optimizer_groups = [
            {"params": parameters, "lr": lr, "weight_decay": decay}
            for (lr, decay), parameters in grouped.items()
            if parameters
        ]
        try:
            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(
                self.args, self.model
            )
        except TypeError:
            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_groups, **optimizer_kwargs)
        rank0_print(
            "Native optimizer LRs: "
            f"language_lora={default_lr}, vision_lora={vision_lr}, merger={merger_lr}"
        )
        return self.optimizer

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        output_path = Path(output_dir or self.args.output_dir)
        merger_state = {
            name: parameter.detach().cpu()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and _is_native_merger_name(name)
        }
        if merger_state:
            torch.save(merger_state, output_path / "native_non_lora_trainables.bin")
            rank0_print(
                f"Saved {len(merger_state)} trainable native merger tensors to {output_path}"
            )


def _dataset_paths(values: Optional[list[str] | str]) -> list[str]:
    return normalize_path_list(values)


def make_data_module(processor, data_args: NativeDataArguments, training_args: NativeTrainingArguments):
    train_dataset = NativeQwen3VLDataset(
        _dataset_paths(data_args.data_path),
        _dataset_paths(data_args.image_folder),
        sample_limit=data_args.train_sample_limit,
        sample_seed=data_args.sample_seed,
    )
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = NativeQwen3VLDataset(
            _dataset_paths(data_args.eval_data_path),
            _dataset_paths(data_args.eval_image_folder or data_args.image_folder),
            sample_limit=data_args.eval_sample_limit,
            sample_seed=data_args.sample_seed,
        )
    data_collator = NativeQwen3VLDataCollator(
        processor,
        model_max_length=training_args.model_max_length,
        system_prompt=data_args.system_prompt,
    )
    return {"train_dataset": train_dataset, "eval_dataset": eval_dataset, "data_collator": data_collator}


def resolve_language_lora_targets(model, requested_names: list[str]) -> list[str]:
    requested = set(requested_names)
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and ".language_model." in f".{name}."
        and name.rsplit(".", 1)[-1] in requested
    ]
    if not targets:
        raise ValueError(
            "Unable to find native Qwen3-VL language-model LoRA targets for "
            f"{sorted(requested)!r}. The model layout may not match Qwen3-VL."
        )
    return targets


def resolve_vision_lora_targets(model, requested_names: list[str]) -> list[str]:
    requested = set(requested_names)
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and _is_native_visual_name(name)
        and not _is_native_merger_name(name)
        and name.rsplit(".", 1)[-1] in requested
    ]
    if not targets:
        raise ValueError(
            "Unable to find native Qwen3-VL visual LoRA targets for "
            f"{sorted(requested)!r}. The model layout may not match Qwen3-VL."
        )
    return targets


def unfreeze_native_merger(model) -> list[str]:
    trainable_names = []
    for name, parameter in model.named_parameters():
        if _is_native_merger_name(name):
            parameter.requires_grad_(True)
            trainable_names.append(name)
    if not trainable_names:
        raise ValueError("Requested native merger training, but no merger parameters were found")
    return trainable_names


def apply_lora(model, model_args: NativeModelArguments):
    if not model_args.lora_enable and not model_args.vision_lora_enable:
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("Native Qwen3-VL LoRA training requires the peft package") from exc

    resolved_targets = []
    if model_args.lora_enable:
        target_modules = [
            item.strip()
            for item in str(model_args.lora_target_modules).split(",")
            if item.strip()
        ]
        if not target_modules:
            raise ValueError("lora_target_modules must contain at least one module name")
        resolved_targets.extend(resolve_language_lora_targets(model, target_modules))
    if model_args.vision_lora_enable:
        vision_target_modules = [
            item.strip()
            for item in str(model_args.vision_lora_target_modules).split(",")
            if item.strip()
        ]
        if not vision_target_modules:
            raise ValueError("vision_lora_target_modules must contain at least one module name")
        resolved_targets.extend(resolve_vision_lora_targets(model, vision_target_modules))
    resolved_targets = sorted(set(resolved_targets))
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(model_args.lora_r),
        lora_alpha=int(model_args.lora_alpha),
        lora_dropout=float(model_args.lora_dropout),
        bias=str(model_args.lora_bias),
        target_modules=resolved_targets,
    )
    model = get_peft_model(model, config)
    merger_parameter_names = []
    if model_args.unfreeze_multimodal_merger:
        merger_parameter_names = unfreeze_native_merger(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if rank0() and hasattr(model, "print_trainable_parameters"):
        rank0_print(
            f"Native Qwen3-VL LoRA targets: {len(resolved_targets)} language/visual linear modules; "
            f"full-train merger tensors={len(merger_parameter_names)}"
        )
        model.print_trainable_parameters()
    return model


def train():
    parser = transformers.HfArgumentParser((NativeModelArguments, NativeDataArguments, NativeTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not training_args.use_hf_progress_bar:
        transformers.logging.disable_progress_bar()
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    processor = load_processor(model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code)
    model = load_native_model(model_args, training_args)
    model = apply_lora(model, model_args)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if training_args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    data_module = make_data_module(processor, data_args, training_args)
    callbacks: list[TrainerCallback] = [
        JsonlMetricLoggerCallback(training_args.output_dir),
        BestTrainLossCallback(),
    ]
    swanlab_callback = build_swanlab_callback(model_args, data_args, training_args)
    if swanlab_callback is not None:
        callbacks.append(swanlab_callback)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "callbacks": callbacks,
        **data_module,
        **trainer_processor_kwarg(processor),
    }
    trainer = NativeMultimodalTrainer(**trainer_kwargs)
    for callback in callbacks:
        if isinstance(callback, BestTrainLossCallback):
            callback.trainer = trainer

    rank0_print("Native Qwen3-VL SFT")
    rank0_print(f"model={model_args.model_name_or_path}")
    rank0_print(
        "lora="
        f"{model_args.lora_enable} r={model_args.lora_r} alpha={model_args.lora_alpha} "
        f"dropout={model_args.lora_dropout} targets={model_args.lora_target_modules}"
    )
    rank0_print(
        f"vision_lora={model_args.vision_lora_enable} "
        f"targets={model_args.vision_lora_target_modules} "
        f"full_train_merger={model_args.unfreeze_multimodal_merger}"
    )
    rank0_print(f"train={data_args.data_path}")
    rank0_print(f"system_prompt={data_args.system_prompt!r}")
    rank0_print(f"output={training_args.output_dir}")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    if rank0():
        metadata = {
            "backend": "native_qwen3vl",
            "model_name_or_path": model_args.model_name_or_path,
            "data_path": data_args.data_path,
            "image_folder": data_args.image_folder,
            "model_max_length": training_args.model_max_length,
            "system_prompt": data_args.system_prompt,
            "lora_enable": model_args.lora_enable,
            "lora_r": model_args.lora_r,
            "lora_alpha": model_args.lora_alpha,
            "lora_dropout": model_args.lora_dropout,
            "lora_target_modules": model_args.lora_target_modules,
            "vision_lora_enable": model_args.vision_lora_enable,
            "vision_lora_target_modules": model_args.vision_lora_target_modules,
            "unfreeze_multimodal_merger": model_args.unfreeze_multimodal_merger,
            "vision_lora_learning_rate": training_args.vision_lora_learning_rate,
            "merger_learning_rate": training_args.merger_learning_rate,
            "global_step": trainer.state.global_step,
        }
        Path(training_args.output_dir, "native_qwen3vl_checkpoint.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    train()
