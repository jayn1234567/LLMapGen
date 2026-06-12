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
    trust_remote_code: bool = field(default=True)
    attn_implementation: Optional[str] = field(default=None)
    replace_patch_embed_conv3d_with_linear: bool = field(default=False)


@dataclass
class NativeDataArguments:
    data_path: list[str] = field(default_factory=list)
    image_folder: list[str] = field(default_factory=list)
    eval_data_path: Optional[list[str]] = field(default=None)
    eval_image_folder: Optional[list[str]] = field(default=None)
    train_sample_limit: int = field(default=0)
    eval_sample_limit: int = field(default=0)
    sample_seed: int = field(default=42)


@dataclass
class NativeTrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(default=4096)
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
    data_collator = NativeQwen3VLDataCollator(processor, model_max_length=training_args.model_max_length)
    return {"train_dataset": train_dataset, "eval_dataset": eval_dataset, "data_collator": data_collator}


def train():
    parser = transformers.HfArgumentParser((NativeModelArguments, NativeDataArguments, NativeTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not training_args.use_hf_progress_bar:
        transformers.logging.disable_progress_bar()
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    processor = load_processor(model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code)
    model = load_native_model(model_args, training_args)
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
    trainer = transformers.Trainer(**trainer_kwargs)
    for callback in callbacks:
        if isinstance(callback, BestTrainLossCallback):
            callback.trainer = trainer

    rank0_print("Native Qwen3-VL SFT")
    rank0_print(f"model={model_args.model_name_or_path}")
    rank0_print(f"train={data_args.data_path}")
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
            "global_step": trainer.state.global_step,
        }
        Path(training_args.output_dir, "native_qwen3vl_checkpoint.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    train()
