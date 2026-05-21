# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

from packaging import version
import os
import sys
import copy
import random
import shutil
import time 
from datetime import datetime
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import tokenizers
from tqdm.auto import tqdm as tqdm_auto

# Transformers 4.56 exposes TrainingArguments.parallelism_config annotated as
# ForwardRef("ParallelismConfig") but does not export that symbol in all wheels.
# HfArgumentParser resolves inherited annotations in this module's globals, so a
# local placeholder keeps argument parsing compatible. We never set this field.
try:
    from transformers.training_args import ParallelismConfig  # type: ignore
except Exception:  # pragma: no cover - version compatibility shim
    class ParallelismConfig:  # type: ignore
        pass
    try:
        import transformers.training_args as _transformers_training_args
        _transformers_training_args.ParallelismConfig = ParallelismConfig
    except Exception:
        pass

from mllm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from mllm.train.llava_trainer import LLaVATrainer
from mllm.train.checkpoint_metadata import (
    sync_qwen_multimodal_config,
    write_qwen_multimodal_checkpoint_metadata,
)
from mllm.train.swanlab_utils import build_swanlab_callback
from transformers import TrainerCallback
from transformers.trainer_callback import PrinterCallback, ProgressCallback

from mllm import conversation as conversation_lib
from mllm.model import *
from mllm.mm_utils import tokenizer_image_token, process_anyres_image
from mllm.model.builder import _load_multimodal_weights_if_present, _load_tokenizer_with_fast_fallback
from mllm.model.qwen3vl_extractor import is_qwen3vl_checkpoint, is_llava_checkpoint, ensure_extracted_llm_from_qwen3vl
from mllm.model.qwen_token_utils import qwen_tokenizer_kwargs, sync_qwen_token_config

from PIL import Image



local_rank = None


def _env_flag(name, default="0"):
    return str(os.environ.get(name, default)).lower() in ("1", "true", "yes", "on")


def _framework_env_flag(name, legacy_name=None, default="0"):
    if name in os.environ:
        return _env_flag(name, default)
    if legacy_name and legacy_name in os.environ:
        return _env_flag(legacy_name, default)
    return _env_flag(name, default)


def _is_global_rank0() -> bool:
    rank = os.environ.get("RANK")
    if rank is not None:
        return int(rank) == 0
    return local_rank in (None, -1, 0)


def silence_non_primary_rank_output():
    """Keep normal training logs on global rank 0 only."""
    if not _framework_env_flag("MLLM_LOG_RANK0_ONLY", "LLAVA_LOG_RANK0_ONLY", "1"):
        return
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank == 0:
        return
    sys.stdout.flush()
    sys.stdout = open(os.devnull, "w")
    if _framework_env_flag("MLLM_SUPPRESS_NONZERO_STDERR", "LLAVA_SUPPRESS_NONZERO_STDERR", "0"):
        sys.stderr.flush()
        sys.stderr = open(os.devnull, "w")


def rank0_print(*args):
    if _is_global_rank0():
        print(*args)


def _config_declares_qwen3(config) -> bool:
    model_type = str(getattr(config, "model_type", "") or "").lower()
    if "qwen3" in model_type or "qwen-3" in model_type:
        return True
    architectures = getattr(config, "architectures", None) or []
    return any("qwen3" in str(arch).lower() or "qwen-3" in str(arch).lower() for arch in architectures)


def _path_declares_qwen3(model_path: str) -> bool:
    path = str(model_path or "").lower()
    if any(kw in path for kw in ("qwen3", "qwen-3", "qwen_3")):
        return True
    try:
        config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        return False
    return _config_declares_qwen3(config)


class JsonlMetricLoggerCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.train_log_path = os.path.join(output_dir, "train_metrics.log")
        self.eval_log_path = os.path.join(output_dir, "eval_metrics.log")
        self.checkpoint_log_path = os.path.join(output_dir, "checkpoint_events.log")
        # 吞吐量计算
        self._throughput_start_time = None
        self._throughput_start_step = 0
        self._train_start_time = None
        self._saw_train_runtime_log = False

    def _is_rank0(self, args, state=None) -> bool:
        if state is not None and hasattr(state, "is_world_process_zero"):
            return state.is_world_process_zero
        rank = os.environ.get("RANK")
        if rank is not None:
            return int(rank) == 0
        return args.local_rank in (-1, 0)

    def _format_log_value(self, value):
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, (list, tuple)):
            return " ".join(str(item) for item in value)
        if isinstance(value, dict):
            return " ".join(f"{key}={self._format_log_value(val)}" for key, val in value.items())
        return str(value)

    def _format_log_line(self, payload: dict):
        return "  ".join(
            f"{key}: {self._format_log_value(value)}"
            for key, value in payload.items()
            if value is not None
        )

    def _append_log_line(self, path: str, payload: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        line = self._format_log_line(payload)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return line

    def _touch_log_file(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8"):
            pass

    def _compute_throughput(self, args, state):
        if self._throughput_start_time is None:
            return None
        step_diff = state.global_step - self._throughput_start_step
        time_diff = time.time() - self._throughput_start_time
        if step_diff <= 0 or time_diff <= 0:
            return None
        per_device_bs = getattr(args, "per_device_train_batch_size", 1)
        gas = getattr(args, "gradient_accumulation_steps", 1)
        max_len = getattr(args, "model_max_length", 4096)
        tokens_per_npu = per_device_bs * gas * step_diff * max_len
        throughput = tokens_per_npu / time_diff
        return f"{throughput:.2f} tokens/s/npu"

    def _compute_total_throughput(self, args, state, runtime):
        if runtime is None or runtime <= 0 or state.global_step <= 0:
            return None
        per_device_bs = getattr(args, "per_device_train_batch_size", 1)
        gas = getattr(args, "gradient_accumulation_steps", 1)
        max_len = getattr(args, "model_max_length", 4096)
        tokens_per_npu = per_device_bs * gas * state.global_step * max_len
        throughput = tokens_per_npu / runtime
        return f"{throughput:.2f} tokens/s/npu"

    def _reset_throughput_window(self, state):
        self._throughput_start_time = time.time()
        self._throughput_start_step = state.global_step

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._is_rank0(args, state):
            return
        self._touch_log_file(self.train_log_path)
        self._touch_log_file(self.eval_log_path)
        self._touch_log_file(self.checkpoint_log_path)
        self._train_start_time = time.time()
        self._reset_throughput_window(state)
        payload = {
            "event": "train_begin",
            "time": time.time(),
            "global_step": state.global_step,
            "max_steps": state.max_steps,
            "num_train_epochs": getattr(args, "num_train_epochs", None),
            "logging_steps": getattr(args, "logging_steps", None),
            "save_steps": getattr(args, "save_steps", None),
            "eval_steps": getattr(args, "eval_steps", None),
        }
        self._append_log_line(self.checkpoint_log_path, payload)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self._is_rank0(args, state) or not logs:
            return

        if "train_runtime" in logs:
            throughput_str = self._compute_total_throughput(args, state, logs.get("train_runtime"))
            self._saw_train_runtime_log = True
        else:
            throughput_str = self._compute_throughput(args, state)
            self._reset_throughput_window(state)

        payload = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "global_step": state.global_step,
            "epoch": state.epoch,
            **logs,
        }
        if throughput_str:
            payload["DI_throughput"] = throughput_str

        is_train_runtime_summary = "train_runtime" in logs
        if "eval_loss" in logs or any(key.startswith("eval_") for key in logs):
            line = self._append_log_line(self.eval_log_path, payload)
        else:
            line = self._append_log_line(self.train_log_path, payload)
        if not is_train_runtime_summary:
            if getattr(args, "use_hf_progress_bar", False):
                if throughput_str:
                    tqdm_auto.write(line)
            else:
                print(line)

    def on_save(self, args, state, control, **kwargs):
        if not self._is_rank0(args, state):
            return
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        payload = {
            "event": "save",
            "time": time.time(),
            "global_step": state.global_step,
            "epoch": state.epoch,
            "checkpoint_dir": checkpoint_dir,
        }
        self._append_log_line(self.checkpoint_log_path, payload)

    def on_train_end(self, args, state, control, **kwargs):
        if not self._is_rank0(args, state):
            return
        train_runtime = None
        if self._train_start_time is not None:
            train_runtime = time.time() - self._train_start_time
        throughput_str = self._compute_total_throughput(args, state, train_runtime)
        if not self._saw_train_runtime_log and (train_runtime is not None or throughput_str):
            train_payload = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "global_step": state.global_step,
                "epoch": state.epoch,
                "train_runtime": train_runtime,
                "DI_throughput": throughput_str,
            }
            self._append_log_line(self.train_log_path, train_payload)
        payload = {
            "event": "train_end",
            "time": time.time(),
            "global_step": state.global_step,
            "epoch": state.epoch,
            "best_model_checkpoint": state.best_model_checkpoint,
            "best_metric": state.best_metric,
        }
        self._append_log_line(self.checkpoint_log_path, payload)


def _copy_checkpoint_tree(src_dir: str, dst_dir: str):
    def copy_or_link(src, dst):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    shutil.copytree(src_dir, dst_dir, symlinks=True, copy_function=copy_or_link)


def _best_checkpoint_save_mode(args) -> str:
    mode = getattr(args, "best_checkpoint_save_mode", "rotating_create_only") or "rotating_create_only"
    return str(mode).strip().lower().replace("-", "_")


def _is_rotating_create_only_best_mode(args) -> bool:
    return _best_checkpoint_save_mode(args) in {"rotating_create_only", "rotate_create_only", "rotating"}


def _metric_for_path(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def _next_available_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    parent = os.path.dirname(path)
    stem, suffix = os.path.splitext(os.path.basename(path))
    for idx in range(1, 10000):
        candidate = os.path.join(parent, f"{stem}_{idx}{suffix}")
        if not os.path.exists(candidate):
            return candidate
    raise RuntimeError(f"Could not find an available create-only path for {path}")


def _best_candidate_root(args, configured_best_dir: str) -> str:
    best_dir = configured_best_dir.rstrip(os.sep)
    if os.path.isabs(best_dir):
        parent = os.path.dirname(best_dir)
        stem = os.path.basename(best_dir)
    else:
        parent = args.output_dir
        stem = best_dir
    return os.path.join(parent, f"{stem}_candidates")


def _rotating_best_dir(args, configured_best_dir: str, metric_label: str, metric_value: float, step: int) -> str:
    root = _best_candidate_root(args, configured_best_dir)
    stem = os.path.basename(configured_best_dir.rstrip(os.sep))
    target = os.path.join(root, f"{stem}_step-{step:08d}_{metric_label}-{_metric_for_path(metric_value)}")
    return _next_available_path(target)


def _write_new_json(path: str, payload: dict) -> str:
    path = _next_available_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _step_from_best_name(name: str) -> int:
    marker = "_step-"
    if marker not in name:
        return -1
    tail = name.split(marker, 1)[1]
    digits = []
    for char in tail:
        if not char.isdigit():
            break
        digits.append(char)
    return int("".join(digits)) if digits else -1


def _successful_best_candidates(candidate_root: str, metadata_filename: str, metric_key: str, step_key: str):
    if not os.path.isdir(candidate_root):
        return []
    candidates = []
    for name in os.listdir(candidate_root):
        path = os.path.join(candidate_root, name)
        if not os.path.isdir(path) or not os.path.isfile(os.path.join(path, "_SUCCESS")):
            continue
        metadata = {}
        metadata_path = os.path.join(path, metadata_filename)
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {}
        step = int(metadata.get(step_key, _step_from_best_name(name)) or -1)
        metric = metadata.get(metric_key)
        candidates.append({"path": path, "step": step, "metric": metric, "metadata": metadata})
    return sorted(candidates, key=lambda item: (item["step"], item["path"]))


def _load_rotating_best_loss(args, configured_best_dir: str, metadata_filename: str, metric_key: str, step_key: str):
    candidate_root = _best_candidate_root(args, configured_best_dir)
    candidates = _successful_best_candidates(candidate_root, metadata_filename, metric_key, step_key)
    valid = []
    for item in candidates:
        try:
            valid.append((float(item["metric"]), item))
        except (TypeError, ValueError):
            continue
    if not valid:
        return None, None
    return min(valid, key=lambda pair: pair[0])


def _best_checkpoint_keep_limit(args) -> int:
    try:
        return int(getattr(args, "best_checkpoint_keep_limit", 1) or 0)
    except (TypeError, ValueError):
        return 1


def _rotate_best_candidates(
    args,
    candidate_root: str,
    metadata_filename: str,
    metric_key: str,
    step_key: str,
    protected_dir: str,
):
    keep_limit = _best_checkpoint_keep_limit(args)
    if keep_limit <= 0:
        return
    protected_dir = os.path.abspath(protected_dir)
    candidates = _successful_best_candidates(candidate_root, metadata_filename, metric_key, step_key)
    stale = candidates[:-keep_limit]
    for item in stale:
        path = os.path.abspath(item["path"])
        if path == protected_dir:
            continue
        try:
            shutil.rmtree(path)
        except Exception as exc:
            rank0_print(f"[WARN] Failed to delete old best checkpoint candidate {path}: {exc}")


def _write_success_marker(target_dir: str, metadata: dict):
    success_path = os.path.join(target_dir, "_SUCCESS")
    with open(success_path, "x", encoding="utf-8") as f:
        f.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def _save_rotating_best_checkpoint(
    args,
    checkpoint_dir: str,
    target_dir: str,
    metadata: dict,
    metadata_filename: str,
    metric_key: str,
    step_key: str,
):
    metadata = dict(metadata)
    metadata["best_checkpoint"] = target_dir
    metadata["source_checkpoint"] = checkpoint_dir
    metadata["best_checkpoint_save_mode"] = "rotating_create_only"
    metadata["best_checkpoint_keep_limit"] = _best_checkpoint_keep_limit(args)
    _copy_checkpoint_tree(checkpoint_dir, target_dir)
    _write_new_json(os.path.join(target_dir, metadata_filename), metadata)
    _write_success_marker(target_dir, metadata)
    _rotate_best_candidates(
        args,
        os.path.dirname(target_dir),
        metadata_filename,
        metric_key,
        step_key,
        target_dir,
    )
    return metadata


class BestTrainLossCallback(TrainerCallback):
    def __init__(self):
        self.best_loss = None
        self.pending_best = None

    def _is_rank0(self, args, state=None) -> bool:
        if state is not None and hasattr(state, "is_world_process_zero"):
            return state.is_world_process_zero
        rank = os.environ.get("RANK")
        if rank is not None:
            return int(rank) == 0
        return args.local_rank in (-1, 0)

    def _enabled(self, args) -> bool:
        return bool(getattr(args, "save_best_train_loss", False))

    def _best_dir(self, args):
        best_dir = getattr(args, "best_train_loss_dir", "best")
        if os.path.isabs(best_dir):
            return best_dir
        return os.path.join(args.output_dir, best_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._enabled(args):
            return
        if _is_rotating_create_only_best_mode(args):
            loaded = _load_rotating_best_loss(
                args,
                self._best_dir(args),
                "best_train_loss.json",
                "best_train_loss",
                "best_train_loss_step",
            )
            if loaded[0] is not None:
                self.best_loss = float(loaded[0])
                if self._is_rank0(args, state):
                    rank0_print(
                        f"Loaded existing rotating best train loss: {self.best_loss:.6g} "
                        f"from {loaded[1]['path']}"
                    )
            return
        metadata_path = os.path.join(self._best_dir(args), "best_train_loss.json")
        if not os.path.isfile(metadata_path):
            return
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.best_loss = float(payload["best_train_loss"])
            if self._is_rank0(args, state):
                rank0_print(f"Loaded existing best train loss: {self.best_loss:.6g}")
        except Exception as exc:
            if self._is_rank0(args, state):
                rank0_print(f"[WARN] Failed to read existing best train loss metadata: {exc}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self._enabled(args) or not logs:
            return control
        if "loss" not in logs or "train_runtime" in logs:
            return control
        start_step = int(getattr(args, "best_train_loss_start_step", 0) or 0)
        if state.global_step < start_step:
            return control

        try:
            loss = float(logs["loss"])
        except (TypeError, ValueError):
            return control
        if self.best_loss is not None and loss >= self.best_loss:
            return control

        self.best_loss = loss
        self.pending_best = {
            "best_train_loss": loss,
            "best_train_loss_step": state.global_step,
            "best_checkpoint": f"checkpoint-{state.global_step}",
            "start_step": start_step,
            "epoch": state.epoch,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        control.should_save = True
        if self._is_rank0(args, state):
            rank0_print(
                f"New best train loss after step {start_step}: "
                f"{loss:.6g} at step {state.global_step}; saving checkpoint."
            )
        return control

    def on_save(self, args, state, control, **kwargs):
        if not self._enabled(args) or not self._is_rank0(args, state):
            return control
        if not self.pending_best or self.pending_best.get("best_train_loss_step") != state.global_step:
            return control

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(checkpoint_dir):
            rank0_print(f"[WARN] Best train loss checkpoint not found after save: {checkpoint_dir}")
            return control

        configured_best_dir = self._best_dir(args)
        if _is_rotating_create_only_best_mode(args):
            metadata = dict(self.pending_best)
            metadata["best_checkpoint"] = checkpoint_dir
            best_dir = _rotating_best_dir(
                args,
                configured_best_dir,
                "loss",
                metadata["best_train_loss"],
                state.global_step,
            )
            metadata = _save_rotating_best_checkpoint(
                args,
                checkpoint_dir,
                best_dir,
                metadata,
                "best_train_loss.json",
                "best_train_loss",
                "best_train_loss_step",
            )
            rank0_print(
                f"Created best train loss checkpoint: {best_dir} "
                f"(loss={metadata['best_train_loss']:.6g}, step={state.global_step}, "
                f"keep={metadata['best_checkpoint_keep_limit']})"
            )
            self.pending_best = None
            return control

        best_dir = configured_best_dir
        tmp_dir = f"{best_dir}.tmp"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        _copy_checkpoint_tree(checkpoint_dir, tmp_dir)

        metadata = dict(self.pending_best)
        metadata["best_checkpoint"] = checkpoint_dir
        with open(os.path.join(tmp_dir, "best_train_loss.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        os.replace(tmp_dir, best_dir)
        rank0_print(
            f"Updated best train loss checkpoint: {best_dir} "
            f"(loss={metadata['best_train_loss']:.6g}, step={state.global_step})"
        )
        self.pending_best = None
        return control


class BestEvalLossCallback(TrainerCallback):
    def __init__(self):
        self.best_loss = None
        self.pending_best = None

    def _is_rank0(self, args, state=None) -> bool:
        if state is not None and hasattr(state, "is_world_process_zero"):
            return state.is_world_process_zero
        rank = os.environ.get("RANK")
        if rank is not None:
            return int(rank) == 0
        return args.local_rank in (-1, 0)

    def _enabled(self, args) -> bool:
        return bool(getattr(args, "save_best_eval_loss", False))

    def _best_dir(self, args):
        best_dir = getattr(args, "best_eval_loss_dir", "eval_best")
        if os.path.isabs(best_dir):
            return best_dir
        return os.path.join(args.output_dir, best_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._enabled(args):
            return
        if _is_rotating_create_only_best_mode(args):
            loaded = _load_rotating_best_loss(
                args,
                self._best_dir(args),
                "best_eval_loss.json",
                "best_eval_loss",
                "best_eval_loss_step",
            )
            if loaded[0] is not None:
                self.best_loss = float(loaded[0])
                if self._is_rank0(args, state):
                    rank0_print(
                        f"Loaded existing rotating best eval loss: {self.best_loss:.6g} "
                        f"from {loaded[1]['path']}"
                    )
            return
        metadata_path = os.path.join(self._best_dir(args), "best_eval_loss.json")
        if not os.path.isfile(metadata_path):
            return
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.best_loss = float(payload["best_eval_loss"])
            if self._is_rank0(args, state):
                rank0_print(f"Loaded existing best eval loss: {self.best_loss:.6g}")
        except Exception as exc:
            if self._is_rank0(args, state):
                rank0_print(f"[WARN] Failed to read existing best eval loss metadata: {exc}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self._enabled(args) or not metrics:
            return control
        if "eval_loss" not in metrics:
            return control

        try:
            loss = float(metrics["eval_loss"])
        except (TypeError, ValueError):
            return control
        if self.best_loss is not None and loss >= self.best_loss:
            return control

        self.best_loss = loss
        self.pending_best = {
            "best_eval_loss": loss,
            "best_eval_loss_step": state.global_step,
            "best_checkpoint": f"checkpoint-{state.global_step}",
            "epoch": state.epoch,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        control.should_save = True
        if self._is_rank0(args, state):
            rank0_print(
                f"New best eval loss: {loss:.6g} at step {state.global_step}; saving checkpoint."
            )
        return control

    def on_save(self, args, state, control, **kwargs):
        if not self._enabled(args) or not self._is_rank0(args, state):
            return control
        if not self.pending_best or self.pending_best.get("best_eval_loss_step") != state.global_step:
            return control

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(checkpoint_dir):
            rank0_print(f"[WARN] Best eval loss checkpoint not found after save: {checkpoint_dir}")
            return control

        configured_best_dir = self._best_dir(args)
        if _is_rotating_create_only_best_mode(args):
            metadata = dict(self.pending_best)
            metadata["best_checkpoint"] = checkpoint_dir
            best_dir = _rotating_best_dir(
                args,
                configured_best_dir,
                "loss",
                metadata["best_eval_loss"],
                state.global_step,
            )
            metadata = _save_rotating_best_checkpoint(
                args,
                checkpoint_dir,
                best_dir,
                metadata,
                "best_eval_loss.json",
                "best_eval_loss",
                "best_eval_loss_step",
            )
            rank0_print(
                f"Created best eval loss checkpoint: {best_dir} "
                f"(loss={metadata['best_eval_loss']:.6g}, step={state.global_step}, "
                f"keep={metadata['best_checkpoint_keep_limit']})"
            )
            self.pending_best = None
            return control

        best_dir = configured_best_dir
        tmp_dir = f"{best_dir}.tmp"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        _copy_checkpoint_tree(checkpoint_dir, tmp_dir)

        metadata = dict(self.pending_best)
        metadata["best_checkpoint"] = checkpoint_dir
        with open(os.path.join(tmp_dir, "best_eval_loss.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        os.replace(tmp_dir, best_dir)
        rank0_print(
            f"Updated best eval loss checkpoint: {best_dir} "
            f"(loss={metadata['best_eval_loss']:.6g}, step={state.global_step})"
        )
        self.pending_best = None
        return control


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_mlp_and_vision_tower: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    unfreeze_mm_vision_tower: bool = field(default=False)
    deepstack_visual_indexes: Optional[List[int]] = field(default=None)
    disable_deepstack: bool = field(default=True)
    input_image_size: Optional[int] = field(default=None)
    mm_vision_tower_type: Optional[str] = field(default=None)
    s2: Optional[bool] = field(default=False)
    hd: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_path: Optional[List[str]] = field(default=None,
                           metadata={"help": "Optional list of paths to the training data."})
    eval_data_path: Optional[List[str]] = field(default=None,
                           metadata={"help": "Optional list of paths to the eval data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[List[str]] = field(default=None)
    eval_image_folder: Optional[List[str]] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)
    train_sample_limit: Optional[int] = field(default=None)
    eval_sample_limit: Optional[int] = field(default=None)
    sample_seed: int = field(default=42)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_target_scope: str = "llm"
    lora_target_modules: Optional[str] = None
    lora_exclude_modules: Optional[str] = "lm_head,embed_tokens"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    mm_vision_tower_lr: Optional[float] = None
    save_best_train_loss: bool = field(default=False)
    best_train_loss_start_step: int = field(default=0)
    best_train_loss_dir: str = field(default="best")
    save_best_eval_loss: bool = field(default=False)
    best_eval_loss_dir: str = field(default="eval_best")
    best_checkpoint_save_mode: str = field(
        default="rotating_create_only",
        metadata={"help": "How best checkpoints are materialized: rotating_create_only or replace."},
    )
    best_checkpoint_keep_limit: int = field(
        default=1,
        metadata={"help": "How many successful rotating best checkpoint candidates to keep."},
    )
    use_hf_progress_bar: bool = field(default=False)
    swanlab_enable: bool = field(default=False)
    swanlab_project: Optional[str] = field(default=None)
    swanlab_experiment_name: Optional[str] = field(default=None)
    swanlab_description: Optional[str] = field(default=None)
    swanlab_tags: Optional[str] = field(default=None)
    swanlab_mode: Optional[str] = field(default=None)
    swanlab_log_dir: Optional[str] = field(default=None)
    swanlab_api_host: Optional[str] = field(default=None)
    swanlab_web_host: Optional[str] = field(default=None)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def _split_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).replace(";", ",").split(",")
    return [str(item).strip() for item in items if str(item).strip()]


def _module_in_scope(name: str, scopes: set[str]) -> bool:
    if "all" in scopes:
        return True
    is_projector = "mm_projector" in name
    is_vision = "vision_tower" in name
    is_deepstack = "deepstack" in name or "deepstack_mergers" in name
    if "projector" in scopes and is_projector:
        return True
    if "vision" in scopes and is_vision and not is_deepstack:
        return True
    if "deepstack" in scopes and is_deepstack:
        return True
    if "llm" in scopes and not (is_projector or is_vision or is_deepstack):
        return True
    return False


def resolve_lora_target_modules(
    model,
    target_scope: str = "llm",
    target_modules: Optional[str] = None,
    exclude_modules: Optional[str] = None,
) -> List[str]:
    manual_targets = _split_csv(target_modules)
    excludes = _split_csv(exclude_modules)
    if manual_targets:
        targets = manual_targets
    else:
        scopes = set(_split_csv(target_scope) or ["llm"])
        targets = []
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not _module_in_scope(name, scopes):
                continue
            targets.append(name)

    filtered = []
    for name in targets:
        if any(exclude and exclude in name for exclude in excludes):
            continue
        filtered.append(name)
    return sorted(set(filtered))


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.is_world_process_zero():
            sync_qwen_multimodal_config(trainer.model)
            trainer.model.config.save_pretrained(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
            write_qwen_multimodal_checkpoint_metadata(trainer.model, output_dir, trainer)
        return

    if trainer.deepspeed:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch, "npu"):
           torch.npu.synchronize() 
        trainer.save_model(output_dir)
        write_qwen_multimodal_checkpoint_metadata(trainer.model, output_dir, trainer)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        write_qwen_multimodal_checkpoint_metadata(trainer.model, output_dir, trainer)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


# fix: add qwen2
# def preprocess_qwen_2(
#     sources,
#     tokenizer: transformers.PreTrainedTokenizer,
#     has_image: bool = False
# ) -> Dict:
#     # print('-----preprocess_qwen_2-------')
#     conv = conversation_lib.default_conversation.copy()
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
#
#     # Apply prompt templates
#     conversations = []
#     for i, source in enumerate(sources):
#         if roles[source[0]["from"]] != conv.roles[0]:
#             # Skip the first one if it is not from human
#             source = source[1:]
#
#         conv.messages = []
#         for j, sentence in enumerate(source):
#             role = roles[sentence["from"]]
#             assert role == conv.roles[j % 2], f"{i}"
#             conv.append_message(role, sentence["value"])
#         conversations.append(conv.get_prompt())
#
#     # Tokenize conversations
#
#     if has_image:
#         input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
#     else:
#         input_ids = tokenizer(
#             conversations,
#             return_tensors="pt",
#             padding="longest",
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#         ).input_ids
#
#     targets = input_ids.clone()
#
#     assert conv.sep_style == conversation_lib.SeparatorStyle.QWEN_2
#
#     rank0_print(50*'S')
#     # Mask targets
#     sep = conv.sep + conv.roles[1] + ": "
#     for conversation, target in zip(conversations, targets):
#         total_len = int(target.ne(tokenizer.pad_token_id).sum())
#         rank0_print(f"target.shape={target.shape}", f"total_len={total_len}")
#
#         rounds = conversation.split(conv.sep2)
#         rounds_len = len(rounds)
#         cur_len = 0
#         # target[:cur_len] = IGNORE_INDEX
#         for i, rou in enumerate(rounds):
#             if rou == "":
#                 break
#
#             parts = rou.split(sep)
#             if len(parts) != 2:
#                 break
#             parts[0] += sep
#
#             if has_image:
#                 round_ids = tokenizer_image_token(rou, tokenizer)
#                 instruction_ids = tokenizer_image_token(parts[0], tokenizer)
#                 equal_parts = [x == y for x, y in zip(round_ids, instruction_ids)]
#
#                 instruction_len = equal_parts.index(False) if False in equal_parts else len(equal_parts)
#                 round_len = len(round_ids)
#
#             else:
#                 round_ids = tokenizer(rou).input_ids
#                 instruction_ids = tokenizer(parts[0]).input_ids
#                 equal_parts = [x == y for x, y in zip(round_ids, instruction_ids)]
#
#                 instruction_len = equal_parts.index(False) if False in equal_parts else len(equal_parts)
#                 round_len = len(round_ids)
#
#             if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
#                 round_len += 1
#                 instruction_len += 1
#
#             rank0_print(i, rou, f"cur_len={cur_len}", f"round_len={round_len}", f"instruction_len={instruction_len}", f"cur_len + instruction_len={cur_len + instruction_len}")
#             target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
#
#             cur_len += round_len
#         rank0_print("Outside Loop")
#         rank0_print(cur_len, len(target))
#         rank0_print(target)
#         rank0_print(50 * 'E')
#         exit(0)
#         target[cur_len:] = IGNORE_INDEX
#
#         if cur_len < tokenizer.model_max_length:
#             if cur_len != total_len + rounds_len - 2:
#                 target[:] = IGNORE_INDEX
#                 print(
#                     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
#                     f" (ignored)"
#                 )
#
#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#     )

def preprocess_qwen_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            try:
                role = roles[sentence["from"]]
            except KeyError as e:
                print("e")
                print("skipping sentence due to unrecognized role: {}".format(sentence["from"]))
                continue
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.QWEN_2

    split_sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds_before = conversation.split(
            conv.sep)  # system->user->assistant->user->assistant->user->assistant->user->assistant->Empty
        if rounds_before[0] == conv.system:
            rounds_before[1] = conv.sep.join([rounds_before[0], rounds_before[1]])
            rounds_before = rounds_before[1:]
        # connect every pair:
        rounds = []
        for i in range(0, len(rounds_before), 2):
            if i < len(rounds_before)-1:
                rounds.append(conv.sep.join([rounds_before[i], rounds_before[i + 1]]))
            else:
                rounds.append(rounds_before[i])
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(split_sep)
            if len(parts) != 2:
                break

            assert parts[0].startswith(conv.roles[0]) or parts[0].startswith(conv.system)
            parts[0] += split_sep

            if has_image:
                round_ids = tokenizer_image_token(rou, tokenizer)
                instruction_ids = tokenizer_image_token(parts[0], tokenizer)
                equal_parts = [x == y for x, y in zip(round_ids, instruction_ids)]

                instruction_len = equal_parts.index(False) if False in equal_parts else len(equal_parts)
                round_len = len(round_ids)

            else:
                round_ids = tokenizer(rou).input_ids
                instruction_ids = tokenizer(parts[0]).input_ids
                equal_parts = [x == y for x, y in zip(round_ids, instruction_ids)]

                instruction_len = equal_parts.index(False) if False in equal_parts else len(equal_parts)
                round_len = len(round_ids)

            round_len += 2 # this is tom compensate for the sep2

            assert rou == parts[0]+parts[1]

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # print("conversation:",conversation_lib.default_conversation.version)
    # conversation_lib.default_conversation.version == "qwen_v2"

    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        # print('--v1--')
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        # print('--mpt--')
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # fix: add qwen2/qwen3
    if conversation_lib.default_conversation.version.startswith("qwen_v2"):
        return preprocess_qwen_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("qwen_v3"):
        return preprocess_qwen_2(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations

    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def _load_json_or_jsonl(file_path: str) -> List[dict]:
    with open(file_path, 'r') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: List[str],
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 sample_limit: Optional[int] = None):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = []
        for i, _data_path in enumerate(data_path):
            data = _load_json_or_jsonl(_data_path)
            data = [{**entry, 'img_path_idx': i} for entry in data]
            list_data_dict += data

        if sample_limit is not None and sample_limit > 0 and len(list_data_dict) > sample_limit:
            rng = random.Random(data_args.sample_seed)
            sampled_indices = sorted(rng.sample(range(len(list_data_dict)), sample_limit))
            list_data_dict = [list_data_dict[idx] for idx in sampled_indices]

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def get_sample(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            img_path_idx = self.list_data_dict[i]['img_path_idx']
            image_folder = self.data_args.image_folder[img_path_idx]
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image_size = image.size
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif self.data_args.image_aspect_ratio == "anyres" or "anyres_max" in self.data_args.image_aspect_ratio:
                image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['image_size'] = image_size
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['image_size'] = (crop_size['height'], crop_size['width'])
        return data_dict

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            return self.get_sample(i)
        except Exception as e:
            print("Error loading sample")
            print()
            return self.get_sample(0)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            batch['image_sizes'] = [instance['image_size'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args,
                                          sample_limit=data_args.train_sample_limit)
    eval_dataset = None
    if data_args.eval_data_path is not None:
        eval_args = copy.deepcopy(data_args)
        eval_args.data_path = data_args.eval_data_path
        if data_args.eval_image_folder is not None:
            eval_args.image_folder = data_args.eval_image_folder
        eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                             data_path=eval_args.data_path,
                                             data_args=eval_args,
                                             sample_limit=data_args.eval_sample_limit)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)

def print_trainable_parameters(model):
    """Print trainable and total parameter storage in MiB."""
    trainable_bytes = 0
    all_bytes = 0
    for name, param in model.named_parameters():
        param_bytes = param.numel() * param.element_size()
        all_bytes += param_bytes
        if param.requires_grad:
            trainable_bytes += param_bytes
            # print(f"Trainable: {name} | {param_bytes / (1024 ** 2):.2f} MiB")
    trainable_mib = trainable_bytes / (1024 ** 2)
    all_mib = all_bytes / (1024 ** 2)
    trainable_percent = 100 * trainable_bytes / all_bytes if all_bytes else 0.0
    print(
        f"trainable params: {trainable_mib:.2f} MiB || "
        f"all params: {all_mib:.2f} MiB || "
        f"trainable%: {trainable_percent:.2f}")


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    silence_non_primary_rank_output()
    if model_args.disable_deepstack:
        model_args.deepstack_visual_indexes = None
        rank0_print("DeepStack disabled: using ViT main feature + mm_projector only.")
    if (
        training_args.gradient_checkpointing
        and training_args.deepspeed
        and training_args.gradient_checkpointing_kwargs is None
    ):
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
        rank0_print("Using reentrant gradient checkpointing with DeepSpeed.")
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    _is_qwen3 = _path_declares_qwen3(model_args.model_name_or_path)

    if is_qwen3vl_checkpoint(model_args.model_name_or_path):
        rank0_print(f"Ensuring extracted LLM cache for Qwen3-VL checkpoint: {model_args.model_name_or_path}")
        cache_path = ensure_extracted_llm_from_qwen3vl(model_args.model_name_or_path)
        rank0_print(f"Using extracted LLM cache: {cache_path}")
        model_args.model_name_or_path = cache_path
        _is_qwen3 = True

    load_multimodal_checkpoint_weights = False
    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
            _is_qwen3 = _is_qwen3 or _config_declares_qwen3(config)
            _original_vt = getattr(config, 'mm_vision_tower', None)
            if _original_vt and _original_vt != model_args.vision_tower:
                delattr(config, 'mm_vision_tower')
                rank0_print(f"checkpoint vision tower '{_original_vt}' != requested '{model_args.vision_tower}', will rebuild vision modules.")
            elif _original_vt:
                load_multimodal_checkpoint_weights = True
            if _is_qwen3:
                from mllm.model.language_model.llava_qwen3 import LlavaQwen3ConfigWrapper
                if not isinstance(config, LlavaQwen3ConfigWrapper):
                    d = config.to_dict()
                    d.pop("model_type",None)
                    config = LlavaQwen3ConfigWrapper(**d)
                model = LlavaQwen3ForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    config=config,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
            else:
                from mllm.model.language_model.llava_qwen import LlavaConfig
                if not isinstance(config, LlavaConfig):
                    d = config.to_dict()
                    d.pop("model_type",None)
                    config = LlavaConfig(**d)
                model = LlavaQwen2ForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    config=config,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
    else:
        if _is_qwen3:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
        else:
            model = transformers.Qwen2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_targets = resolve_lora_target_modules(
            model,
            target_scope=training_args.lora_target_scope,
            target_modules=training_args.lora_target_modules,
            exclude_modules=training_args.lora_exclude_modules,
        )
        if not lora_targets:
            raise ValueError(
                "No LoRA target modules were resolved. "
                f"scope={training_args.lora_target_scope!r}, "
                f"manual={training_args.lora_target_modules!r}, "
                f"exclude={training_args.lora_exclude_modules!r}"
            )
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=lora_targets,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print(
            "Adding LoRA adapters: "
            f"scope={training_args.lora_target_scope}, targets={len(lora_targets)}"
        )
        rank0_print("LoRA target modules: " + ", ".join(lora_targets[:80]))
        if len(lora_targets) > 80:
            rank0_print(f"... {len(lora_targets) - 80} more LoRA target modules omitted")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = _load_tokenizer_with_fast_fallback(
            model_args.model_name_or_path,
            use_fast=None,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            **qwen_tokenizer_kwargs(model_args.model_name_or_path),
        )
    else:
        tokenizer = _load_tokenizer_with_fast_fallback(
            model_args.model_name_or_path,
            use_fast=False,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            **qwen_tokenizer_kwargs(model_args.model_name_or_path),
        )

    sync_qwen_token_config(
        tokenizer=tokenizer,
        model=model,
        model_name_or_path=model_args.model_name_or_path,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.pad_token is None and tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.legacy = False
        if model_args.version in conversation_lib.conv_templates:
            # print('version:', model_args.version)
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    sync_qwen_token_config(
        tokenizer=tokenizer,
        model=model,
        model_name_or_path=model_args.model_name_or_path,
    )

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        # The pretrained config can carry its own unfreeze flag; enforce the CLI choice here.
        vision_tower.tune_vision_tower = model_args.unfreeze_mm_vision_tower
        vision_tower.vision_tower.requires_grad_(model_args.unfreeze_mm_vision_tower)
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        if load_multimodal_checkpoint_weights:
            _load_multimodal_weights_if_present(model, model_args.model_name_or_path)


        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        # freeze llm
        if training_args.freeze_llm:
            model.requires_grad_(False)
            for p in model.get_model().vision_tower.parameters():
                p.requires_grad = True
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter

        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        sync_qwen_token_config(
            tokenizer=tokenizer,
            model=model,
            model_name_or_path=model_args.model_name_or_path,
        )
        sync_qwen_multimodal_config(model)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
                        
    if _is_global_rank0():
            print_trainable_parameters(model)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    callbacks = [
        JsonlMetricLoggerCallback(training_args.output_dir),
        BestTrainLossCallback(),
        BestEvalLossCallback(),
    ]
    swanlab_callback = build_swanlab_callback(model_args, data_args, training_args)
    if swanlab_callback is not None:
        callbacks.append(swanlab_callback)
    trainer = LLaVATrainer(model=model,
                           processing_class=tokenizer,
                           args=training_args,
                           callbacks=callbacks,
                           **data_module)
    trainer.remove_callback(PrinterCallback)
    if not training_args.use_hf_progress_bar:
        trainer.remove_callback(ProgressCallback)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # if trainer.state.best_model_checkpoint:
    #     best_output_dir = os.path.join(training_args.output_dir, "best")
    #     if os.path.exists(best_output_dir):
    #         shutil.rmtree(best_output_dir)
    #     shutil.copytree(trainer.state.best_model_checkpoint, best_output_dir)
    #     rank0_print(f"Copied best checkpoint to {best_output_dir}")

    model.config.use_cache = True
    sync_qwen_token_config(
        tokenizer=tokenizer,
        model=model,
        model_name_or_path=model_args.model_name_or_path,
    )

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if trainer.is_world_process_zero():
            sync_qwen_multimodal_config(model)
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            tokenizer.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            write_qwen_multimodal_checkpoint_metadata(model, training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
