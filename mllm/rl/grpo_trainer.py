from __future__ import annotations

from dataclasses import asdict
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from mllm.constants import IGNORE_INDEX
from mllm.coord_utils import COORD_MODE_PIXEL, normalize_coord_mode
from mllm.train.swanlab_utils import finish_swanlab, init_swanlab_run, log_swanlab

from .config import GRPOArguments, RLDataArguments, RLModelArguments
from .dataset import RLDataCollator, RLSFTJsonlDataset
from .export import export_merged_lora_checkpoint, export_text_decoder_checkpoint
from .modeling import (
    apply_trainable_policy,
    create_optimizer,
    get_base_policy_model,
    is_peft_checkpoint,
    load_policy_model,
    print_policy_parameters,
    save_policy_checkpoint,
    unwrap_model,
)
from .reward import MapRewardScorer, RewardSettings
from .rollout import RolloutBatch, RolloutCompletion, RolloutPrompt, RolloutSample, VLLMPromptEmbedRolloutWorker


def rank0_print(*args):
    print(*args)


def _split_paths(paths):
    if paths is None:
        return []
    if isinstance(paths, str):
        return [item for item in paths.replace(";", ",").split(",") if item]
    return list(paths)


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        pass
    return hasattr(torch, "npu") and torch.npu.is_available()


def _device_from_visible() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    if _npu_available():
        torch.npu.set_device(0)
        return torch.device("npu", 0)
    return torch.device("cpu")


def _move_images_to_device(images, device: torch.device, dtype: torch.dtype):
    if isinstance(images, torch.Tensor):
        return images.to(device=device, dtype=dtype)
    return [image.to(device=device, dtype=dtype) for image in images]


def _select_images_for_samples(images, samples: list[RolloutSample], device: torch.device, dtype: torch.dtype):
    if isinstance(images, torch.Tensor):
        selected = torch.stack([images[sample.image_index] for sample in samples])
        return selected.to(device=device, dtype=dtype)
    return [images[sample.image_index].to(device=device, dtype=dtype) for sample in samples]


def _find_multimodal_preparer(model):
    unwrapped = unwrap_model(model)
    if hasattr(unwrapped, "prepare_inputs_labels_for_multimodal"):
        return unwrapped
    base = get_base_policy_model(unwrapped)
    if hasattr(base, "prepare_inputs_labels_for_multimodal"):
        return base
    raise AttributeError("Policy model does not expose prepare_inputs_labels_for_multimodal")


def _safe_metric_name(name: str) -> str:
    return str(name).strip().replace("/", "_").replace(" ", "_")


def _metric_for_path(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(1, 10000):
        candidate = path.with_name(f"{path.name}_{idx}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available create-only path for {path}")


def _is_child_path(path: Path, root: Path) -> bool:
    abs_path = os.path.abspath(os.fspath(path))
    abs_root = os.path.abspath(os.fspath(root))
    try:
        return os.path.commonpath([abs_path, abs_root]) == abs_root
    except ValueError:
        return False


def _is_prefixed_step_dir(path: Path, prefix: str) -> bool:
    name = path.name
    return name.startswith(prefix) and name[len(prefix):].isdigit()


def _safe_delete_tree(path: Path, output_dir: Path) -> bool:
    if not _is_child_path(path, output_dir):
        return False

    abs_path = Path(os.path.abspath(os.fspath(path)))
    abs_output = Path(os.path.abspath(os.fspath(output_dir)))
    parent = abs_path.parent.name
    if abs_path.parent == abs_output and _is_prefixed_step_dir(abs_path, "checkpoint-"):
        return True
    if parent == "runtime_lora" and _is_prefixed_step_dir(abs_path, "step-"):
        return True
    if parent.endswith("_candidates") and "_step-" in abs_path.name:
        return True
    return False


def _remove_tree_with_rm_rf(path: Path, output_dir: Path) -> bool:
    if not _safe_delete_tree(path, output_dir):
        rank0_print(f"[WARN] Refusing to delete unexpected GRPO path: {path}")
        return False
    if not path.exists():
        return True

    try:
        subprocess.run(
            ["rm", "-rf", "--", os.fspath(path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        rank0_print(f"[WARN] rm -rf fallback failed for GRPO path {path}: {exc}")
        return False

    if path.exists():
        rank0_print(f"[WARN] rm -rf fallback finished but path still exists: {path}")
        return False
    return True


def _best_reward_candidate_root(output_dir: Path, best_reward_dir: str) -> Path:
    stem = Path(best_reward_dir.rstrip(os.sep)).name or "best_reward"
    return output_dir / f"{stem}_candidates"


def _best_reward_candidate_dir(output_dir: Path, best_reward_dir: str, step: int, reward: float) -> Path:
    root = _best_reward_candidate_root(output_dir, best_reward_dir)
    stem = Path(best_reward_dir.rstrip(os.sep)).name or "best_reward"
    target = root / f"{stem}_step-{step:08d}_reward-{_metric_for_path(reward)}"
    return _next_available_path(target)


def _step_from_candidate_name(name: str) -> int:
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


def _successful_best_reward_candidates(candidate_root: Path) -> list[Path]:
    if not candidate_root.is_dir():
        return []
    candidates = []
    for path in candidate_root.iterdir():
        if not path.is_dir() or not (path / "_SUCCESS").is_file():
            continue
        step = _step_from_candidate_name(path.name)
        metadata_path = path / "best_reward.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                step = int(metadata.get("best_reward_step", metadata.get("global_step", step)))
            except Exception:
                pass
        candidates.append((step, path))
    return [path for _, path in sorted(candidates, key=lambda item: (item[0], str(item[1])))]


def _best_reward_keep_limit(args) -> int:
    try:
        return int(getattr(args, "best_checkpoint_keep_limit", 1) or 0)
    except (TypeError, ValueError):
        return 1


def _rotate_best_reward_candidates(args, candidate_root: Path, protected_dir: Path) -> None:
    keep_limit = _best_reward_keep_limit(args)
    if keep_limit <= 0:
        return
    candidates = _successful_best_reward_candidates(candidate_root)
    for path in candidates[:-keep_limit]:
        if os.path.abspath(os.fspath(path)) == os.path.abspath(os.fspath(protected_dir)):
            continue
        _remove_tree_with_rm_rf(path, Path(args.output_dir))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _numeric_mapping_means(payloads: list[dict[str, Any]], field: str, prefix: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for payload in payloads:
        mapping = payload.get(field)
        if not isinstance(mapping, dict):
            continue
        for key, value in mapping.items():
            if isinstance(value, bool):
                value = float(value)
            if isinstance(value, (int, float)):
                buckets.setdefault(_safe_metric_name(key), []).append(float(value))
    return {f"{prefix}/{key}": _mean(values) for key, values in sorted(buckets.items())}


def _aggregate_reward_metrics(
    rewards_payload: list[dict[str, Any]],
    rewards: list[float],
    samples: list[RolloutSample],
    group_count: int,
    num_generations: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if rewards_payload:
        metrics["reward/parse_ok_rate"] = _mean([1.0 if item.get("parse_ok") else 0.0 for item in rewards_payload])
        metrics.update(_numeric_mapping_means(rewards_payload, "components", "reward_component"))
        metrics.update(_numeric_mapping_means(rewards_payload, "counts", "reward_count"))

    if samples:
        lengths = [float(sample.completion_ids.numel()) for sample in samples]
        metrics["rollout/completion_tokens_mean"] = _mean(lengths)
        metrics["rollout/completion_tokens_max"] = float(max(lengths))

    if rewards and group_count > 0 and num_generations > 0:
        grouped = [rewards[idx: idx + num_generations] for idx in range(0, len(rewards), num_generations)]
        stds = []
        for group in grouped[:group_count]:
            if len(group) != num_generations:
                continue
            mean = _mean(group)
            variance = _mean([(value - mean) ** 2 for value in group])
            stds.append(math.sqrt(variance))
        if stds:
            metrics["reward/group_std_mean"] = _mean(stds)
            metrics["reward/group_zero_std_rate"] = _mean([1.0 if value <= 1e-12 else 0.0 for value in stds])
    return metrics


class ActorWorker:
    """GRPO actor role.

    The actor owns the trainable HF policy and computes project-specific
    multimodal prompt embeddings. It does not generate completions locally; that
    is delegated to the vLLM rollout role.
    """

    def __init__(self, model_args: RLModelArguments, data_args: RLDataArguments, train_args: GRPOArguments):
        _set_seed(train_args.seed)
        self.model_args = model_args
        self.data_args = data_args
        self.args = train_args
        self.device = _device_from_visible()
        self.global_step = 0
        self.best_reward: float | None = None
        self.pending: dict[str, Any] = {}
        self.rollout_counter = itertools.count(1)

        if not model_args.disable_deepstack:
            raise ValueError(
                "vLLM prompt-embed GRPO currently requires --disable_deepstack True. "
                "DeepStack residual injection is not representable as text-decoder prompt embeddings."
            )
        if train_args.rollout_backend != "vllm_prompt_embeds":
            raise ValueError("Formal GRPO training only supports --rollout_backend vllm_prompt_embeds.")

        tokenizer, model, image_processor = load_policy_model(model_args, train_args, str(self.device))
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model = apply_trainable_policy(model, train_args).to(self.device)
        if train_args.gradient_checkpointing:
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
        print_policy_parameters(self.model)

        self.optimizer = create_optimizer(self.model, train_args)
        coord_mode = normalize_coord_mode(data_args.coord_mode)
        default_coord_mode = COORD_MODE_PIXEL if coord_mode == "auto" else coord_mode
        data_paths = _split_paths(data_args.data_path)
        image_folders = _split_paths(data_args.image_folder)
        if not data_paths:
            raise ValueError("--data_path is required for GRPO training")
        if not image_folders:
            raise ValueError("--image_folder is required for GRPO training")

        dataset = RLSFTJsonlDataset(
            data_paths=data_paths,
            image_folders=image_folders,
            tokenizer=tokenizer,
            image_processor=image_processor,
            conv_template=model_args.version,
            image_aspect_ratio=data_args.image_aspect_ratio,
            image_grid_pinpoints=data_args.image_grid_pinpoints,
            sample_limit=data_args.train_sample_limit,
            sample_seed=data_args.sample_seed,
            default_coord_mode=default_coord_mode,
            default_patch_size=data_args.patch_size,
            default_coord_range=data_args.coord_range,
        )
        if len(dataset) == 0:
            raise ValueError("RL training dataset is empty.")
        self.dataloader = DataLoader(
            dataset,
            batch_size=train_args.per_device_train_batch_size,
            shuffle=True,
            num_workers=train_args.dataloader_num_workers,
            collate_fn=RLDataCollator(tokenizer),
            pin_memory=torch.cuda.is_available(),
        )
        self._data_iter = iter(self.dataloader)
        self.total_steps = train_args.max_steps if train_args.max_steps > 0 else math.ceil(len(self.dataloader) * train_args.num_train_epochs)
        warmup_steps = int(self.total_steps * max(train_args.warmup_ratio, 0.0))
        self.scheduler = get_scheduler(
            train_args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )

        out = Path(train_args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.metrics_path = out / "grpo_metrics.jsonl"
        self.rollout_path = out / "grpo_rollouts.jsonl"

    def _next_batch(self):
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.dataloader)
            return next(self._data_iter)

    @torch.no_grad()
    def prepare_rollout_batch(self) -> dict[str, Any]:
        self.model.eval()
        batch = self._next_batch()
        dtype = next(unwrap_model(self.model).parameters()).dtype
        batch.images = _move_images_to_device(batch.images, self.device, dtype)
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        preparer = _find_multimodal_preparer(self.model)
        (
            _,
            _,
            expanded_attention_mask,
            _,
            inputs_embeds,
            _,
            visual_pos_mask,
            deepstack_visual_embeds,
        ) = preparer.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=None,
            images=batch.images,
            image_sizes=batch.image_sizes,
        )
        if deepstack_visual_embeds is not None or visual_pos_mask is not None:
            raise ValueError("DeepStack prompt embeddings are not supported by vLLM text-decoder rollout.")
        if expanded_attention_mask is None:
            expanded_attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)

        rollout_id = f"rollout-{next(self.rollout_counter)}"
        prompts: list[RolloutPrompt] = []
        for idx, sample_id in enumerate(batch.sample_ids):
            prompt_ids = batch.input_ids[idx][batch.attention_mask[idx].bool()].detach().cpu().tolist()
            embed = inputs_embeds[idx][expanded_attention_mask[idx].bool()].detach().to("cpu")
            prompts.append(
                RolloutPrompt(
                    group_index=idx,
                    sample_id=sample_id,
                    prompt_ids=[int(token) for token in prompt_ids],
                    prompt_embeds=embed,
                    ground_truth=batch.ground_truths[idx],
                    coord_config=batch.coord_configs[idx],
                    image_index=idx,
                )
            )
        self.pending[rollout_id] = batch
        adapter_dir, adapter_id = self.save_rollout_adapter()
        return {
            "rollout_id": rollout_id,
            "prompts": prompts,
            "lora_adapter_path": adapter_dir,
            "lora_int_id": adapter_id,
        }

    def save_rollout_adapter(self) -> tuple[str | None, int | None]:
        if not self.args.lora_enable:
            return None, None
        adapter_id = self.global_step + 1
        adapter_dir = Path(self.args.output_dir) / "runtime_lora" / f"step-{self.global_step:08d}"
        if adapter_dir.exists():
            _remove_tree_with_rm_rf(adapter_dir, Path(self.args.output_dir))
        save_policy_checkpoint(self.model, self.tokenizer, adapter_dir, self.args)
        return str(adapter_dir), adapter_id

    def _sequence_batch(self, batch, samples: list[RolloutSample]) -> dict[str, Any]:
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else pad_id
        input_rows = []
        label_rows = []
        for sample in samples:
            completion = sample.completion_ids
            if completion.numel() == 0:
                completion = torch.tensor([eos_id], dtype=torch.long)
            seq = torch.cat([sample.prompt_ids.long(), completion.long()], dim=0)
            labels = torch.full_like(seq, IGNORE_INDEX)
            labels[sample.prompt_ids.numel():] = seq[sample.prompt_ids.numel():]
            if seq.numel() > self.args.model_max_length:
                seq = seq[: self.args.model_max_length]
                labels = labels[: self.args.model_max_length]
            input_rows.append(seq)
            label_rows.append(labels)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_rows, batch_first=True, padding_value=pad_id).to(self.device)
        labels = torch.nn.utils.rnn.pad_sequence(label_rows, batch_first=True, padding_value=IGNORE_INDEX).to(self.device)
        attention_mask = input_ids.ne(pad_id)
        dtype = next(unwrap_model(self.model).parameters()).dtype
        images = _select_images_for_samples(batch.images, samples, self.device, dtype)
        image_sizes = [batch.image_sizes[sample.image_index] for sample in samples]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "images": images,
            "image_sizes": image_sizes,
        }

    def _expanded_logprobs(self, model, seq_batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        preparer = _find_multimodal_preparer(model)
        (
            _,
            position_ids,
            expanded_attention_mask,
            _,
            inputs_embeds,
            expanded_labels,
            visual_pos_mask,
            deepstack_visual_embeds,
        ) = preparer.prepare_inputs_labels_for_multimodal(
            input_ids=seq_batch["input_ids"],
            position_ids=None,
            attention_mask=seq_batch["attention_mask"],
            past_key_values=None,
            labels=seq_batch["labels"],
            images=seq_batch["images"],
            image_sizes=seq_batch["image_sizes"],
        )
        outputs = model(
            input_ids=None,
            attention_mask=expanded_attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=False,
            visual_pos_mask=visual_pos_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        logits = outputs.logits[:, :-1, :].float()
        targets = expanded_labels[:, 1:]
        mask = targets.ne(IGNORE_INDEX)
        safe_targets = targets.masked_fill(~mask, 0)
        logprobs = F.log_softmax(logits, dim=-1).gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        return logprobs, mask

    @torch.no_grad()
    def _reference_logprobs(self, seq_batch: dict[str, Any]) -> torch.Tensor | None:
        if self.args.kl_beta <= 0:
            return None
        policy = unwrap_model(self.model)
        if not hasattr(policy, "disable_adapter"):
            raise ValueError("KL_BETA > 0 currently requires LoRA policy so reference can disable adapters.")
        with policy.disable_adapter():
            ref_logps, _ = self._expanded_logprobs(policy, seq_batch)
        return ref_logps.detach()

    def _advantages(self, rewards: list[float], group_count: int) -> torch.Tensor:
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        grouped = reward_tensor.view(group_count, self.args.num_generations)
        mean = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, keepdim=True, unbiased=False)
        advantages = (grouped - mean) / (std + self.args.advantage_epsilon)
        return advantages.view(-1).detach()

    def _loss_from_logprobs(
        self,
        current_logps: torch.Tensor,
        old_logps: torch.Tensor,
        ref_logps: torch.Tensor | None,
        mask: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        adv = advantages[:, None].to(current_logps.device)
        token_mask = mask.float()
        ratio = torch.exp((current_logps - old_logps).clamp(min=-20, max=20))
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - self.args.clip_range, 1.0 + self.args.clip_range) * adv
        pg_loss = -torch.minimum(unclipped, clipped)
        approx_kl = torch.zeros_like(current_logps)
        loss_tokens = pg_loss
        if ref_logps is not None and self.args.kl_beta > 0:
            ref_delta = (ref_logps - current_logps).clamp(min=-20, max=20)
            approx_kl = torch.exp(ref_delta) - ref_delta - 1.0
            loss_tokens = loss_tokens + self.args.kl_beta * approx_kl
        denom = token_mask.sum().clamp_min(1.0)
        loss = (loss_tokens * token_mask).sum() / denom
        return loss, {
            "policy_loss": ((pg_loss * token_mask).sum() / denom).detach().item(),
            "approx_kl": ((approx_kl * token_mask).sum() / denom).detach().item(),
            "clip_fraction": (((ratio - 1.0).abs() > self.args.clip_range).float() * token_mask).sum().div(denom).detach().item(),
            "action_tokens": denom.detach().item(),
        }

    def _samples_from_completions(self, batch, completions: list[RolloutCompletion]) -> list[RolloutSample]:
        samples: list[RolloutSample] = []
        by_group = {prompt_idx: [] for prompt_idx in range(len(batch.sample_ids))}
        for completion in completions:
            by_group.setdefault(completion.group_index, []).append(completion)
        ordered = []
        for group_idx in range(len(batch.sample_ids)):
            group = sorted(by_group.get(group_idx, []), key=lambda item: item.completion_index)
            if len(group) != self.args.num_generations:
                raise ValueError(
                    f"Expected {self.args.num_generations} completions for group {group_idx}, got {len(group)}"
                )
            ordered.extend(group)
        for completion in ordered:
            prompt_ids = batch.input_ids[completion.group_index][batch.attention_mask[completion.group_index].bool()].detach().cpu()
            samples.append(
                RolloutSample(
                    group_index=completion.group_index,
                    sample_id=completion.sample_id,
                    prompt_ids=prompt_ids,
                    completion_ids=torch.tensor(completion.token_ids, dtype=torch.long),
                    text=completion.text,
                    ground_truth=batch.ground_truths[completion.group_index],
                    coord_config=batch.coord_configs[completion.group_index],
                    image_index=completion.group_index,
                )
            )
        return samples

    def _write_jsonl(self, path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _save(self, name: str, metrics: dict[str, Any]) -> None:
        ckpt_dir = Path(self.args.output_dir) / name
        state = {
            "global_step": self.global_step,
            "best_reward": self.best_reward,
            "metrics": metrics,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_policy_checkpoint(self.model, self.tokenizer, ckpt_dir, self.args, self.optimizer, self.scheduler, state)
        self._rotate_checkpoints()

    def _rotate_checkpoints(self) -> None:
        limit = self.args.save_total_limit
        if not limit:
            return
        out = Path(self.args.output_dir)
        checkpoints = sorted(
            [path for path in out.glob("checkpoint-*") if path.is_dir()],
            key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else -1,
        )
        while len(checkpoints) > limit:
            _remove_tree_with_rm_rf(checkpoints.pop(0), out)

    def _maybe_save_best(self, mean_reward: float, metrics: dict[str, Any]) -> None:
        if not self.args.save_best_reward:
            return
        if self.best_reward is not None and mean_reward <= self.best_reward:
            return
        self.best_reward = mean_reward
        output_dir = Path(self.args.output_dir)
        best_dir = _best_reward_candidate_dir(output_dir, self.args.best_reward_dir, self.global_step, self.best_reward)
        state = {
            "global_step": self.global_step,
            "best_reward_step": self.global_step,
            "best_reward": self.best_reward,
            "metrics": metrics,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "best_checkpoint": str(best_dir),
            "source_checkpoint": "direct_model_save",
            "best_checkpoint_save_mode": "rotating_create_only",
            "best_checkpoint_keep_limit": _best_reward_keep_limit(self.args),
        }
        save_policy_checkpoint(self.model, self.tokenizer, best_dir, self.args, self.optimizer, self.scheduler, state)
        (best_dir / "best_reward.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        (best_dir / "_SUCCESS").write_text(json.dumps(state, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        _rotate_best_reward_candidates(self.args, best_dir.parent, best_dir)

    def update_with_rollouts(self, rollout_id: str, completions: list[RolloutCompletion], rewards_payload: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.pending.pop(rollout_id)
        samples = self._samples_from_completions(batch, completions)
        rewards = [float(item.get("reward", self.args.reward_invalid)) for item in rewards_payload]
        seq_batch = self._sequence_batch(batch, samples)

        self.model.train()
        with torch.no_grad():
            old_logps, mask = self._expanded_logprobs(self.model, seq_batch)
            old_logps = old_logps.detach()
            ref_logps = self._reference_logprobs(seq_batch)
        advantages = self._advantages(rewards, group_count=len(batch.sample_ids))

        step_metrics: dict[str, float] = {}
        step_loss = None
        for _ in range(max(1, self.args.num_ppo_epochs)):
            current_logps, mask = self._expanded_logprobs(self.model, seq_batch)
            loss, step_metrics = self._loss_from_logprobs(current_logps, old_logps, ref_logps, mask, advantages)
            loss.backward()
            step_loss = loss.detach()

        if self.args.max_grad_norm and self.args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in unwrap_model(self.model).parameters() if p.requires_grad],
                self.args.max_grad_norm,
            )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

        mean_reward = sum(rewards) / max(len(rewards), 1)
        reward_metrics = _aggregate_reward_metrics(
            rewards_payload=rewards_payload,
            rewards=rewards,
            samples=samples,
            group_count=len(batch.sample_ids),
            num_generations=self.args.num_generations,
        )
        metrics = {
            "step": self.global_step,
            "loss": float(step_loss.item() if step_loss is not None else 0.0),
            "reward_mean": float(mean_reward),
            "reward_min": float(min(rewards) if rewards else 0.0),
            "reward_max": float(max(rewards) if rewards else 0.0),
            "lr": self.scheduler.get_last_lr()[0],
            **step_metrics,
            **reward_metrics,
        }
        if self.global_step % self.args.logging_steps == 0 or self.global_step == 1:
            self._write_jsonl(self.metrics_path, metrics)
            for payload, sample in zip(rewards_payload[:2], samples[:2]):
                self._write_jsonl(
                    self.rollout_path,
                    {
                        "step": self.global_step,
                        "sample_id": sample.sample_id,
                        "reward": payload.get("reward"),
                        "parse_ok": payload.get("parse_ok"),
                        "components": payload.get("components"),
                        "counts": payload.get("counts"),
                        "prediction": sample.text[:500],
                    },
                )
        self._maybe_save_best(mean_reward, metrics)
        if self.global_step % self.args.save_steps == 0:
            self._save(f"checkpoint-{self.global_step}", metrics)
        metrics["done"] = self.global_step >= self.total_steps
        return metrics

    def save_final(self) -> dict[str, Any]:
        metrics = {"step": self.global_step, "best_reward": self.best_reward}
        self._save("final", metrics)
        merged_dir = None
        if self.args.export_merged_checkpoints and self.args.lora_enable:
            merged_dir = Path(self.args.output_dir) / self.args.merged_dir_name
            export_merged_lora_checkpoint(unwrap_model(self.model), self.tokenizer, merged_dir)
        return {"final": str(Path(self.args.output_dir) / "final"), "merged": str(merged_dir) if merged_dir else None}


class RewardWorker:
    def __init__(self, data_args: RLDataArguments, train_args: GRPOArguments):
        intersection_weight = train_args.reward_intersection_weight
        if data_args.map_task == "lane":
            intersection_weight = 0.0
        self.scorer = MapRewardScorer(
            RewardSettings(
                map_task=data_args.map_task,
                coord_mode=data_args.coord_mode,
                coord_range=data_args.coord_range,
                patch_size=data_args.patch_size,
                meter_per_pixel=data_args.meter_per_pixel,
                invalid_reward=train_args.reward_invalid,
                format_weight=train_args.reward_format_weight,
                centerline_instance_weight=train_args.reward_centerline_instance_weight,
                centerline_length_weight=train_args.reward_centerline_length_weight,
                cut_type_weight=train_args.reward_cut_type_weight,
                cut_continuity_weight=train_args.reward_cut_continuity_weight,
                intersection_weight=intersection_weight,
                buffer_size=train_args.reward_buffer_size,
                match_threshold=train_args.reward_match_threshold,
            )
        )

    def score(self, batch: RolloutBatch, completions: list[RolloutCompletion]) -> list[dict[str, Any]]:
        prompt_by_group = {prompt.group_index: prompt for prompt in batch.prompts}
        samples = []
        for completion in completions:
            prompt = prompt_by_group[completion.group_index]
            samples.append(
                RolloutSample(
                    group_index=completion.group_index,
                    sample_id=completion.sample_id,
                    prompt_ids=torch.tensor(prompt.prompt_ids, dtype=torch.long),
                    completion_ids=torch.tensor(completion.token_ids, dtype=torch.long),
                    text=completion.text,
                    ground_truth=prompt.ground_truth,
                    coord_config=prompt.coord_config,
                    image_index=prompt.image_index,
                )
            )
        return self.scorer.score_many(samples)


class GRPOCoordinator:
    def __init__(self, model_args: RLModelArguments, data_args: RLDataArguments, train_args: GRPOArguments):
        self.model_args = model_args
        self.data_args = data_args
        self.args = train_args
        Path(train_args.output_dir).mkdir(parents=True, exist_ok=True)
        self.run_config = {
            "model_args": asdict(model_args),
            "data_args": asdict(data_args),
            "training_args": asdict(train_args),
        }
        self.run_config_path = Path(train_args.output_dir) / "grpo_run_config.json"
        self.run_config_path.write_text(
            json.dumps(self.run_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.swanlab_run = init_swanlab_run(
            train_args,
            self.run_config,
            default_experiment_name=Path(train_args.output_dir).name,
        )

    def _resolve_vllm_model_path(self) -> str:
        if self.args.vllm_model_path:
            return self.args.vllm_model_path
        source_checkpoint = self.model_args.model_name_or_path
        if is_peft_checkpoint(source_checkpoint):
            adapter_config_path = Path(source_checkpoint) / "adapter_config.json"
            adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            source_checkpoint = self.model_args.model_base or adapter_config.get("base_model_name_or_path")
            if not source_checkpoint:
                raise ValueError(
                    "PEFT GRPO checkpoint requires --model_base or adapter_config.base_model_name_or_path "
                    "so vLLM can export the text decoder base model."
                )
        export_dir = Path(self.args.output_dir) / "vllm_text_model"
        return str(export_text_decoder_checkpoint(source_checkpoint, export_dir))

    def _device_backend(self) -> str:
        requested = str(self.args.device_backend or "auto").lower()
        if requested in {"ascend", "npu"}:
            return "npu"
        if requested in {"cuda", "gpu"}:
            return "cuda"
        if requested != "auto":
            raise ValueError("--device_backend must be one of: auto, cuda, npu")
        if torch.cuda.is_available():
            return "cuda"
        if _npu_available():
            return "npu"
        raise RuntimeError("No CUDA GPU or Ascend NPU device is available for vLLM GRPO.")

    def _npu_worker_env(self, devices: str | None) -> dict[str, str]:
        env = {
            "VLLM_TARGET_DEVICE": "npu",
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "false"),
        }
        for key in (
            "ASCEND_CUSTOM_PATH",
            "ASCEND_CUSTOM_OPP_PATH",
            "ASCEND_OPP_PATH",
            "HCCL_WHITELIST_DISABLE",
            "HCCL_CONNECT_TIMEOUT",
            "HCCL_EXEC_TIMEOUT",
            "HCCL_IF_BASE_PORT",
            "INF_NAN_MODE_ENABLE",
            "WITHOUT_JIT_COMPILE",
            "COMBINED_ENABLE",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "PATH",
        ):
            value = os.environ.get(key)
            if value:
                env[key] = value
        if devices:
            env["ASCEND_RT_VISIBLE_DEVICES"] = str(devices)
            env["ASCEND_VISIBLE_DEVICES"] = str(devices)
        return env

    def _remote_workers(self, ray, device_backend: str):
        if device_backend == "cuda":
            actor_cls = ray.remote(num_gpus=self.args.actor_num_gpus)(ActorWorker)
            rollout_cls = ray.remote(num_gpus=self.args.rollout_num_gpus)(VLLMPromptEmbedRolloutWorker)
            return actor_cls, rollout_cls
        if device_backend == "npu":
            actor_cls = ray.remote(
                num_cpus=self.args.actor_num_cpus,
                runtime_env={"env_vars": self._npu_worker_env(self.args.actor_npu_devices)},
            )(ActorWorker)
            rollout_cls = ray.remote(
                num_cpus=self.args.rollout_num_cpus,
                runtime_env={"env_vars": self._npu_worker_env(self.args.rollout_npu_devices)},
            )(VLLMPromptEmbedRolloutWorker)
            return actor_cls, rollout_cls
        raise ValueError(f"Unsupported GRPO device backend: {device_backend}")

    def train(self) -> dict[str, Any]:
        try:
            import ray
        except ImportError as exc:
            raise ImportError(
                "Ray is required for the formal verl-style GRPO coordinator. "
                "Install ray and vllm before running RL training."
            ) from exc

        if not ray.is_initialized():
            ray.init(address=self.args.ray_address, ignore_reinit_error=True)

        device_backend = self._device_backend()
        vllm_model_path = self._resolve_vllm_model_path()
        Actor, Rollout = self._remote_workers(ray, device_backend)
        Reward = ray.remote(num_cpus=self.args.reward_num_cpus)(RewardWorker)

        actor = Actor.remote(self.model_args, self.data_args, self.args)
        rollout = Rollout.remote(
            model_path=vllm_model_path,
            num_generations=self.args.num_generations,
            max_new_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            tensor_parallel_size=self.args.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
            dtype=self.args.vllm_dtype,
            max_model_len=self.args.vllm_max_model_len,
            enable_lora=self.args.lora_enable,
            max_lora_rank=self.args.lora_r,
            enforce_eager=self.args.vllm_enforce_eager,
            trust_remote_code=self.args.vllm_trust_remote_code,
            device_backend=device_backend,
        )
        reward = Reward.remote(self.data_args, self.args)

        progress = tqdm(total=self.args.max_steps, desc=f"GRPO(vLLM-{device_backend})", disable=False)
        last_metrics = {}
        try:
            while True:
                prepared = ray.get(actor.prepare_rollout_batch.remote())
                batch = RolloutBatch(**prepared)
                completions = ray.get(rollout.generate.remote(batch))
                rewards = ray.get(reward.score.remote(batch, completions))
                last_metrics = ray.get(actor.update_with_rollouts.remote(batch.rollout_id, completions, rewards))
                progress.update(1)
                progress.set_postfix(
                    reward=f"{last_metrics.get('reward_mean', 0.0):.4g}",
                    loss=f"{last_metrics.get('loss', 0.0):.4g}",
                    kl=f"{last_metrics.get('approx_kl', 0.0):.4g}",
                )
                if self.swanlab_run is not None:
                    log_swanlab(last_metrics, step=int(last_metrics.get("step", 0) or 0))
                if last_metrics.get("step", 0) % self.args.logging_steps == 0 or last_metrics.get("step") == 1:
                    print(json.dumps(last_metrics, ensure_ascii=False))
                if last_metrics.get("done"):
                    break
            final_info = ray.get(actor.save_final.remote())
            if self.swanlab_run is not None:
                log_swanlab(
                    {
                        "final_step": last_metrics.get("step", 0),
                        "final_reward_mean": last_metrics.get("reward_mean", 0.0),
                        "final_loss": last_metrics.get("loss", 0.0),
                        "final_checkpoint_saved": 1.0,
                        "merged_checkpoint_saved": 1.0 if final_info.get("merged") else 0.0,
                    },
                    step=int(last_metrics.get("step", 0) or 0),
                )
            return {
                "last_metrics": last_metrics,
                "final": final_info,
                "vllm_model_path": vllm_model_path,
                "device_backend": device_backend,
            }
        finally:
            progress.close()
            if self.swanlab_run is not None:
                finish_swanlab()
