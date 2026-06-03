import json
import os
import subprocess
import torch
import torch.nn as nn

from torch.utils.data import Sampler

import transformers
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    # ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional
from mllm.train.checkpoint_metadata import (
    sync_qwen_multimodal_config,
    write_qwen_multimodal_checkpoint_metadata,
)
from mllm.model.qwen_token_utils import sync_qwen_token_config


ALL_LAYERNORM_LAYERS = [nn.LayerNorm, nn.BatchNorm2d]


def _ensure_generation_config(model):
    if getattr(model, "generation_config", None) is not None:
        return

    config = getattr(model, "config", None)
    if hasattr(config, "to_dict"):
        model.generation_config = transformers.GenerationConfig.from_model_config(config)
    elif isinstance(config, dict):
        model.generation_config = transformers.GenerationConfig.from_dict(config)
    else:
        model.generation_config = transformers.GenerationConfig()


def _save_config_pretrained(model, output_dir):
    config = getattr(model, "config", None)
    if hasattr(config, "save_pretrained"):
        config.save_pretrained(output_dir)
        return
    if isinstance(config, dict):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)


def _safe_regular_checkpoint_delete_path(path: str, output_dir: str) -> bool:
    abs_path = os.path.abspath(path)
    abs_output_dir = os.path.abspath(output_dir)
    if not abs_path or abs_path == os.path.sep:
        return False
    try:
        if os.path.commonpath([abs_path, abs_output_dir]) != abs_output_dir:
            return False
    except ValueError:
        return False
    basename = os.path.basename(abs_path)
    if not basename.startswith("checkpoint-"):
        return False
    return basename[len("checkpoint-"):].isdigit()


def _remove_regular_checkpoint_tree(path: str, output_dir: str) -> bool:
    if not _safe_regular_checkpoint_delete_path(path, output_dir):
        logger.warning(f"Refusing to delete unexpected checkpoint path: {path}")
        return False
    if not os.path.exists(path):
        return True

    # Some NPU/cloud filesystems do not allow rename/replace-style directory
    # updates, but do allow deleting a validated directory with rm -rf.
    try:
        subprocess.run(
            ["rm", "-rf", "--", path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        logger.warning(f"rm -rf fallback failed for old checkpoint {path}: {exc}")
        return False

    if os.path.exists(path):
        logger.warning(f"rm -rf fallback finished but checkpoint still exists: {path}")
        return False
    return True


def _sorted_regular_checkpoints(output_dir: str, use_mtime: bool = False) -> List[str]:
    if not os.path.isdir(output_dir):
        return []

    checkpoints = []
    for name in os.listdir(output_dir):
        if not name.startswith("checkpoint-"):
            continue
        step = name[len("checkpoint-"):]
        if not step.isdigit():
            continue
        path = os.path.join(output_dir, name)
        if not os.path.isdir(path):
            continue
        order = os.path.getmtime(path) if use_mtime else int(step)
        checkpoints.append((order, path))

    if use_mtime and len(checkpoints) > 1:
        mtimes = [item[0] for item in checkpoints]
        if max(mtimes) - min(mtimes) < 1.0:
            return _sorted_regular_checkpoints(output_dir, use_mtime=False)

    return [path for _, path in sorted(checkpoints)]


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


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
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return
        if not getattr(self.args, "should_save", True):
            return

        output_dir = output_dir or self.args.output_dir
        checkpoints_sorted = _sorted_regular_checkpoints(output_dir, use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        save_total_limit = self.args.save_total_limit
        protected = {os.path.abspath(checkpoints_sorted[-1])}
        best_checkpoint = getattr(self.state, "best_model_checkpoint", None)
        if best_checkpoint is not None:
            protected.add(os.path.abspath(best_checkpoint))
            save_total_limit = max(save_total_limit, len(protected))

        remaining = len(checkpoints_sorted)
        for checkpoint in checkpoints_sorted:
            if remaining <= save_total_limit:
                break
            if os.path.abspath(checkpoint) in protected:
                continue
            logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            if _remove_regular_checkpoint_tree(checkpoint, output_dir):
                remaining -= 1

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if not getattr(self.args, "lora_enable", False):
            sync_qwen_token_config(model=self.model)
            sync_qwen_multimodal_config(self.model)
            return super().save_model(output_dir=output_dir, _internal_call=_internal_call)

        output_dir = output_dir or self.args.output_dir
        model = self.model
        sync_qwen_token_config(model=model)
        sync_qwen_multimodal_config(model)

        # Avoid HF Trainer's DeepSpeed ZeRO-3 PEFT state-dict path, which can
        # try to consolidate frozen parameters before our LoRA-only save.
        # These gather helpers are collective under ZeRO-3, so all ranks enter.
        lora_state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(),
            getattr(self.args, "lora_bias", "none"),
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())

        if self.is_world_process_zero():
            os.makedirs(output_dir, exist_ok=True)
            _ensure_generation_config(model)
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            _save_config_pretrained(model, output_dir)
            model.save_pretrained(output_dir, state_dict=lora_state_dict)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
            torch.save(non_lora_state_dict, os.path.join(output_dir, "non_lora_trainables.bin"))
            write_qwen_multimodal_checkpoint_metadata(model, output_dir, self)
        return

    def _get_train_sampler(self, train_dataset=None) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            lr_mapper = []
            if self.args.mm_projector_lr is not None:
                lr_mapper.append(("mm_projector", self.args.mm_projector_lr))
            if getattr(self.args, "mm_vision_fusion_lr", None) is not None:
                fusion_lr = self.args.mm_vision_fusion_lr
                lr_mapper.extend([
                    ("vision_tower.expert_adapters", fusion_lr),
                    ("vision_tower.router", fusion_lr),
                    ("vision_tower.post_fusion", fusion_lr),
                    ("vision_tower.out_norm", fusion_lr),
                ])
            if self.args.mm_vision_tower_lr is not None:
                lr_mapper.append(("vision_tower", self.args.mm_vision_tower_lr))

            if len(lr_mapper) > 0:
                def _matched_lr(name):
                    for module_keyword, lr in lr_mapper:
                        if module_keyword in name:
                            return lr
                    return None

                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if _matched_lr(name) is not None]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if
                                   (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if
                                   (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                special_lrs = sorted({float(lr) for _, lr in lr_mapper})
                for lr in special_lrs:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if
                                           (n in decay_parameters and _matched_lr(n) == lr and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if
                                           (n not in decay_parameters and _matched_lr(n) == lr and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        sync_qwen_multimodal_config(self.model)

        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.is_world_process_zero():
                sync_qwen_multimodal_config(self.model)
                _save_config_pretrained(self.model, output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
                write_qwen_multimodal_checkpoint_metadata(self.model, output_dir, self)
            self._rotate_checkpoints(output_dir=run_dir)
        else:
            # Workaround for the issue: https://github.com/haotian-liu/LLaVA/issues/1144
            _ensure_generation_config(model)
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            sync_qwen_token_config(model=model)
            super(LLaVATrainer, self)._save_checkpoint(model, trial)
            write_qwen_multimodal_checkpoint_metadata(self.model, output_dir, self)
            self._rotate_checkpoints(output_dir=run_dir)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            # Workaround for the issue: https://github.com/haotian-liu/LLaVA/issues/1144
            _ensure_generation_config(self.model)
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            sync_qwen_token_config(model=self.model)
            sync_qwen_multimodal_config(self.model)
            non_lora_state_dict = None
            if getattr(self.args, "lora_enable", False):
                # ZeRO-3 parameter gather is a collective context. Every rank
                # must enter it even though only rank0 writes the resulting file.
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters())
            super(LLaVATrainer, self)._save(output_dir, state_dict)
            if getattr(self.args, "lora_enable", False) and self.is_world_process_zero():
                _save_config_pretrained(self.model, output_dir)
                torch.save(non_lora_state_dict, os.path.join(output_dir, "non_lora_trainables.bin"))
            write_qwen_multimodal_checkpoint_metadata(self.model, output_dir, self)
