from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RLModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "SFT multimodal checkpoint or PEFT adapter checkpoint used as the GRPO actor policy."}
    )
    model_base: Optional[str] = field(
        default=None,
        metadata={"help": "Optional base multimodal checkpoint for PEFT adapter actor checkpoints."},
    )
    version: str = "conv_qwen_3_Dinov2_huawei"
    vision_tower: Optional[str] = None
    mm_vision_tower_type: Optional[str] = None
    mm_vision_select_layer: int = -1
    mm_vision_select_feature: str = "patch"
    mm_projector_type: str = "mlp2x_gelu"
    mm_patch_merge_type: str = "flat"
    mm_use_im_start_end: bool = False
    mm_use_im_patch_token: bool = True
    input_image_size: Optional[int] = None
    deepstack_visual_indexes: Optional[List[int]] = None
    disable_deepstack: bool = True
    tokenizer_use_fast: Optional[bool] = False


@dataclass
class RLDataArguments:
    data_path: List[str] = field(default_factory=list)
    image_folder: List[str] = field(default_factory=list)
    image_aspect_ratio: str = "pad"
    image_grid_pinpoints: Optional[str] = None
    train_sample_limit: Optional[int] = None
    sample_seed: int = 42
    map_task: str = "lane"
    coord_mode: str = "auto"
    coord_range: int = 1000
    patch_size: int = 256
    meter_per_pixel: float = 0.2


@dataclass
class GRPOArguments:
    output_dir: str = field(metadata={"help": "Directory for RL checkpoints, logs, rollout cache, and exports."})

    # Formal rollout path. HF local generation is intentionally not a training backend.
    rollout_backend: str = field(
        default="vllm_prompt_embeds",
        metadata={"help": "Formal rollout backend. Only vllm_prompt_embeds is accepted for training."},
    )
    vllm_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Text-decoder checkpoint for vLLM. If omitted, it is exported from the actor "
                "multimodal checkpoint to output_dir/vllm_text_model."
            )
        },
    )
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.70
    vllm_dtype: str = "auto"
    vllm_max_model_len: Optional[int] = field(
        default=4096,
        metadata={
            "help": (
                "Maximum vLLM context length for rollout. Keep this bounded; "
                "Qwen3 configs may advertise very long contexts that make vLLM "
                "memory profiling unnecessarily heavy."
            )
        },
    )
    vllm_enforce_eager: bool = False
    vllm_trust_remote_code: bool = True

    # Ray/verl-style role placement.
    ray_address: Optional[str] = None
    actor_num_gpus: float = 1.0
    rollout_num_gpus: float = 1.0
    reward_num_cpus: float = 1.0

    seed: int = 42
    max_steps: int = 100
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: Optional[int] = None
    save_best_reward: bool = True
    best_reward_dir: str = "best_reward"
    export_merged_checkpoints: bool = True
    merged_dir_name: str = "merged"

    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = False
    model_max_length: int = 4096
    dataloader_num_workers: int = 2

    num_generations: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    num_ppo_epochs: int = 1
    clip_range: float = 0.2
    kl_beta: float = 0.02
    advantage_epsilon: float = 1e-6

    reward_invalid: float = -1.0
    reward_format_weight: float = 0.08
    reward_centerline_instance_weight: float = 0.37
    reward_centerline_length_weight: float = 0.45
    reward_cut_type_weight: float = 0.05
    reward_cut_continuity_weight: float = 0.05
    reward_intersection_weight: float = 0.0
    reward_buffer_size: float = 1.0
    reward_match_threshold: float = 0.33

    lora_enable: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_scope: str = "llm"
    lora_target_modules: Optional[str] = None
    lora_exclude_modules: Optional[str] = "lm_head,embed_tokens"
    full_train_scope: str = "all"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None

    report_to: str = "none"
    swanlab_enable: bool = False
    swanlab_project: Optional[str] = None
    swanlab_workspace: Optional[str] = None
    swanlab_experiment_name: Optional[str] = None
    swanlab_group: Optional[str] = None
    swanlab_job_type: Optional[str] = None
    swanlab_description: Optional[str] = None
    swanlab_tags: Optional[str] = None
    swanlab_mode: Optional[str] = None
    swanlab_log_dir: Optional[str] = None
    swanlab_api_host: Optional[str] = None
    swanlab_web_host: Optional[str] = None
    local_rank: int = -1
