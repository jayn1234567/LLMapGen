"""Post-training RL utilities.

The RL stack is separate from SFT. Hard-sample pools, vLLM prompt-embedding
rollout, reward scoring, and GRPO coordination live here so the stable SFT and
data-processing paths do not need RL-specific branches.
"""

from .data_pool import HardPoolConfig, build_hard_pool
from .export import export_text_decoder_checkpoint
from .schemas import PoolBucket, RewardBreakdown, RolloutCandidate, RolloutRequest, RolloutResult

__all__ = [
    "HardPoolConfig",
    "PoolBucket",
    "RewardBreakdown",
    "RolloutCandidate",
    "RolloutRequest",
    "RolloutResult",
    "build_hard_pool",
    "export_text_decoder_checkpoint",
]
