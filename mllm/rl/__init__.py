"""Post-training RL utilities.

This package intentionally does not contain a Trainer implementation yet. The
old in-Trainer GRPO prototype was removed; new RL work should build on these
data/reward/rollout interfaces without changing the stable SFT path.
"""

from .data_pool import HardPoolConfig, build_hard_pool
from .schemas import PoolBucket, RewardBreakdown, RolloutCandidate, RolloutRequest, RolloutResult

__all__ = [
    "HardPoolConfig",
    "PoolBucket",
    "RewardBreakdown",
    "RolloutCandidate",
    "RolloutRequest",
    "RolloutResult",
    "build_hard_pool",
]
