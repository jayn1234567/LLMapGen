"""Compatibility entrypoint for GRPO training.

The implementation lives under ``mllm.train.rl`` so RL code is grouped in one
place. Existing scripts can keep using ``python -m mllm.train.train_grpo``.
"""

from mllm.train.rl.grpo import train


if __name__ == "__main__":
    train()
