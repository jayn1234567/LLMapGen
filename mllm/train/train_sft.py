"""SFT entrypoint.

The implementation is kept in ``train_qwen`` for compatibility with existing
scripts.  New scripts can call ``python -m mllm.train.train_sft``.
"""

from .train_qwen import train


if __name__ == "__main__":
    train()
