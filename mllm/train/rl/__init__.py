"""Reinforcement-learning training entrypoints."""

__all__ = [
    "GRPODataArguments",
    "GRPOModelArguments",
    "GRPOTrainingArguments",
    "MapGRPOTrainer",
    "train",
]


def __getattr__(name):
    if name in __all__:
        from mllm.train.rl import grpo

        return getattr(grpo, name)
    raise AttributeError(name)
