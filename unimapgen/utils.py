import os
import random
import re
from typing import Any, Dict

import numpy as np
import torch
import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env_value(s: str) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(1)
        default = m.group(2)
        if key in os.environ:
            return os.environ[key]
        if default is not None:
            return default
        return m.group(0)

    return os.path.expanduser(_ENV_PATTERN.sub(repl, s))


def _expand_env_recursive(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _expand_env_recursive(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_expand_env_recursive(v) for v in x]
    if isinstance(x, str):
        return _expand_env_value(x)
    return x


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    return _expand_env_recursive(data)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cosine_lr(global_step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if global_step < warmup_steps:
        return base_lr * float(global_step + 1) / float(max(1, warmup_steps))
    progress = float(global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
