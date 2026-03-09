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


def select_torch_device(prefer_cuda: bool = True) -> torch.device:
    forced = str(os.environ.get("UNIMAPGEN_DEVICE", "")).strip().lower()
    if forced:
        if forced == "cpu":
            print("[Device] Forced to CPU by UNIMAPGEN_DEVICE=cpu", flush=True)
            return torch.device("cpu")
        if forced.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "UNIMAPGEN_DEVICE requests CUDA, but torch.cuda.is_available() is False."
                )
            return torch.device(forced)
        raise ValueError(f"Unsupported UNIMAPGEN_DEVICE value: {forced}")

    if not prefer_cuda or not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        cap = torch.cuda.get_device_capability(0)
        arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
        supported = []
        for arch in arch_list:
            if not str(arch).startswith("sm_"):
                continue
            num = str(arch)[3:]
            if len(num) == 2 and num.isdigit():
                supported.append((int(num[0]), int(num[1])))
            elif len(num) == 3 and num.isdigit():
                supported.append((int(num[:2]), int(num[2])))
        if supported and cap > max(supported):
            print(
                "[Device] CUDA detected but current GPU capability "
                f"sm_{cap[0]}{cap[1]} is newer than this PyTorch build supports "
                f"(max compiled arch: sm_{max(supported)[0]}{max(supported)[1]}). "
                "Falling back to CPU. Set UNIMAPGEN_DEVICE=cpu to silence this warning.",
                flush=True,
            )
            return torch.device("cpu")
    except Exception as exc:
        print(f"[Device] CUDA capability probe failed ({exc}); falling back to CPU.", flush=True)
        return torch.device("cpu")

    return torch.device("cuda")


def cosine_lr(global_step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if global_step < warmup_steps:
        return base_lr * float(global_step + 1) / float(max(1, warmup_steps))
    progress = float(global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
