import os


def maybe_disable_cudnn_from_env(torch_module=None):
    value = str(os.environ.get("MLLM_DISABLE_CUDNN", "")).strip().lower()
    if value not in {"1", "true", "yes", "on"}:
        return False

    if torch_module is None:
        import torch as torch_module

    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.enabled = False
        print("MLLM_DISABLE_CUDNN=1: torch.backends.cudnn.enabled=False")
        return True
    return False
