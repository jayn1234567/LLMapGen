from __future__ import annotations

from dataclasses import asdict, is_dataclass
import inspect
import json
import os
from typing import Any


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _is_global_rank0() -> bool:
    for name in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        value = os.environ.get(name)
        if value is not None:
            return int(value) == 0
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _field(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _split_csv(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    items = [item.strip() for item in str(value).replace(";", ",").split(",")]
    return [item for item in items if item] or None


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return {key: value for key, value in kwargs.items() if value is not None and value != ""}
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return {key: value for key, value in kwargs.items() if value is not None and value != ""}
    allowed = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in allowed and value is not None and value != ""}


def _set_swanlab_env(args: Any) -> None:
    mapping = {
        "swanlab_project": "SWANLAB_PROJECT",
        "swanlab_mode": "SWANLAB_MODE",
        "swanlab_log_dir": "SWANLAB_LOG_DIR",
        "swanlab_api_host": "SWANLAB_API_HOST",
        "swanlab_web_host": "SWANLAB_WEB_HOST",
    }
    for attr, env_name in mapping.items():
        value = _field(args, attr)
        if value and not os.environ.get(env_name):
            os.environ[env_name] = str(value)


def _login_if_needed(swanlab_module: Any, args: Any) -> None:
    api_key = os.environ.get("SWANLAB_API_KEY")
    if not api_key:
        return
    login_kwargs = {
        "api_key": api_key,
        "host": _field(args, "swanlab_api_host"),
        "web_host": _field(args, "swanlab_web_host"),
        "save": False,
    }
    swanlab_module.login(**_filter_kwargs(swanlab_module.login, login_kwargs))


def swanlab_is_enabled(args: Any) -> bool:
    return _as_bool(_field(args, "swanlab_enable", False))


def build_swanlab_config(
    model_args: Any = None,
    data_args: Any = None,
    training_args: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = {}
    if model_args is not None:
        config["model_args"] = _jsonable(model_args)
    if data_args is not None:
        config["data_args"] = _jsonable(data_args)
    if training_args is not None:
        config["training_args"] = _jsonable(training_args)
    if extra:
        config["extra"] = _jsonable(extra)
    return config


def build_swanlab_callback(model_args: Any, data_args: Any, training_args: Any):
    if not swanlab_is_enabled(training_args):
        return None
    if not _is_global_rank0():
        return None

    try:
        import swanlab
        from swanlab.integration.transformers import SwanLabCallback
    except ImportError as exc:
        raise ImportError(
            "SwanLab logging is enabled but swanlab is not installed. "
            "Install it with `pip install swanlab` or disable --swanlab_enable."
        ) from exc

    _set_swanlab_env(training_args)
    _login_if_needed(swanlab, training_args)

    kwargs = {
        "project": _field(training_args, "swanlab_project"),
        "experiment_name": _field(training_args, "swanlab_experiment_name") or _field(training_args, "run_name"),
        "description": _field(training_args, "swanlab_description"),
        "config": build_swanlab_config(model_args, data_args, training_args),
        "tags": _split_csv(_field(training_args, "swanlab_tags")),
    }
    return SwanLabCallback(**_filter_kwargs(SwanLabCallback, kwargs))


def init_swanlab_run(
    args: Any,
    config: dict[str, Any],
    *,
    default_experiment_name: str | None = None,
):
    if not swanlab_is_enabled(args):
        return None
    if not _is_global_rank0():
        return None

    try:
        import swanlab
    except ImportError as exc:
        raise ImportError(
            "SwanLab logging is enabled but swanlab is not installed. "
            "Install it with `pip install swanlab` or disable --swanlab_enable."
        ) from exc

    _set_swanlab_env(args)
    _login_if_needed(swanlab, args)

    kwargs = {
        "project": _field(args, "swanlab_project"),
        "experiment_name": _field(args, "swanlab_experiment_name") or default_experiment_name,
        "description": _field(args, "swanlab_description"),
        "config": _jsonable(config),
        "mode": _field(args, "swanlab_mode"),
        "logdir": _field(args, "swanlab_log_dir"),
        "tags": _split_csv(_field(args, "swanlab_tags")),
    }
    return swanlab.init(**_filter_kwargs(swanlab.init, kwargs))


def log_swanlab(metrics: dict[str, Any], step: int | None = None) -> None:
    try:
        import swanlab
    except ImportError:
        return
    kwargs = {"step": step} if step is not None else {}
    swanlab.log(_jsonable(metrics), **_filter_kwargs(swanlab.log, kwargs))


def finish_swanlab() -> None:
    try:
        import swanlab
    except ImportError:
        return
    finish = getattr(swanlab, "finish", None)
    if callable(finish):
        finish()
