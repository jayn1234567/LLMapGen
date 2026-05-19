import json
import os


def _unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _get_config(model):
    model = _unwrap_model(model)
    return getattr(model, "config", None)


def _cfg_get(config, name, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _cfg_set(config, name, value):
    if isinstance(config, dict):
        config[name] = value
    else:
        setattr(config, name, value)


def _get_vision_tower(model):
    model = _unwrap_model(model)
    get_model = getattr(model, "get_model", None)
    if callable(get_model):
        model = get_model()

    get_vision_tower = getattr(model, "get_vision_tower", None)
    if callable(get_vision_tower):
        vision_tower = get_vision_tower()
    else:
        vision_tower = getattr(model, "vision_tower", None)

    if isinstance(vision_tower, list):
        vision_tower = vision_tower[0] if vision_tower else None
    return vision_tower


def _infer_vision_tower_type(vision_tower, config=None):
    explicit_type = getattr(vision_tower, "mm_vision_tower_type", None)
    if explicit_type:
        return explicit_type
    if config is not None:
        config_type = _cfg_get(config, "mm_vision_tower_type")
        if config_type:
            return config_type

    class_name = vision_tower.__class__.__name__.lower() if vision_tower is not None else ""
    tower_name = str(getattr(vision_tower, "vision_tower_name", "")).lower()
    if "dinov3" in class_name or "dinov3" in tower_name:
        return "dinov3"
    if "dinov2" in class_name or "dinov2" in tower_name:
        return "dinov2"
    if "mobileclip" in class_name or "mobileclip" in tower_name:
        return "mobileclip"
    if "clip" in class_name or "clip" in tower_name:
        return "clip"
    return None


def _as_jsonable_list(value):
    if value is None:
        return None
    return [int(item) for item in value]


def _resolve_input_image_size(vision_tower, config):
    for attr in ("_target_size", "input_image_size"):
        value = getattr(vision_tower, attr, None)
        if value is not None:
            return int(value)

    inner_tower = getattr(vision_tower, "vision_tower", None)
    inner_config = getattr(inner_tower, "config", None)
    value = getattr(inner_config, "image_size", None)
    if value is not None:
        return int(value)

    value = _cfg_get(config, "input_image_size")
    return int(value) if value is not None else None


def _infer_language_model_type(model, config):
    raw_values = [
        _cfg_get(config, "model_type", ""),
        _cfg_get(config, "_name_or_path", ""),
        model.__class__.__name__,
    ]
    joined = " ".join(str(value).lower() for value in raw_values if value is not None)
    if "qwen3" in joined or "qwen-3" in joined:
        return "qwen3"
    if "qwen2" in joined or "qwen-2" in joined:
        return "qwen2"
    return _cfg_get(config, "model_type")


def sync_qwen_multimodal_config(model):
    unwrapped_model = _unwrap_model(model)
    config = _get_config(model)
    if config is None:
        return None

    model_type = _infer_language_model_type(unwrapped_model, config)
    if model_type:
        _cfg_set(config, "model_type", model_type)

    vision_tower = _get_vision_tower(model)
    if vision_tower is None:
        return config

    vision_tower_name = getattr(vision_tower, "vision_tower_name", None)
    if vision_tower_name:
        _cfg_set(config, "mm_vision_tower", vision_tower_name)
    elif not _cfg_get(config, "mm_vision_tower"):
        _cfg_set(config, "mm_vision_tower", _cfg_get(config, "vision_tower"))

    if not _cfg_get(config, "vision_tower"):
        _cfg_set(config, "vision_tower", _cfg_get(config, "mm_vision_tower"))

    vision_tower_type = _infer_vision_tower_type(vision_tower, config)
    if vision_tower_type:
        _cfg_set(config, "mm_vision_tower_type", vision_tower_type)

    _cfg_set(config, "input_image_size", _resolve_input_image_size(vision_tower, config))
    deepstack_visual_indexes = _as_jsonable_list(getattr(vision_tower, "deepstack_visual_indexes", None))
    _cfg_set(config, "deepstack_visual_indexes", deepstack_visual_indexes)
    _cfg_set(config, "disable_deepstack", deepstack_visual_indexes is None)
    return config


def write_qwen_multimodal_checkpoint_metadata(model, output_dir: str, trainer=None):
    if trainer is not None and not trainer.is_world_process_zero():
        return
    if output_dir is None:
        return

    config = sync_qwen_multimodal_config(model)
    if config is None:
        return
    if not (_cfg_get(config, "mm_vision_tower") or _cfg_get(config, "vision_tower")):
        return

    payload = {
        "format": "qwen_multimodal_checkpoint",
        "framework": "mllm",
        "model_type": _cfg_get(config, "model_type"),
        "mm_vision_tower": _cfg_get(config, "mm_vision_tower"),
        "vision_tower": _cfg_get(config, "vision_tower"),
        "mm_vision_tower_type": _cfg_get(config, "mm_vision_tower_type"),
        "input_image_size": _cfg_get(config, "input_image_size"),
        "deepstack_visual_indexes": _cfg_get(config, "deepstack_visual_indexes"),
        "disable_deepstack": _cfg_get(config, "disable_deepstack"),
        "bundled_vision_tower": True,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "qwen_multimodal_checkpoint.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
