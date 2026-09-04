from .map_schema import MapParseResult, extract_json_payload, parse_map_json

__all__ = [
    "MapParseResult",
    "MapRewardConfig",
    "compute_map_reward",
    "compute_map_rewards",
    "extract_json_payload",
    "parse_map_json",
]


def __getattr__(name):
    """Load geometry-based rewards only when a caller actually requests them.

    Prediction-only code uses ``mllm.reward.map_schema`` for JSON parsing.  The
    reward implementation imports the Shapely-backed geometry evaluator, so an
    eager import here made otherwise geometry-free inference fail on minimal DI
    images that do not carry Shapely.
    """
    if name in {"MapRewardConfig", "compute_map_reward", "compute_map_rewards"}:
        from . import map_reward

        value = getattr(map_reward, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
