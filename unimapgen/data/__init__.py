from .builders import build_dataset_from_cfg
from .dataset import DatasetConfig, NuScenesSatelliteMapDataset, collate_fn
from .nuscenes_sdmap_dataset import NuScenesSDMapDataset, NuScenesSDMapDatasetConfig
from .opensatmap_dataset import OpenSatMapDataset, OpenSatMapDatasetConfig
from .qwen_map_dataset import (
    OpenSatMapQwenDataset,
    OpenSatMapQwenDatasetConfig,
    QwenMapCollator,
    build_state_token_ids_from_lines,
    filter_state_prefix_lines,
)


def __getattr__(name):
    if name == "QwenMapTokenizer":
        from .qwen_map_tokenizer import QwenMapTokenizer

        return QwenMapTokenizer
    raise AttributeError(name)


__all__ = [
    "DatasetConfig",
    "NuScenesSatelliteMapDataset",
    "NuScenesSDMapDatasetConfig",
    "NuScenesSDMapDataset",
    "OpenSatMapDatasetConfig",
    "OpenSatMapDataset",
    "OpenSatMapQwenDatasetConfig",
    "OpenSatMapQwenDataset",
    "build_state_token_ids_from_lines",
    "filter_state_prefix_lines",
    "QwenMapTokenizer",
    "QwenMapCollator",
    "build_dataset_from_cfg",
    "collate_fn",
]
