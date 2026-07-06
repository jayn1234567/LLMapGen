from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mllm.coord_utils import COORD_MODE_PIXEL, normalize_coord_mode
from mllm.reward import MapRewardConfig, compute_map_reward

from .rollout import RolloutSample


@dataclass
class RewardSettings:
    map_task: str = "lane"
    coord_mode: str = "auto"
    coord_range: int = 1000
    patch_size: int = 256
    meter_per_pixel: float = 0.2
    invalid_reward: float = -1.0
    format_weight: float = 0.08
    centerline_instance_weight: float = 0.37
    centerline_length_weight: float = 0.45
    cut_type_weight: float = 0.05
    cut_continuity_weight: float = 0.05
    intersection_weight: float = 0.0
    buffer_size: float = 1.0
    match_threshold: float = 0.33


def _coord_mode_for_sample(settings: RewardSettings, coord_config: dict[str, Any]) -> str:
    if normalize_coord_mode(settings.coord_mode) != "auto":
        return normalize_coord_mode(settings.coord_mode)
    return normalize_coord_mode(coord_config.get("coord_mode") or COORD_MODE_PIXEL)


class MapRewardScorer:
    def __init__(self, settings: RewardSettings):
        self.settings = settings

    def score(self, sample: RolloutSample) -> dict[str, Any]:
        coord = sample.coord_config or {}
        patch_size = int(coord.get("patch_size") or self.settings.patch_size)
        coord_range = int(coord.get("coord_range") or self.settings.coord_range)
        reward_config = MapRewardConfig(
            map_task=self.settings.map_task,
            patch_size=patch_size,
            coord_mode=_coord_mode_for_sample(self.settings, coord),
            coord_range=coord_range,
            invalid_reward=self.settings.invalid_reward,
            format_weight=self.settings.format_weight,
            centerline_instance_weight=self.settings.centerline_instance_weight,
            centerline_length_weight=self.settings.centerline_length_weight,
            cut_type_weight=self.settings.cut_type_weight,
            cut_continuity_weight=self.settings.cut_continuity_weight,
            intersection_weight=self.settings.intersection_weight,
            meter_per_pixel=self.settings.meter_per_pixel,
            buffer_size=self.settings.buffer_size,
            match_threshold=self.settings.match_threshold,
        )
        result = compute_map_reward(sample.text, sample.ground_truth, reward_config)
        result["sample_id"] = sample.sample_id
        result["prediction"] = sample.text
        return result

    def score_many(self, samples: list[RolloutSample]) -> list[dict[str, Any]]:
        return [self.score(sample) for sample in samples]
