"""Resolution-aware Dataset V2/V3 difficulty profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


DIFFICULTY_PROFILE_VERSION = "geometry_v3_resolution_aware_five_tier"
FIVE_TIER_BUCKETS = ("very_easy", "easy", "medium", "hard", "very_hard")


@dataclass(frozen=True)
class BucketCaps:
    max_score: float
    max_centerlines: int
    max_points: int
    max_intersections: int
    max_forks: int
    max_cycles: int
    max_crossings: int
    max_lane_changes: int
    max_non_common_lanes: int
    max_short_fragments: int
    max_total_turn: float
    max_single_turn: float


@dataclass(frozen=True)
class DifficultyProfile:
    name: str
    patch_size: int
    score_free_centerlines: int
    score_free_points: int
    caps: dict[str, BucketCaps]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": DIFFICULTY_PROFILE_VERSION,
            "patch_size": self.patch_size,
            "score_free_centerlines": self.score_free_centerlines,
            "score_free_points": self.score_free_points,
            "caps": {name: asdict(caps) for name, caps in self.caps.items()},
        }


LOCAL512_PROFILE = DifficultyProfile(
    name="local512_profile_a",
    patch_size=512,
    score_free_centerlines=6,
    score_free_points=32,
    caps={
        "very_easy": BucketCaps(0.35, 2, 10, 0, 0, 0, 0, 0, 0, 0, 45.0, 30.0),
        "easy": BucketCaps(1.5, 6, 32, 0, 0, 0, 0, 0, 999, 1, 180.0, 75.0),
        "medium": BucketCaps(4.5, 10, 56, 1, 1, 1, 1, 1, 999, 3, 500.0, 120.0),
        "hard": BucketCaps(9.0, 18, 96, 3, 3, 3, 3, 3, 999, 6, 900.0, 165.0),
    },
)


ROI256_PROFILE = DifficultyProfile(
    name="roi256_profile_a",
    patch_size=256,
    score_free_centerlines=3,
    score_free_points=16,
    caps={
        "very_easy": BucketCaps(0.25, 1, 6, 0, 0, 0, 0, 0, 0, 0, 30.0, 20.0),
        "easy": BucketCaps(1.25, 3, 16, 0, 0, 0, 0, 0, 999, 0, 120.0, 60.0),
        "medium": BucketCaps(4.0, 7, 36, 1, 1, 1, 1, 1, 999, 2, 360.0, 110.0),
        "hard": BucketCaps(8.5, 14, 72, 3, 3, 3, 3, 3, 999, 5, 720.0, 160.0),
    },
)


PROFILES = {
    LOCAL512_PROFILE.name: LOCAL512_PROFILE,
    ROI256_PROFILE.name: ROI256_PROFILE,
}


def resolve_difficulty_profile(name: str = "", patch_size: int | None = None) -> DifficultyProfile:
    normalized = str(name or "").strip().lower()
    aliases = {
        "local512": LOCAL512_PROFILE.name,
        "512": LOCAL512_PROFILE.name,
        "roi256": ROI256_PROFILE.name,
        "context512_roi256": ROI256_PROFILE.name,
        "256": ROI256_PROFILE.name,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized:
        if normalized not in PROFILES:
            raise ValueError(f"unknown difficulty profile: {name!r}; choices={sorted(PROFILES)}")
        profile = PROFILES[normalized]
    elif patch_size is not None and int(patch_size) >= 512:
        profile = LOCAL512_PROFILE
    else:
        profile = ROI256_PROFILE
    if patch_size is not None and int(patch_size) != profile.patch_size:
        raise ValueError(
            f"difficulty profile {profile.name} expects patch_size={profile.patch_size}, got {patch_size}"
        )
    return profile


def resolution_aware_score(metrics: dict[str, Any], profile: DifficultyProfile) -> float:
    components = dict(metrics.get("difficulty_score_components") or {})
    components["line_instances"] = min(
        4.0,
        max(0, int(metrics.get("centerline_count", 0)) - profile.score_free_centerlines) * 0.4,
    )
    components["output_points"] = min(
        3.0,
        max(0, int(metrics.get("point_count", 0)) - profile.score_free_points) * 0.04,
    )
    metrics["difficulty_score_components"] = {
        key: round(float(value), 3) for key, value in components.items()
    }
    score = float(sum(components.values()))
    metrics["difficulty_score"] = round(score, 3)
    return score


def within_caps(metrics: dict[str, Any], caps: BucketCaps) -> bool:
    return all((
        float(metrics.get("difficulty_score", 0.0)) <= caps.max_score,
        int(metrics.get("centerline_count", 0)) <= caps.max_centerlines,
        int(metrics.get("point_count", 0)) <= caps.max_points,
        int(metrics.get("intersection_count", 0)) <= caps.max_intersections,
        int(metrics.get("fork_node_count", 0)) <= caps.max_forks,
        int(metrics.get("cycle_count", 0)) <= caps.max_cycles,
        int(metrics.get("crossing_count", 0)) <= caps.max_crossings,
        int(metrics.get("lane_change_like_count", 0)) <= caps.max_lane_changes,
        int(metrics.get("non_common_lane_count", 0)) <= caps.max_non_common_lanes,
        int(metrics.get("short_fragment_count", 0)) <= caps.max_short_fragments,
        float(metrics.get("total_turn_degrees", 0.0)) <= caps.max_total_turn,
        float(metrics.get("max_turn_degrees", 0.0)) <= caps.max_single_turn,
    ))


def classify_metrics(metrics: dict[str, Any], profile: DifficultyProfile) -> str:
    if int(metrics.get("centerline_count", 0)) == 0 and int(metrics.get("intersection_count", 0)) == 0:
        return "empty"
    for name in ("very_easy", "easy", "medium", "hard"):
        if within_caps(metrics, profile.caps[name]):
            return name
    return "very_hard"
