from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class PoolBucket(str, Enum):
    HARD_PARSE_FAIL = "hard_parse_fail"
    HARD_ZERO_MATCH = "hard_zero_match"
    HARD_LOW_F1 = "hard_low_f1"
    HARD_CUT_ERROR = "hard_cut_error"
    MEDIUM = "medium"
    RANDOM_KEEP = "random_keep"


@dataclass
class RewardBreakdown:
    reward: float
    parse_ok: bool
    components: dict[str, float] = field(default_factory=dict)
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RolloutRequest:
    sample_id: str
    image: str
    prompt: str
    meta: dict[str, Any] = field(default_factory=dict)
    num_generations: int = 4
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RolloutCandidate:
    text: str
    finish_reason: str | None = None
    token_count: int | None = None
    reward: RewardBreakdown | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.reward is not None:
            result["reward"] = self.reward.to_dict()
        return result


@dataclass
class RolloutResult:
    sample_id: str
    candidates: list[RolloutCandidate]
    backend: str
    latency_s: float | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "backend": self.backend,
            "latency_s": self.latency_s,
            "error": self.error,
            "extra": self.extra,
        }
