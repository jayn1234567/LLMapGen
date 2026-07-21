import json
from pathlib import Path

import pytest

from scripts.tools.build_stratified_train_subset import (
    allocate_quotas,
    build_subset,
    parse_ratios,
)


def _write_records(path: Path, per_bucket: int = 20) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for difficulty in ("easy", "medium", "hard", "very_hard"):
            for index in range(per_bucket):
                handle.write(json.dumps({"id": f"{difficulty}-{index}", "difficulty": difficulty}) + "\n")


def _classifier(record):
    return record["difficulty"], False


def test_final_550k_ratios_scale_to_exact_200k_quotas() -> None:
    ratios = parse_ratios("easy=0.30,medium=0.3560290909,hard=0.2439709091,very_hard=0.10")
    assert allocate_quotas(200_000, ratios) == {
        "easy": 60_000,
        "medium": 71_206,
        "hard": 48_794,
        "very_hard": 20_000,
    }


def test_exact_difficulty_quotas_and_determinism(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    _write_records(source)
    ratios = parse_ratios("easy=0.30,medium=0.33,hard=0.27,very_hard=0.10")
    expected = allocate_quotas(20, ratios)

    outputs = []
    for name in ("a", "b"):
        output = tmp_path / f"{name}.jsonl"
        summary = build_subset(
            source,
            output,
            tmp_path / f"{name}.summary.json",
            target_samples=20,
            ratios=ratios,
            seed=42,
            progress_every=0,
            classifier=_classifier,
        )
        assert summary["selected_counts"] == expected
        assert sum(1 for _ in output.open(encoding="utf-8")) == 20
        outputs.append(output.read_text(encoding="utf-8"))

    assert outputs[0] == outputs[1]


def test_fails_when_a_bucket_cannot_fill_quota(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    _write_records(source, per_bucket=1)
    ratios = parse_ratios("easy=0.25,medium=0.25,hard=0.25,very_hard=0.25")

    with pytest.raises(ValueError, match="Insufficient samples"):
        build_subset(
            source,
            tmp_path / "subset.jsonl",
            tmp_path / "summary.json",
            target_samples=8,
            ratios=ratios,
            seed=42,
            progress_every=0,
            classifier=_classifier,
        )
