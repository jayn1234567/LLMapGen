import random


def deterministic_sample_indices(total: int, limit: int | None, seed: int) -> list[int]:
    """Return source-ordered indices for a reproducible subset."""
    if limit is None or limit <= 0 or total <= limit:
        return list(range(total))
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), limit))
