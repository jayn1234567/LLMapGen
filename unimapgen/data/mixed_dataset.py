import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset


class InterleavedMapDataset(Dataset):
    def __init__(
        self,
        datasets: Sequence[Dataset],
        dataset_names: Sequence[str],
        probs: Sequence[float],
        seed: int = 42,
        strategy: str = "interleave_over",
    ) -> None:
        self.datasets = list(datasets)
        self.dataset_names = [str(x) for x in dataset_names]
        self.probs = [float(x) for x in probs]
        self.seed = int(seed)
        self.strategy = str(strategy)
        if len(self.datasets) == 0:
            raise ValueError("InterleavedMapDataset requires at least one dataset.")
        if len(self.datasets) != len(self.probs) or len(self.datasets) != len(self.dataset_names):
            raise ValueError("datasets, dataset_names and probs must have the same length.")
        if any(float(p) <= 0.0 for p in self.probs):
            raise ValueError("All interleave probabilities must be positive.")
        self.child_lengths = [int(len(ds)) for ds in self.datasets]
        if any(x <= 0 for x in self.child_lengths):
            raise ValueError("All child datasets in an interleave profile must be non-empty.")
        self.mapping = self._build_mapping()
        self.reference_dataset = self.datasets[0]

    def _build_mapping(self) -> List[Tuple[int, int]]:
        probs = np.asarray(self.probs, dtype=np.float64)
        probs = probs / float(probs.sum())
        lengths = np.asarray(self.child_lengths, dtype=np.int64)
        if self.strategy == "interleave_over":
            total_length = int(max(math.ceil(float(lengths[i]) / float(probs[i])) for i in range(len(lengths))))
        else:
            total_length = int(lengths.sum())
        target_counts = np.maximum(lengths, np.round(total_length * probs).astype(np.int64))
        remainder = int(total_length - int(target_counts.sum()))
        if remainder > 0:
            order = np.argsort(-probs)
            for i in range(remainder):
                target_counts[int(order[i % len(order)])] += 1

        rng = np.random.default_rng(self.seed)
        dataset_ids: List[int] = []
        for ds_idx, count in enumerate(target_counts.tolist()):
            dataset_ids.extend([int(ds_idx)] * int(count))
        rng.shuffle(dataset_ids)

        per_dataset_indices: Dict[int, List[int]] = {}
        for ds_idx, ds_len in enumerate(self.child_lengths):
            repeats = int(math.ceil(float(target_counts[ds_idx]) / float(ds_len)))
            seq = list(range(ds_len)) * repeats
            rng.shuffle(seq)
            per_dataset_indices[int(ds_idx)] = seq[: int(target_counts[ds_idx])]

        offsets = {int(i): 0 for i in range(len(self.datasets))}
        mapping: List[Tuple[int, int]] = []
        for ds_idx in dataset_ids:
            pos = offsets[int(ds_idx)]
            mapping.append((int(ds_idx), int(per_dataset_indices[int(ds_idx)][pos])))
            offsets[int(ds_idx)] = pos + 1
        return mapping

    def __len__(self) -> int:
        return len(self.mapping)

    def __getitem__(self, index: int):
        ds_idx, local_idx = self.mapping[int(index)]
        sample = dict(self.datasets[ds_idx][local_idx])
        sample["_mix_dataset_name"] = self.dataset_names[ds_idx]
        sample["_mix_dataset_index"] = int(ds_idx)
        return sample
