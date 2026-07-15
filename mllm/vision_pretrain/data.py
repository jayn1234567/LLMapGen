from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
PATCH_SUFFIX_PATTERNS = (
    re.compile(r"(?:_r\d+_c\d+)$", re.IGNORECASE),
    re.compile(r"(?:_row\d+_col\d+)$", re.IGNORECASE),
)


@dataclass(frozen=True)
class SegmentationSample:
    source: str
    image_path: str
    mask_path: str
    sample_id: str
    group_id: str


@dataclass(frozen=True)
class DatasetDiscoveryReport:
    roots: tuple[str, ...]
    total_samples: int
    train_samples: int
    val_samples: int
    groups: int
    missing_images: int
    val_fraction: float
    split_seed: int


def _resolve_dataset_root(root: str | Path) -> Path:
    path = Path(root).expanduser().resolve()
    if (path / "labels_lane").is_dir() and (path / "images").is_dir():
        return path
    train_path = path / "train"
    if (train_path / "labels_lane").is_dir() and (train_path / "images").is_dir():
        return train_path
    raise FileNotFoundError(
        f"Dataset root must contain images/ and labels_lane/, either directly or under train/: {path}"
    )


def infer_group_id(sample_stem: str) -> str:
    group_id = sample_stem
    for pattern in PATCH_SUFFIX_PATTERNS:
        group_id = pattern.sub("", group_id)
    return group_id or sample_stem


def _split_value(group_id: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{group_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64)


def _find_image(images_dir: Path, relative_mask: Path) -> Path | None:
    direct = images_dir / relative_mask
    if direct.is_file():
        return direct
    parent = direct.parent
    stem = relative_mask.stem
    for extension in IMAGE_EXTENSIONS:
        candidate = parent / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def discover_segmentation_samples(
    roots: Sequence[str | Path],
    *,
    val_fraction: float = 0.1,
    split_seed: int = 42,
    split_strategy: str = "hash_group",
    max_train_samples: int = 0,
    max_val_samples: int = 0,
) -> tuple[list[SegmentationSample], list[SegmentationSample], DatasetDiscoveryReport]:
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    split_strategy = str(split_strategy or "hash_group").strip().lower()
    if split_strategy not in {"hash_group", "ordered_per_root"}:
        raise ValueError(
            f"Unsupported split_strategy={split_strategy!r}; expected hash_group or ordered_per_root."
        )

    resolved_roots = [_resolve_dataset_root(root) for root in roots]
    samples: list[SegmentationSample] = []
    samples_by_root: list[list[SegmentationSample]] = []
    missing_images = 0
    for root in resolved_roots:
        labels_dir = root / "labels_lane"
        images_dir = root / "images"
        source = root.parent.name if root.name == "train" else root.name
        masks = sorted(
            path
            for path in labels_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        root_samples: list[SegmentationSample] = []
        for mask_path in masks:
            relative_mask = mask_path.relative_to(labels_dir)
            image_path = _find_image(images_dir, relative_mask)
            if image_path is None:
                missing_images += 1
                continue
            sample_id = relative_mask.with_suffix("").as_posix()
            group_id = infer_group_id(relative_mask.stem)
            root_samples.append(
                SegmentationSample(
                    source=source,
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                    sample_id=f"{source}/{sample_id}",
                    group_id=group_id,
                )
            )
        root_samples.sort(key=lambda item: item.sample_id)
        samples.extend(root_samples)
        samples_by_root.append(root_samples)

    if not samples:
        raise RuntimeError(f"No paired images and labels_lane masks found under: {resolved_roots}")

    samples.sort(key=lambda item: item.sample_id)
    if split_strategy == "ordered_per_root":
        train_samples = []
        val_samples = []
        for root_samples in samples_by_root:
            if not root_samples:
                continue
            val_count = max(1, round(len(root_samples) * float(val_fraction)))
            train_count = max(0, len(root_samples) - val_count)
            if train_count == 0 and len(root_samples) > 1:
                train_count = len(root_samples) - 1
            train_samples.extend(root_samples[:train_count])
            val_samples.extend(root_samples[train_count:])
    else:
        train_samples = [
            sample for sample in samples if _split_value(sample.group_id, split_seed) >= val_fraction
        ]
        val_samples = [
            sample for sample in samples if _split_value(sample.group_id, split_seed) < val_fraction
        ]
    if not train_samples or not val_samples:
        groups = sorted({sample.group_id for sample in samples})
        if len(groups) < 2:
            raise RuntimeError("At least two distinct sample groups are required for a train/validation split.")
        val_group_count = max(1, min(len(groups) - 1, round(len(groups) * val_fraction)))
        ordered_groups = sorted(groups, key=lambda group: _split_value(group, split_seed))
        val_groups = set(ordered_groups[:val_group_count])
        train_samples = [sample for sample in samples if sample.group_id not in val_groups]
        val_samples = [sample for sample in samples if sample.group_id in val_groups]

    if max_train_samples > 0:
        train_samples = train_samples[: int(max_train_samples)]
    if max_val_samples > 0:
        val_samples = val_samples[: int(max_val_samples)]

    report = DatasetDiscoveryReport(
        roots=tuple(str(path) for path in resolved_roots),
        total_samples=len(samples),
        train_samples=len(train_samples),
        val_samples=len(val_samples),
        groups=len({sample.group_id for sample in samples}),
        missing_images=missing_images,
        val_fraction=float(val_fraction),
        split_seed=int(split_seed),
    )
    return train_samples, val_samples, report


def save_split_manifest(
    path: str | Path,
    train_samples: Sequence[SegmentationSample],
    val_samples: Sequence[SegmentationSample],
    report: DatasetDiscoveryReport,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "report": asdict(report),
        "train": [asdict(sample) for sample in train_samples],
        "val": [asdict(sample) for sample in val_samples],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


class RoadLaneSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SegmentationSample],
        *,
        input_size: int,
        image_mean: Sequence[float],
        image_std: Sequence[float],
        augment: bool,
        ignore_mask_value: int | None = None,
    ) -> None:
        self.samples = list(samples)
        self.input_size = int(input_size)
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        self.augment = bool(augment)
        self.ignore_mask_value = ignore_mask_value
        if self.input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if len(image_mean) != 3 or len(image_std) != 3:
            raise ValueError("image_mean and image_std must each contain three values.")

    def __len__(self) -> int:
        return len(self.samples)

    def _augment_pair(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        if random.random() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
        rotations = random.randint(0, 3)
        if rotations:
            image = np.rot90(image, rotations, axes=(0, 1))
            mask = np.rot90(mask, rotations, axes=(0, 1))
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image_file:
            image = np.asarray(image_file.convert("RGB"))
        with Image.open(sample.mask_path) as mask_file:
            mask = np.asarray(mask_file)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image/mask size mismatch for {sample.sample_id}: image={image.shape[:2]} mask={mask.shape[:2]}"
            )
        if self.augment:
            image, mask = self._augment_pair(image, mask)

        image_pil = Image.fromarray(image).resize(
            (self.input_size, self.input_size),
            resample=Image.Resampling.BICUBIC,
        )
        mask_pil = Image.fromarray(mask).resize(
            (self.input_size, self.input_size),
            resample=Image.Resampling.NEAREST,
        )
        image_array = np.asarray(image_pil, dtype=np.float32) / 255.0
        raw_mask = np.asarray(mask_pil)
        target = (raw_mask > 0).astype(np.int64)
        if self.ignore_mask_value is not None:
            target[raw_mask == int(self.ignore_mask_value)] = 255

        pixel_values = torch.from_numpy(image_array.transpose(2, 0, 1)).float()
        pixel_values = (pixel_values - self.image_mean) / self.image_std
        return {
            "pixel_values": pixel_values,
            "labels": torch.from_numpy(target).long(),
            "sample_id": sample.sample_id,
        }


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
