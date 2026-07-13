#!/usr/bin/env python3
"""Download the paired image/labels_lane datasets used by the private DINO training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


OBS_BASE = "obs://yw-ads-training-gy1/data/external/personal/q00649977/rc-lane-train-from0425"
DATASET_NAMES = (
    "1111_1153_label_refine_fix_rl",
    "1120_2889_label_refine_fix_rl",
    "gamma_208p_label_refine_1124",
    "0427_0901_label_refine_fix_rl",
    "0426_1935_label_refine",
    "0426_1639_label_refine",
    "right_turn_label_refine",
    "gamma_label_refine_0831",
    "gamma_144p_label_refine",
    "gamma_63p_label_refine",
    "gamma_167p_label_refine_1010",
    "gamma_224p_label_refine_1014",
    "0427_2100_label_refine_fix_rl",
    "gamma_187p_label_refine_1030",
    "1023_2143_label_refine_fix_rl",
    "1029_1153_label_refine_fix_rl",
)


@dataclass(frozen=True)
class DownloadedDataset:
    name: str
    obs_path: str
    local_path: str
    train_root: str
    reused: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="/cache/jn/data/rc_lane_segmentation")
    parser.add_argument("--limit", type=int, default=0, help="Download only the first N datasets; 0 uses all.")
    parser.add_argument("--only", nargs="*", default=None, help="Optional explicit dataset names.")
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--skip-download", action="store_true", help="Only validate already downloaded data.")
    return parser.parse_args()


def selected_names(args: argparse.Namespace) -> list[str]:
    names = list(args.only) if args.only else list(DATASET_NAMES)
    unknown = sorted(set(names) - set(DATASET_NAMES))
    if unknown:
        raise ValueError(f"Unknown dataset names: {unknown}")
    if args.limit > 0:
        names = names[: args.limit]
    if not names:
        raise ValueError("No datasets selected.")
    return names


def validate_train_root(local_path: Path) -> Path:
    candidates = (local_path / "train", local_path)
    for candidate in candidates:
        if (candidate / "images").is_dir() and (candidate / "labels_lane").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Downloaded dataset has no paired train/images and train/labels_lane directories: {local_path}"
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    names = selected_names(args)
    mox = None
    if not args.skip_download:
        try:
            import moxing as mox_module
        except ImportError as exc:
            raise RuntimeError("moxing is required to download OBS segmentation data.") from exc
        if not hasattr(mox_module, "file"):
            raise RuntimeError("The imported moxing package does not provide mox.file.")
        mox = mox_module

    downloaded: list[DownloadedDataset] = []
    for name in names:
        obs_path = f"{OBS_BASE}/{name}/"
        local_path = output_root / name
        marker = local_path / ".obs_download_complete.json"
        reused = marker.is_file()
        if not reused and not args.skip_download:
            local_path.mkdir(parents=True, exist_ok=True)
            print(f"[rc-seg-download] {obs_path} -> {local_path}", flush=True)
            mox.file.copy_parallel(obs_path, str(local_path), threads=int(args.threads))
        train_root = validate_train_root(local_path)
        if not reused and not args.skip_download:
            marker.write_text(
                json.dumps({"name": name, "obs_path": obs_path}, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
        downloaded.append(
            DownloadedDataset(
                name=name,
                obs_path=obs_path,
                local_path=str(local_path),
                train_root=str(train_root),
                reused=reused,
            )
        )

    roots_file = output_root / "train_roots.txt"
    roots_file.write_text(
        "".join(f"{item.train_root}\n" for item in downloaded),
        encoding="utf-8",
    )
    summary = {
        "output_root": str(output_root),
        "roots_file": str(roots_file),
        "datasets": [asdict(item) for item in downloaded],
    }
    (output_root / "download_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
