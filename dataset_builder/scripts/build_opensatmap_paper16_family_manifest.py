"""Build a paper16 family manifest from raw OpenSatMap-style 4096 images.

This script only defines the cropping plan. It does not crop images or
serialize training targets. The output manifest is later consumed by the
patch-only exporter.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for manifest building.

    Parameters exposed on the CLI:
    - `--opensatmap-root`: raw dataset root containing split image folders.
    - `--ann-json`: annotation json used only to filter valid images.
    - `--output-manifest`: destination jsonl path for all family records.
    - `--splits`: raw split names to scan, usually `train val`.
    - `--crop-size`: patch side length in pixels.
    - `--base-start`: first patch-center coordinate on each axis.
    - `--base-stride`: stride between adjacent patch centers.
    - `--axis-count`: number of patch centers per axis.
    - `--family-grid-size`: number of patches per side inside one family.
    """
    parser = argparse.ArgumentParser(description="Build 16-patch paper-style family manifests from raw OpenSatMap 4096 images.")
    parser.add_argument("--opensatmap-root", type=str, required=True, help="Raw OpenSatMap root. Expected to contain picuse20trainvaltest/<split>.")
    parser.add_argument("--ann-json", type=str, default=None, help="Annotation json path. Defaults to <opensatmap-root>/annotrainval20.json.")
    parser.add_argument("--output-manifest", type=str, required=True, help="Output jsonl manifest path.")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"], help="Raw split folders to scan under picuse20trainvaltest.")
    parser.add_argument("--crop-size", type=int, default=896, help="Patch side length in pixels.")
    parser.add_argument("--base-start", type=int, default=448, help="Center coordinate of the first patch on each axis.")
    parser.add_argument("--base-stride", type=int, default=664, help="Distance between adjacent patch centers.")
    parser.add_argument("--axis-count", type=int, default=5, help="How many patch centers to place on each axis.")
    parser.add_argument("--family-grid-size", type=int, default=4, help="Family width/height in patches. 4 means one family contains 16 patches.")
    return parser.parse_args()


def load_json(path: Path):
    """Load the annotation json used to decide which images are valid."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    """Write family records to a jsonl manifest and return the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_centers(base_start: int, base_stride: int, axis_count: int) -> List[int]:
    """Build the ordered patch-center coordinates for one image axis."""
    return [int(base_start + base_stride * i) for i in range(axis_count)]


def build_patch_record(
    patch_id: int,
    row: int,
    col: int,
    center_x: int,
    center_y: int,
    crop_size: int,
) -> Dict:
    """Build one patch description inside a family.

    Parameters:
    - `patch_id`: row-major patch index inside the family.
    - `row` / `col`: patch coordinates inside the family-local 4x4 grid.
    - `center_x` / `center_y`: patch center in raw-image coordinates.
    - `crop_size`: patch side length used to derive the crop box.
    """
    half = crop_size // 2
    return {
        "patch_id": int(patch_id),
        "row": int(row),
        "col": int(col),
        "center_x": int(center_x),
        "center_y": int(center_y),
        "crop_box": {
            "x_min": int(center_x - half),
            "y_min": int(center_y - half),
            "x_max": int(center_x + half),
            "y_max": int(center_y + half),
            "center_x": int(center_x),
            "center_y": int(center_y),
        },
    }


def build_families_for_image(
    image_name: str,
    split: str,
    image_path: Path,
    crop_size: int,
    centers: List[int],
    family_grid_size: int,
) -> List[Dict]:
    """Enumerate all sliding paper16 families for one raw image.

    The script first defines a larger center grid, then extracts every
    contiguous `family_grid_size x family_grid_size` sub-grid as one family.
    """
    max_start = len(centers) - family_grid_size
    families: List[Dict] = []
    for row0 in range(max_start + 1):
        for col0 in range(max_start + 1):
            patches: List[Dict] = []
            for row in range(family_grid_size):
                for col in range(family_grid_size):
                    center_x = centers[col0 + col]
                    center_y = centers[row0 + row]
                    patch_id = row * family_grid_size + col
                    patches.append(
                        build_patch_record(
                            patch_id=patch_id,
                            row=row,
                            col=col,
                            center_x=center_x,
                            center_y=center_y,
                            crop_size=crop_size,
                        )
                    )
            family_id = f"{Path(image_name).stem}__paper16_r{row0}_c{col0}"
            families.append(
                {
                    "family_id": family_id,
                    "split": split,
                    "source_image": image_name,
                    "source_image_path": str(image_path),
                    "image_size": [4096, 4096],
                    "crop_size": int(crop_size),
                    "paper_grid": {
                        "base_start": int(centers[0]),
                        "base_stride": int(centers[1] - centers[0]) if len(centers) > 1 else 0,
                        "axis_count": int(len(centers)),
                        "family_grid_size": int(family_grid_size),
                        "row0": int(row0),
                        "col0": int(col0),
                    },
                    "patches": patches,
                }
            )
    return families


def main() -> None:
    """Scan raw splits, build all family records, and write the manifest."""
    args = parse_args()
    opensatmap_root = Path(args.opensatmap_root).resolve()
    ann_json = Path(args.ann_json).resolve() if args.ann_json else opensatmap_root / "annotrainval20.json"
    output_manifest = Path(args.output_manifest).resolve()
    # 这里只用 annotation json 做“图片是否有标注”的过滤，不在这一层解析线几何。
    annotations = load_json(ann_json)
    # 先按默认 paper16 几何参数生成一维中心坐标，再在 x/y 两轴上组成 5x5 center grid。
    centers = build_centers(
        base_start=int(args.base_start),
        base_stride=int(args.base_stride),
        axis_count=int(args.axis_count),
    )

    families: List[Dict] = []
    for split in args.splits:
        split_dir = opensatmap_root / "picuse20trainvaltest" / str(split)
        image_names = sorted(x.name for x in split_dir.iterdir() if x.is_file())
        for image_name in image_names:
            if image_name not in annotations:
                continue
            image_path = split_dir / image_name
            families.extend(
                build_families_for_image(
                    image_name=image_name,
                    split=str(split),
                    image_path=image_path,
                    crop_size=int(args.crop_size),
                    centers=centers,
                    family_grid_size=int(args.family_grid_size),
                )
            )

    total = write_jsonl(output_manifest, families)
    summary = {
        "opensatmap_root": str(opensatmap_root),
        "ann_json": str(ann_json),
        "output_manifest": str(output_manifest),
        "splits": [str(x) for x in args.splits],
        "crop_size": int(args.crop_size),
        "base_start": int(args.base_start),
        "base_stride": int(args.base_stride),
        "axis_count": int(args.axis_count),
        "family_grid_size": int(args.family_grid_size),
        "num_families": int(total),
    }
    summary_path = output_manifest.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Built {total} families")
    print(f"Manifest: {output_manifest}")


if __name__ == "__main__":
    main()
