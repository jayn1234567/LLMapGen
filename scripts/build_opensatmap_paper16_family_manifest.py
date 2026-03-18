import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 16-patch paper-style family manifests from raw OpenSatMap 4096 images.")
    parser.add_argument("--opensatmap-root", type=str, required=True)
    parser.add_argument("--ann-json", type=str, default=None)
    parser.add_argument("--output-manifest", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--crop-size", type=int, default=896)
    parser.add_argument("--base-start", type=int, default=448)
    parser.add_argument("--base-stride", type=int, default=664)
    parser.add_argument("--axis-count", type=int, default=5)
    parser.add_argument("--family-grid-size", type=int, default=4)
    parser.add_argument("--max-images-per-split", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_centers(base_start: int, base_stride: int, axis_count: int) -> List[int]:
    return [int(base_start + base_stride * i) for i in range(axis_count)]


def build_patch_record(
    patch_id: int,
    row: int,
    col: int,
    center_x: int,
    center_y: int,
    crop_size: int,
) -> Dict:
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
    args = parse_args()
    opensatmap_root = Path(args.opensatmap_root).resolve()
    ann_json = Path(args.ann_json).resolve() if args.ann_json else opensatmap_root / "annotrainval20.json"
    output_manifest = Path(args.output_manifest).resolve()
    annotations = load_json(ann_json)
    centers = build_centers(
        base_start=int(args.base_start),
        base_stride=int(args.base_stride),
        axis_count=int(args.axis_count),
    )

    families: List[Dict] = []
    for split in args.splits:
        split_dir = opensatmap_root / "picuse20trainvaltest" / str(split)
        image_names = sorted(x.name for x in split_dir.iterdir() if x.is_file())
        count = 0
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
            count += 1
            if int(args.max_images_per_split) > 0 and count >= int(args.max_images_per_split):
                break

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
