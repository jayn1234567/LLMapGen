import argparse
import json
import os

from PIL import Image, ImageDraw

from unimapgen.utils import ensure_dir


COLORS = {
    "divider": (255, 128, 0),
    "boundary": (0, 255, 0),
    "ped_crossing": (255, 0, 255),
    "centerline": (0, 170, 255),
}


def draw_lines(img, lines, width=2):
    canvas = img.copy()
    dr = ImageDraw.Draw(canvas)
    for line in lines:
        pts = line.get("points", [])
        if len(pts) < 2:
            continue
        cat = line.get("category", "unknown")
        color = COLORS.get(cat, (255, 255, 255))
        pts_xy = [(float(p[0]), float(p[1])) for p in pts]
        dr.line(pts_xy, fill=color, width=width)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_json", type=str, required=True)
    parser.add_argument("--sat_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/vis_v1")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    with open(args.pred_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    for i, item in enumerate(items[: args.limit]):
        token = item["token"]
        sat_path = os.path.join(args.sat_root, f"{token}_satellite.png")
        if not os.path.exists(sat_path):
            continue
        sat = Image.open(sat_path).convert("RGB")
        gt = draw_lines(sat, item.get("gt_lines", []), width=2)
        pred = draw_lines(sat, item.get("pred_lines", []), width=2)
        gt.save(os.path.join(args.out_dir, f"{i:03d}_{token}_gt.png"))
        pred.save(os.path.join(args.out_dir, f"{i:03d}_{token}_pred.png"))
    print(f"saved visualizations to {args.out_dir}")


if __name__ == "__main__":
    main()
