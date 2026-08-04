import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROLE_TEXT = """<image>
<image>
<image>
The first image is the clean BEV road-structure image.
The second image is a lane image predicted by a PV camera model: white lines are predicted lanes on a black background. Do not copy it blindly when it conflicts with the visible BEV evidence.
The third image is a historical vehicle-trajectory image: white lines are historical vehicle trajectories on a black background.
Return only valid JSON in the form {"lines":[...]} with no extra explanation.
"""


def write_dataset(root: Path, *, swap_auxiliary_images: bool = False) -> None:
    for split in ("train", "eval", "test"):
        paths = [
            f"images/{split}/sample.png",
            f"raw_lane_images/{split}/sample.png",
            f"pose_images/{split}/sample.png",
        ]
        for relative in paths:
            image_path = root / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (256, 256), color=(0, 0, 0)).save(image_path)
        images = list(paths)
        if swap_auxiliary_images:
            images[1], images[2] = images[2], images[1]
        record = {
            "id": f"{split}-sample",
            "image": images[0],
            "images": images,
            "raw_lane_image": images[1],
            "pose_image": images[2],
            "meta": {
                "raw_lane_overlay": False,
                "raw_lane_separate_image": True,
                "input_image_roles": [
                    "bev_road_structure",
                    "pv_camera_raw_lane",
                    "historical_vehicle_trajectory",
                ],
            },
            "conversations": [
                {"from": "human", "value": ROLE_TEXT},
                {
                    "from": "gpt",
                    "value": json.dumps(
                        {
                            "lines": [
                                {
                                    "category": "centerline",
                                    "lane_type": "common",
                                    "points": [[0, 0], [1000, 1000]],
                                }
                            ]
                        }
                    ),
                },
            ],
        }
        phase = root / "phase_a"
        phase.mkdir(parents=True, exist_ok=True)
        (phase / f"{split}.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )


def run_inspector(root: Path, report: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/tools/inspect_lane_intersection_training_dataset.py",
            "--dataset-root",
            str(root),
            "--require-three-image-rawlane-pose",
            "--expected-image-size",
            "256",
            "--image-checks-per-split",
            "1",
            "--strict",
            "--report",
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_three_image_contract_passes(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    report = tmp_path / "report.json"
    write_dataset(dataset)

    result = run_inspector(dataset, report)

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["splits"]["train"]["checked_image_sizes"] == {"256x256": 3}


def test_three_image_contract_rejects_swapped_auxiliary_images(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    report = tmp_path / "report.json"
    write_dataset(dataset, swap_auxiliary_images=True)

    result = run_inspector(dataset, report)

    assert result.returncode != 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "image order/prefix" in json.dumps(payload["failures"])
