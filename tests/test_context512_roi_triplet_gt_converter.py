import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.tools.convert_context512_roi_triplet_gt_to_dataset_v2 import main, materialize_file


class Context512RoiTripletGtConverterTest(unittest.TestCase):
    def _write_triplet(self, root: Path, size: int) -> tuple[Path, str]:
        sample_id = "A0_demo_r0_c0_p00"
        group = "A0_demo"
        image_root = root / "images" / "train" / group
        image_root.mkdir(parents=True)
        names = [
            "r0_c0_p00.png",
            "r0_c0_p00_pose.png",
            "r0_c0_p00_raw_lane.png",
        ]
        colors = [(10, 20, 30), (255, 255, 255), (200, 200, 200)]
        for name, color in zip(names, colors):
            Image.new("RGB", (size, size), color).save(image_root / name)

        annotation = [{
            "id": sample_id,
            "GT": [
                {
                    "lane": json.dumps([
                        {"category": 1, "coords": [[0, 0], [128, 64], [255, 255]]},
                        {"category": 3, "coords": [[0, 10], [255, 10]]},
                    ])
                },
                {
                    "intersection": json.dumps([
                        {
                            "category": "1_1",
                            "coords": [[0, 0], [255, 0], [255, 255], [0, 255], [0, 255]],
                        }
                    ])
                },
                {"patch_size": 256},
            ],
            "utm_zone": 50,
            "image": [f"{group}/{name}" for name in names],
        }]
        annotation_path = root / "annotations" / "train" / f"{group}.json"
        annotation_path.parent.mkdir(parents=True)
        annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
        return annotation_path, sample_id

    def test_converts_triplet_and_norm1000_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            output = Path(temp_dir) / "converted"
            annotation_path, sample_id = self._write_triplet(root, 512)

            main([
                "--input-root", str(root),
                "--annotation-root", str(annotation_path.parents[1]),
                "--output-root", str(output),
                "--copy-mode", "copy",
                "--image-check-mode", "all",
            ])

            record = json.loads((output / "phase_a" / "train.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["id"], sample_id)
            self.assertEqual(len(record["images"]), 3)
            self.assertTrue(record["images"][0].startswith("images/train/"))
            self.assertTrue(record["images"][1].startswith("raw_lane_images/train/"))
            self.assertTrue(record["images"][2].startswith("pose_images/train/"))
            self.assertEqual(record["conversations"][0]["value"].count("<image>"), 3)

            target = json.loads(record["conversations"][1]["value"])
            centerlines = [item for item in target["lines"] if item["category"] == "centerline"]
            intersections = [item for item in target["lines"] if item["category"] == "intersection"]
            self.assertEqual(len(centerlines), 1)
            self.assertEqual(centerlines[0]["lane_type"], "common")
            self.assertIn([0, 0], centerlines[0]["points"])
            self.assertIn([1000, 1000], centerlines[0]["points"])
            self.assertEqual(len(intersections), 1)
            self.assertEqual(intersections[0]["intersection_type"], "common")
            self.assertEqual(intersections[0]["points"][0], intersections[0]["points"][-1])

            info = json.loads((output / "dataset_info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["target_patch_size"], 256)
            self.assertEqual(info["context_size"], 512)
            self.assertEqual(info["target_roi_in_image"], [128, 128, 384, 384])
            validation = json.loads((output / "conversion_validation.json").read_text(encoding="utf-8"))
            self.assertEqual(validation["status"], "passed")

    def test_rejects_non_context_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            output = Path(temp_dir) / "converted"
            annotation_path, _ = self._write_triplet(root, 256)

            with self.assertRaisesRegex(ValueError, "expected 512x512 context images"):
                main([
                    "--input-root", str(root),
                    "--annotation-root", str(annotation_path.parents[1]),
                    "--output-root", str(output),
                    "--copy-mode", "copy",
                    "--image-check-mode", "all",
                ])

    def test_resume_replaces_stale_same_size_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(b"clean")
            destination.write_bytes(b"wrong")

            mode = materialize_file(source, destination, "copy", resume=True)

            self.assertEqual(mode, "replaced_copy")
            self.assertEqual(destination.read_bytes(), b"clean")


if __name__ == "__main__":
    unittest.main()
