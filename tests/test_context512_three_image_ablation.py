import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from scripts.tools import build_context512_roi256_three_image_ablation_windows as ablation


@unittest.skipUnless(ablation.box is not None, "rotation tests require shapely")
class NeighborRotationDatasetTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, list[dict]]:
        dataset = root / "input"
        records = []
        train_positions = [
            ("sample_00", 0, 0),
            ("sample_10", 256, 0),
            ("sample_01", 0, 256),
            ("sample_11", 256, 256),
            ("sample_20", 512, 0),
        ]
        for sample_id, x0, y0 in train_positions:
            records.append(self._make_record(dataset, "train", sample_id, x0, y0))
        eval_record = self._make_record(dataset, "eval", "eval_00", 0, 0)
        test_record = self._make_record(dataset, "test", "test_00", 0, 0)
        for split, rows in (("train", records), ("eval", [eval_record]), ("test", [test_record])):
            phase = dataset / "phase_a"
            phase.mkdir(parents=True, exist_ok=True)
            (phase / f"{split}.jsonl").write_text(
                "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
                encoding="utf-8",
                newline="\n",
            )
            (phase / f"meta_{split}.jsonl").write_text(
                "".join(
                    json.dumps(
                        {"id": row["id"], "image": row["image"], "images": row["images"], "meta": row["meta"]},
                        separators=(",", ":"),
                    )
                    + "\n"
                    for row in rows
                ),
                encoding="utf-8",
                newline="\n",
            )
        (dataset / "dataset_info.json").write_text(
            json.dumps(
                {
                    "train_stride": 256,
                    "eval_test_stride": 256,
                    "context_size": 512,
                    "target_patch_size": 256,
                    "coord_mode": "norm1000",
                    "three_image_input": True,
                }
            ),
            encoding="utf-8",
        )
        (dataset / "split_manifest.json").write_text("{}", encoding="utf-8")
        return dataset, records

    @staticmethod
    def _make_record(dataset: Path, split: str, sample_id: str, x0: int, y0: int) -> dict:
        image_paths = [
            f"images/{split}/{sample_id}.png",
            f"raw_lane_images/{split}/{sample_id}.png",
            f"pose_images/{split}/{sample_id}.png",
        ]
        colors = [(30, 40, 50), (60, 70, 80), (90, 100, 110)]
        for relative, color in zip(image_paths, colors):
            path = dataset / Path(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (512, 512), color).save(path, format="PNG")
        target = {
            "lines": [
                {
                    "category": "centerline",
                    "lane_type": "common",
                    "start_type": "cut",
                    "end_type": "cut",
                    "points": [[0, 500], [1000, 500]],
                }
            ]
        }
        meta = {
            "x0": x0,
            "y0": y0,
            "target_size": 256,
            "context_image_size": 512,
            "coord_mode": "norm1000",
            "coord_range": 1000,
        }
        return {
            "id": sample_id,
            "image": image_paths[0],
            "images": image_paths,
            "raw_lane_image": image_paths[1],
            "pose_image": image_paths[2],
            "meta": meta,
            "conversations": [
                {"from": "human", "value": "<image> <image> <image>"},
                {"from": "gpt", "value": json.dumps(target, separators=(",", ":"))},
            ],
        }

    def test_neighbor_rotation_replaces_rows_and_keeps_stride_256(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_root, source_rows = self._write_fixture(Path(temp_dir))
            output_root = Path(temp_dir) / "neighbor_rotation"
            package_path = Path(temp_dir) / "neighbor_rotation.tar"
            args = SimpleNamespace(
                mode="neighbor_rotation",
                input_root=str(input_root),
                output_root=str(output_root),
                angles="45,135",
                neighbor_angles="0,45,135",
                neighbor_grid_stride=256,
                copy_mode="copy",
                image_resample="bilinear",
                png_compress_level=4,
                package_path=str(package_path),
                validation_sample_limit=100,
                progress_every=0,
                rotate_empty=False,
                resume=False,
                skip_package=False,
            )

            summary = ablation.build(args)

            self.assertEqual(summary["status"], "built")
            train_path = output_root / "phase_a" / "train.jsonl"
            output_rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(output_rows), len(source_rows))
            self.assertEqual(
                {row["id"] for row in output_rows},
                {row["id"] for row in source_rows},
            )

            angles = {
                row["id"]: row["meta"]["neighbor_rotation_applied_angle_degrees"]
                for row in output_rows
            }
            self.assertEqual(angles["sample_00"], 0.0)
            self.assertEqual(angles["sample_10"], 45.0)
            self.assertEqual(angles["sample_01"], 45.0)
            self.assertEqual(angles["sample_11"], 135.0)
            self.assertEqual(angles["sample_20"], 135.0)
            self.assertEqual(
                {row["meta"]["augmentation"] for row in output_rows},
                {"neighbor_rotation"},
            )
            self.assertTrue(
                all(row["conversations"][0]["value"].count("<image>") == 3 for row in output_rows)
            )

            info = json.loads((output_root / "dataset_info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["train_stride"], 256)
            self.assertFalse(info["ablation"]["rotated_rows_are_additive"])
            self.assertEqual(
                info["ablation"]["neighbor_rotation"]["phase_formula"],
                "(grid_x + grid_y) % 3",
            )
            self.assertTrue(package_path.is_file())
            self.assertTrue((output_root / "ablation_validation.json").is_file())
            self.assertTrue((output_root / "build_complete.json").is_file())

    def test_neighbor_rotation_rejects_non_256_aligned_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_root, _source_rows = self._write_fixture(Path(temp_dir))
            broken_path = input_root / "phase_a" / "train.jsonl"
            rows = [json.loads(line) for line in broken_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["meta"]["x0"] = 128
            broken_path.write_text(
                "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
                encoding="utf-8",
                newline="\n",
            )
            args = SimpleNamespace(
                mode="neighbor_rotation",
                input_root=str(input_root),
                output_root=str(Path(temp_dir) / "broken_output"),
                angles="45,135",
                neighbor_angles="0,45,135",
                neighbor_grid_stride=256,
                neighbor_source_grid_policy="require",
                copy_mode="copy",
                image_resample="bilinear",
                png_compress_level=4,
                package_path=str(Path(temp_dir) / "broken_output.tar"),
                validation_sample_limit=0,
                progress_every=0,
                rotate_empty=False,
                resume=False,
                skip_package=True,
            )
            with self.assertRaisesRegex(ValueError, "origins aligned"):
                ablation.build(args)

    def test_neighbor_rotation_filters_unaligned_rows_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_root, source_rows = self._write_fixture(Path(temp_dir))
            info_path = input_root / "dataset_info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["train_stride"] = 128
            info_path.write_text(
                json.dumps(info, separators=(",", ":")),
                encoding="utf-8",
                newline="\n",
            )
            train_path = input_root / "phase_a" / "train.jsonl"
            rows = [
                json.loads(line)
                for line in train_path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["meta"]["x0"] = 128
            train_path.write_text(
                "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
                encoding="utf-8",
                newline="\n",
            )
            output_root = Path(temp_dir) / "filtered_output"
            args = SimpleNamespace(
                mode="neighbor_rotation",
                input_root=str(input_root),
                output_root=str(output_root),
                angles="45,135",
                neighbor_angles="0,45,135",
                neighbor_grid_stride=256,
                copy_mode="copy",
                image_resample="bilinear",
                png_compress_level=4,
                package_path=str(Path(temp_dir) / "filtered_output.tar"),
                validation_sample_limit=0,
                progress_every=0,
                rotate_empty=False,
                resume=False,
                skip_package=True,
            )

            summary = ablation.build(args)

            self.assertEqual(summary["split_counts"]["train"], len(source_rows) - 1)
            output_ids = {
                json.loads(line)["id"]
                for line in (output_root / "phase_a" / "train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            }
            self.assertNotIn("sample_00", output_ids)
            balance = json.loads(
                (output_root / "balance_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(balance["neighbor_source_train_stride"], 128)
            self.assertEqual(balance["neighbor_filtered_train_samples"], 1)


if __name__ == "__main__":
    unittest.main()
