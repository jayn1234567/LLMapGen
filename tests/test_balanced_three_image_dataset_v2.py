import json
import tempfile
import unittest
from pathlib import Path


from scripts.tools.build_balanced_three_image_dataset_v2 import main


RATIOS = "empty=0.05,easy=0.25,medium=0.33,hard=0.27,very_hard=0.10"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def record(sample_id: str, split: str, stratum: str) -> dict:
    group = sample_id.split("_r", 1)[0]
    images = [
        f"images/{split}/{group}/{sample_id}.png",
        f"raw_lane_images/{split}/{group}/{sample_id}.png",
        f"pose_images/{split}/{group}/{sample_id}.png",
    ]
    return {
        "id": sample_id,
        "image": images[0],
        "images": images,
        "raw_lane_image": images[1],
        "pose_image": images[2],
        "meta": {
            "stratum": stratum,
            "difficulty": "easy" if stratum == "empty" else stratum,
            "difficulty_score": 0.0,
            "has_intersection": False,
            "pixel_patch_size": 256,
            "patch_width": 256,
            "patch_height": 256,
            "context_image_size": 512,
            "coord_mode": "norm1000",
            "coord_range": 1000,
        },
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n<image>\n<image>\nReturn lane_type and intersection_type JSON.",
            },
            {"from": "gpt", "value": '{"lines":[]}'},
        ],
    }


def materialize_fixture_record(root: Path, item: dict) -> None:
    for index, relative in enumerate(item["images"]):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes((index + 1,)))


def write_pool(root: Path, counts: dict[str, int]) -> None:
    phase = root / "phase_a"
    phase.mkdir(parents=True)
    train = []
    for stratum, count in counts.items():
        for index in range(count):
            item = record(f"pool_{stratum}_{index:04d}_r0_c0_p00", "train", stratum)
            materialize_fixture_record(root, item)
            train.append(item)
    eval_record = record("pool_eval_r0_c0_p00", "eval", "easy")
    test_record = record("pool_test_r0_c0_p00", "test", "medium")
    materialize_fixture_record(root, eval_record)
    materialize_fixture_record(root, test_record)
    splits = {"train": train, "eval": [eval_record], "test": [test_record]}
    for split, records in splits.items():
        text = "".join(json.dumps(item) + "\n" for item in records)
        (phase / f"{split}.jsonl").write_text(text, encoding="utf-8")
        (phase / f"meta_{split}.jsonl").write_text(text, encoding="utf-8")
    write_json(root / "dataset_info.json", {"variant": "fixture", "record_counts": {}})
    write_json(root / "split_manifest.json", {"counts": {key: len(value) for key, value in splits.items()}})
    write_json(root / "semantic_schema_report.json", {"status": "passed"})


def write_empty_donor_staging(clean_root: Path, aux_root: Path) -> None:
    clean_stage = clean_root / "00_source"
    aux_stage = aux_root / "00_source"
    marker_name = "stage_complete.json"
    write_json(clean_stage / marker_name, {
        "source_index": 0,
        "variants": ["context512_roi256"],
        "raw_lane_overlay": False,
        "semantic_validation_passed": True,
    })
    write_json(aux_stage / marker_name, {
        "source_index": 0,
        "variants": ["context512_roi256"],
        "raw_lane_overlay": True,
        "semantic_validation_passed": True,
    })
    item = record("donor_empty_r0_c0_p00", "train", "empty")
    clean_image = clean_stage / "variants" / "context512_roi256" / item["images"][0]
    clean_image.parent.mkdir(parents=True, exist_ok=True)
    clean_image.write_bytes(b"clean")
    for relative in item["images"][1:]:
        path = aux_stage / "variants" / "context512_roi256" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"aux")
    index = {
        "id": item["id"],
        "stratum": "empty",
        "difficulty": "easy",
        "difficulty_score": 0.0,
    }
    index_path = aux_stage / "records" / "train.index.jsonl"
    record_path = aux_stage / "records" / "context512_roi256" / "train.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index) + "\n", encoding="utf-8")
    record_path.write_text(json.dumps(item) + "\n", encoding="utf-8")


class BalancedThreeImageDatasetV2Test(unittest.TestCase):
    def test_exact_difficulty_quotas(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pool = root / "pool"
            output = root / "balanced"
            write_pool(pool, {
                "empty": 8,
                "easy": 30,
                "medium": 40,
                "hard": 30,
                "very_hard": 15,
            })

            main([
                "--input-root", str(pool),
                "--output-root", str(output),
                "--train-target-samples", "100",
                "--difficulty-ratios", RATIOS,
                "--copy-mode", "copy",
            ])

            report = json.loads((output / "balance_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["selected"], {
                "empty": 5,
                "easy": 25,
                "medium": 33,
                "hard": 27,
                "very_hard": 10,
            })
            self.assertEqual(report["record_counts"], {"train": 100, "eval": 1, "test": 1})
            records = [
                json.loads(line)
                for line in (output / "phase_a" / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 100)
            self.assertEqual(len({item["id"] for item in records}), 100)
            self.assertTrue(all((output / path).is_file() for item in records for path in item["images"]))

    def test_shortage_fails_with_preflight_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pool = root / "pool"
            output = root / "balanced"
            write_pool(pool, {
                "empty": 4,
                "easy": 25,
                "medium": 33,
                "hard": 27,
                "very_hard": 10,
            })

            with self.assertRaisesRegex(ValueError, "strict difficulty quotas"):
                main([
                    "--input-root", str(pool),
                    "--output-root", str(output),
                    "--train-target-samples", "100",
                    "--difficulty-ratios", RATIOS,
                    "--copy-mode", "copy",
                ])

            report = json.loads((output / "balance_preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(report["deficits"], {"empty": 1})

    def test_empty_shortage_is_filled_from_paired_staging(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pool = root / "pool"
            output = root / "balanced"
            clean = root / "clean_stage"
            aux = root / "aux_stage"
            write_pool(pool, {
                "empty": 4,
                "easy": 25,
                "medium": 33,
                "hard": 27,
                "very_hard": 10,
            })
            write_empty_donor_staging(clean, aux)

            main([
                "--input-root", str(pool),
                "--output-root", str(output),
                "--train-target-samples", "100",
                "--difficulty-ratios", RATIOS,
                "--copy-mode", "copy",
                "--empty-donor-clean-staging-root", str(clean),
                "--empty-donor-aux-staging-root", str(aux),
            ])

            report = json.loads((output / "balance_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["selected"]["empty"], 5)
            self.assertEqual(report["empty_donor"]["selected"], 1)
            train_text = (output / "phase_a" / "train.jsonl").read_text(encoding="utf-8")
            self.assertIn("donor_empty_r0_c0_p00", train_text)


if __name__ == "__main__":
    unittest.main()
