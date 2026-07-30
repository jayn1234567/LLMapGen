import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.tools.convert_intersection_prompt_eval_set import main
from scripts.tools.derive_intersection_prompt_dataset import extract_prompt_intersections, parse_target


def source_record(sample_id: str) -> dict:
    return {
        "id": sample_id,
        "image": f"images/eval/{sample_id}/{sample_id}.png",
        "meta": {
            "coord_mode": "norm1000",
            "coord_range": 1000,
            "pixel_patch_size": 512,
            "patch_width": 512,
            "patch_height": 512,
            "target_size": 512,
        },
        "conversations": [
            {"from": "human", "value": "<image>\nsource prompt"},
            {
                "from": "gpt",
                "value": json.dumps({
                    "lines": [
                        {
                            "category": "centerline",
                            "lane_type": "common",
                            "start_type": "cut",
                            "end_type": "cut",
                            "points": [[0, 500], [1000, 500]],
                        },
                        {
                            "category": "intersection",
                            "intersection_type": "t_intersection",
                            "is_cut": False,
                            "points": [[200, 200], [200, 800], [800, 800], [800, 200], [200, 200]],
                        },
                    ]
                }),
            },
        ],
    }


class IntersectionPromptEvalSetTest(unittest.TestCase):
    def test_conversion_preserves_ids_and_moves_intersections_to_prompt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "fixed"
            output_root = root / "converted"
            source_root.mkdir()
            ordered_ids = []
            for difficulty in ("easy", "medium", "hard", "very_hard"):
                sample_id = f"{difficulty}_sample"
                ordered_ids.append(sample_id)
                (source_root / f"{difficulty}.jsonl").write_text(
                    json.dumps(source_record(sample_id)) + "\n",
                    encoding="utf-8",
                )
            (source_root / "all_selected.jsonl").write_text(
                "".join(json.dumps(source_record(sample_id)) + "\n" for sample_id in ordered_ids),
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "convert_intersection_prompt_eval_set.py",
                    "--input-root",
                    str(source_root),
                    "--output-root",
                    str(output_root),
                ],
            ):
                main()

            converted = [
                json.loads(line)
                for line in (output_root / "all_selected.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["id"] for record in converted], ordered_ids)
            for record in converted:
                target = parse_target(record)["lines"]
                self.assertTrue(target)
                self.assertTrue(all(item["category"] == "centerline" for item in target))
                prompt_text = record["conversations"][0]["value"]
                intersections = extract_prompt_intersections(prompt_text)
                self.assertEqual(len(intersections), 1)
                self.assertEqual(intersections[0]["intersection_type"], "t_intersection")
                self.assertEqual(record["meta"]["task_mode"], "centerline_conditioned_on_gt_intersections")


if __name__ == "__main__":
    unittest.main()
