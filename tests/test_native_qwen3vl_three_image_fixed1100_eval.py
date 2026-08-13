from pathlib import Path
import tempfile
import unittest

from scripts.tools.compare_fixed1100_patch_metrics import compare, render_markdown


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "scripts/qwen3vl_native/test/"
    "test_local_stage_a_lane_intersection_three_image_local256_800k_qwen3vl8b_lora_fixed1100_npu.sh"
)


def sample_eval(lane_f1: float, intersection_f1: float, area_iou: float):
    return {
        "centerline_eval": {
            "instance_pre": lane_f1,
            "instance_recall": lane_f1,
            "instance_f1": lane_f1,
            "length_pre": lane_f1,
            "length_recall": lane_f1,
            "length_f1": lane_f1,
        },
        "intersection_eval": {
            "instance_pre": intersection_f1,
            "instance_recall": intersection_f1,
            "instance_f1": intersection_f1,
            "micro_area_iou": area_iou,
            "mean_matched_iou": area_iou,
        },
        "lane_intersection_eval": {"instance_f1": lane_f1, "length_f1": lane_f1},
        "lane_type_eval": {"status": "evaluated", "matched_type_accuracy": lane_f1},
        "intersection_type_eval": {
            "status": "evaluated",
            "matched_type_accuracy": intersection_f1,
        },
    }


class NativeThreeImageFixed1100ScriptTest(unittest.TestCase):
    def test_formal_launcher_contract(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("checkpoint-36000/", text)
        self.assertIn("checkpoint-50000/", text)
        self.assertIn("ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}", text)
        self.assertIn("NPROC_PER_NODE=${NPROC_PER_NODE:-4}", text)
        self.assertIn("PER_DEVICE_INFER_BATCH_SIZE=${PER_DEVICE_INFER_BATCH_SIZE:-32}", text)
        self.assertIn("easy=300,medium=300,hard=300,very_hard=200", text)
        self.assertIn("FIXED_EVAL_SEED=${FIXED_EVAL_SEED:-42}", text)
        self.assertIn("len(images) != 3", text)
        self.assertIn('prompt.count("<image>") != 3', text)
        self.assertIn("source_jsonl_sha256", text)
        self.assertIn("split_sha256", text)
        self.assertIn("tar -xf \"${DATASET_ARCHIVE_PATH}\" -C \"${DATASET_EXTRACT_ROOT}\" -T", text)
        self.assertIn("python -m mllm.native_qwen3vl.infer", text)
        self.assertIn("--per-device-infer-batch-size", text)
        self.assertIn("split_single_pass_eval_by_difficulty.py", text)
        self.assertIn("compare_fixed1100_patch_metrics.py", text)
        self.assertIn("DI_throughput:", text)

    def test_comparison_prefers_higher_balanced_geometry_score(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoints = []
            for label, metrics in (
                ("checkpoint-36000", sample_eval(0.80, 0.70, 0.60)),
                ("checkpoint-50000", sample_eval(0.82, 0.74, 0.69)),
            ):
                metrics_root = root / label / "by_difficulty"
                for bucket in ("all_selected", "easy", "medium", "hard", "very_hard"):
                    bucket_root = metrics_root / bucket
                    bucket_root.mkdir(parents=True)
                    (bucket_root / "eval.json").write_text(
                        __import__("json").dumps(metrics), encoding="utf-8"
                    )
                checkpoints.append((label, metrics_root))

            payload = compare(checkpoints)
            self.assertEqual(payload["recommended_by_primary_score"], ["checkpoint-50000"])
            self.assertEqual(
                payload["winners"]["all_selected"]["intersection_eval"]["micro_area_iou"][
                    "checkpoints"
                ],
                ["checkpoint-50000"],
            )
            report = render_markdown(payload)
            self.assertIn("Intersection micro area IoU", report)
            self.assertIn(
                "Recommended by the declared balanced-score policy: checkpoint-50000",
                report,
            )


if __name__ == "__main__":
    unittest.main()
