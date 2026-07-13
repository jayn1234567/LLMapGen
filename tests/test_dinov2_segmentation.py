import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from mllm.vision_pretrain.data import (
    RoadLaneSegmentationDataset,
    discover_segmentation_samples,
    infer_group_id,
)
from mllm.vision_pretrain.dinov2_segmentation import Dinov2RoadSegmentationModel
from mllm.vision_pretrain.metrics import confusion_matrix, metrics_from_confusion


class FakeVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(patch_size=2, hidden_size=8, num_register_tokens=0)
        self.patch_embed = nn.Conv2d(3, 8, kernel_size=2, stride=2)
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])

    def forward(
        self,
        pixel_values,
        output_hidden_states=True,
        return_dict=True,
        interpolate_pos_encoding=True,
    ):
        del output_hidden_states, return_dict, interpolate_pos_encoding
        patches = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        cls = patches.new_zeros((patches.shape[0], 1, patches.shape[2]))
        hidden = torch.cat((cls, patches), dim=1)
        hidden_states = [hidden]
        for layer in self.layers:
            hidden = layer(hidden)
            hidden_states.append(hidden)
        return SimpleNamespace(hidden_states=tuple(hidden_states))

    def save_pretrained(self, output_dir, state_dict=None, safe_serialization=True):
        del safe_serialization
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        torch.save(state_dict or self.state_dict(), Path(output_dir) / "pytorch_model.bin")


class Dinov2SegmentationTests(unittest.TestCase):
    def _make_dataset(self, root: Path, count: int = 30):
        images = root / "train" / "images"
        masks = root / "train" / "labels_lane"
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        for index in range(count):
            stem = f"scene_{index:03d}_r000_c000"
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            image[..., index % 3] = 100 + index
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:6, 3:5] = 255
            Image.fromarray(image).save(images / f"{stem}.png")
            Image.fromarray(mask).save(masks / f"{stem}.png")

    def test_group_inference_removes_patch_suffix(self):
        self.assertEqual(infer_group_id("tile_a_r012_c003"), "tile_a")

    def test_deterministic_dataset_split_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            self._make_dataset(root)
            train_a, val_a, report_a = discover_segmentation_samples(
                [root], val_fraction=0.2, split_seed=17
            )
            train_b, val_b, report_b = discover_segmentation_samples(
                [root], val_fraction=0.2, split_seed=17
            )
            self.assertEqual([item.sample_id for item in train_a], [item.sample_id for item in train_b])
            self.assertEqual([item.sample_id for item in val_a], [item.sample_id for item in val_b])
            self.assertEqual(report_a, report_b)
            self.assertEqual(report_a.total_samples, 30)
            self.assertTrue(train_a)
            self.assertTrue(val_a)
            dataset = RoadLaneSegmentationDataset(
                train_a,
                input_size=14,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
                augment=False,
            )
            item = dataset[0]
            self.assertEqual(tuple(item["pixel_values"].shape), (3, 14, 14))
            self.assertEqual(tuple(item["labels"].shape), (14, 14))
            self.assertEqual(set(item["labels"].unique().tolist()), {0, 1})

    def test_model_forward_uses_raw_two_class_logits(self):
        model = Dinov2RoadSegmentationModel(
            FakeVisionEncoder(),
            input_size=8,
            hidden_state_indices=[1, 2, 3, 4],
            projection_channels=16,
        )
        output = model(torch.randn(2, 3, 8, 8))
        self.assertEqual(tuple(output.shape), (2, 2, 8, 8))
        output.mean().backward()
        self.assertTrue(all(parameter.requires_grad for parameter in model.vision_encoder.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in model.vision_encoder.parameters()))

    def test_huggingface_dinov2_forward_contract(self):
        from transformers import Dinov2Config, Dinov2Model

        config = Dinov2Config(
            image_size=8,
            patch_size=2,
            num_channels=3,
            hidden_size=8,
            num_hidden_layers=4,
            num_attention_heads=2,
            intermediate_size=16,
        )
        model = Dinov2RoadSegmentationModel(
            Dinov2Model(config),
            input_size=8,
            hidden_state_indices=[1, 2, 3, 4],
            projection_channels=16,
        )
        output = model(torch.randn(1, 3, 8, 8))
        self.assertEqual(tuple(output.shape), (1, 2, 8, 8))
        output.square().mean().backward()
        self.assertIsNotNone(model.vision_encoder.embeddings.patch_embeddings.projection.weight.grad)

    def test_fixed_two_class_confusion_metrics(self):
        logits = torch.tensor(
            [[[[4.0, -1.0], [4.0, -1.0]], [[-1.0, 4.0], [-1.0, 4.0]]]]
        )
        labels = torch.tensor([[[0, 1], [1, 1]]])
        matrix = confusion_matrix(logits, labels)
        self.assertEqual(matrix.tolist(), [[1.0, 0.0], [1.0, 2.0]])
        metrics = metrics_from_confusion(matrix)
        self.assertAlmostEqual(metrics["lane_precision"], 1.0)
        self.assertAlmostEqual(metrics["lane_recall"], 2.0 / 3.0)

    def test_obs_source_manifest_has_all_private_segmentation_sets(self):
        script_path = (
            Path(__file__).parents[1] / "scripts" / "tools" / "download_rc_lane_segmentation_obs.py"
        )
        spec = importlib.util.spec_from_file_location("rc_seg_download", script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.assertEqual(len(module.DATASET_NAMES), 16)
        self.assertIn("1029_1153_label_refine_fix_rl", module.DATASET_NAMES)


if __name__ == "__main__":
    unittest.main()
