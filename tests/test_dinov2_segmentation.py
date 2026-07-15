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
from mllm.vision_pretrain.dinov2_segmentation import (
    Dinov2RoadSegmentationModel,
    merged_lora_state_dict,
)
from mllm.vision_pretrain.metrics import confusion_matrix, metrics_from_confusion


class FakeVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(patch_size=2, hidden_size=8, num_register_tokens=0)
        self.patch_embed = nn.Conv2d(3, 8, kernel_size=2, stride=2)
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
        self.norm = nn.LayerNorm(8)

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
        return SimpleNamespace(
            hidden_states=tuple(hidden_states),
            last_hidden_state=self.norm(hidden_states[-1]),
        )

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

    def test_ordered_per_root_split_matches_private_dino_recipe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            self._make_dataset(root, count=10)
            train, val, _ = discover_segmentation_samples(
                [root],
                val_fraction=0.2,
                split_seed=17,
                split_strategy="ordered_per_root",
            )
            self.assertEqual(len(train), 8)
            self.assertEqual(len(val), 2)
            self.assertTrue(train[0].sample_id.endswith("scene_000_r000_c000"))
            self.assertTrue(train[-1].sample_id.endswith("scene_007_r000_c000"))
            self.assertTrue(val[0].sample_id.endswith("scene_008_r000_c000"))

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
        self.assertFalse(model.vision_encoder.embeddings.mask_token.requires_grad)
        self.assertIsNotNone(model.vision_encoder.embeddings.patch_embeddings.projection.weight.grad)
        self.assertIsNotNone(model.vision_encoder.layernorm.weight.grad)
        self.assertIsNotNone(model.vision_encoder.layernorm.bias.grad)

    def test_mllm_dinov2_wrapper_unfreezes_only_last_two_blocks(self):
        from transformers import Dinov2Config, Dinov2Model

        from mllm.model.multimodal_encoder.dinov2_encoder import DINOv2VisionTower

        config = Dinov2Config(
            image_size=8,
            patch_size=2,
            num_channels=3,
            hidden_size=8,
            num_hidden_layers=4,
            num_attention_heads=2,
            intermediate_size=16,
        )
        tower = DINOv2VisionTower.__new__(DINOv2VisionTower)
        nn.Module.__init__(tower)
        tower.vision_tower = Dinov2Model(config)
        tower.unfreeze_last_n_blocks = -1
        tower.select_feature = "patch"
        tower.select_layer = 4
        tower.num_layers = 4
        tower.vision_layer_fusion = None
        tower.deepstack_mergers = None
        tower.set_vision_tower_trainable(True, last_n_blocks=2)
        tower._resolve_select_layer_index()

        blocks = tower.vision_tower.encoder.layer
        self.assertEqual(tower.trainable_vision_block_indices, [2, 3])
        self.assertFalse(next(blocks[0].parameters()).requires_grad)
        self.assertFalse(next(blocks[1].parameters()).requires_grad)
        self.assertTrue(next(blocks[2].parameters()).requires_grad)
        self.assertTrue(next(blocks[3].parameters()).requires_grad)
        self.assertFalse(
            tower.vision_tower.embeddings.patch_embeddings.projection.weight.requires_grad
        )
        self.assertTrue(tower.vision_tower.layernorm.weight.requires_grad)

        hidden_states = tuple(torch.zeros(1, 5, 8) for _ in range(5))
        final_output = torch.full((1, 5, 8), 7.0)
        selected, deepstack = tower.feature_select(
            SimpleNamespace(
                hidden_states=hidden_states,
                last_hidden_state=final_output,
            )
        )
        self.assertIsNone(deepstack)
        self.assertTrue(torch.equal(selected, final_output[:, 1:]))

    def test_legacy_decoder_trains_only_tail_blocks_and_final_norm(self):
        model = Dinov2RoadSegmentationModel(
            FakeVisionEncoder(),
            input_size=8,
            hidden_state_indices=[4],
            projection_channels=16,
            decoder_type="legacy_single_layer",
            vision_unfreeze_last_n_blocks=2,
        )
        self.assertEqual(model.trainable_vision_block_indices, (2, 3))
        self.assertEqual(len(model.decoder.stages), 4)
        self.assertFalse(model.vision_encoder.patch_embed.weight.requires_grad)
        self.assertFalse(next(model.vision_encoder.layers[0].parameters()).requires_grad)
        self.assertFalse(next(model.vision_encoder.layers[1].parameters()).requires_grad)
        self.assertTrue(next(model.vision_encoder.layers[2].parameters()).requires_grad)
        self.assertTrue(next(model.vision_encoder.layers[3].parameters()).requires_grad)
        self.assertTrue(model.vision_encoder.norm.weight.requires_grad)

        output = model(torch.randn(1, 3, 8, 8))
        self.assertEqual(tuple(output.shape), (1, 2, 8, 8))
        output.square().mean().backward()
        self.assertIsNone(model.vision_encoder.layers[1].weight.grad)
        self.assertIsNotNone(model.vision_encoder.layers[2].weight.grad)
        self.assertIsNotNone(model.vision_encoder.norm.weight.grad)

    def test_legacy_decoder_rejects_multiple_hidden_layers(self):
        with self.assertRaisesRegex(ValueError, "requires exactly one"):
            Dinov2RoadSegmentationModel(
                FakeVisionEncoder(),
                input_size=8,
                hidden_state_indices=[3, 4],
                projection_channels=16,
                decoder_type="legacy_single_layer",
                vision_unfreeze_last_n_blocks=2,
            )

    def test_dinov2_lora_merges_to_hf_state_dict_keys(self):
        from transformers import Dinov2Config, Dinov2Model

        config = Dinov2Config(
            image_size=8,
            patch_size=2,
            num_channels=3,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
        )
        model = Dinov2RoadSegmentationModel(
            Dinov2Model(config),
            input_size=8,
            hidden_state_indices=[1, 2],
            projection_channels=16,
            vision_lora_enable=True,
            vision_lora_r=2,
            vision_lora_alpha=4,
            vision_lora_target_modules="query,value",
        )
        self.assertTrue(model.vision_lora_modules)
        self.assertFalse(model.vision_encoder.embeddings.patch_embeddings.projection.weight.requires_grad)
        self.assertTrue(any("lora_A" in name for name, _ in model.vision_encoder.named_parameters()))
        state_dict, merged_count = merged_lora_state_dict(model.vision_encoder)
        self.assertGreater(merged_count, 0)
        self.assertIn("encoder.layer.0.attention.attention.query.weight", state_dict)
        self.assertIn("encoder.layer.0.attention.attention.value.weight", state_dict)
        self.assertFalse(any("lora_A" in key or "lora_B" in key for key in state_dict))
        self.assertFalse(any("base_layer" in key for key in state_dict))

    def test_fixed_two_class_confusion_metrics(self):
        logits = torch.tensor(
            [[[[4.0, -1.0], [4.0, -1.0]], [[-1.0, 4.0], [-1.0, 4.0]]]]
        )
        labels = torch.tensor([[[0, 1], [1, 1]]])
        matrix = confusion_matrix(logits, labels)
        self.assertEqual(matrix.dtype, torch.float32)
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
