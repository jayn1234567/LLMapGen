import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from mllm.model.multimodal_encoder.dinov3_encoder import DINOv3VisionTower
from mllm.vision_pretrain.dinov2_segmentation import Dinov2RoadSegmentationModel
from mllm.vision_pretrain.dinov3_segmentation import Dinov3RoadSegmentationModel


class FakeDinoV3Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            patch_size=2,
            hidden_size=8,
            num_hidden_layers=4,
            num_register_tokens=4,
        )
        self.patch_embed = nn.Conv2d(3, 8, kernel_size=2, stride=2)
        self.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
        self.norm = nn.LayerNorm(8)

    def forward(self, pixel_values, output_hidden_states=True, return_dict=True):
        del output_hidden_states, return_dict
        patches = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        prefix = patches.new_zeros((patches.shape[0], 5, patches.shape[2]))
        hidden = torch.cat((prefix, patches), dim=1)
        hidden_states = [hidden]
        for block in self.blocks:
            hidden = block(hidden)
            hidden_states.append(hidden)
        return SimpleNamespace(
            hidden_states=tuple(hidden_states),
            last_hidden_state=self.norm(hidden_states[-1]),
        )


class Dinov3SegmentationTests(unittest.TestCase):
    def test_huggingface_dinov3_forward_contract(self):
        from transformers import DINOv3ViTConfig, DINOv3ViTModel

        config = DINOv3ViTConfig(
            image_size=8,
            patch_size=2,
            num_channels=3,
            hidden_size=8,
            num_hidden_layers=4,
            num_attention_heads=2,
            intermediate_size=16,
            num_register_tokens=4,
        )
        model = Dinov3RoadSegmentationModel(
            DINOv3ViTModel(config),
            input_size=8,
            hidden_state_indices=[4],
            projection_channels=16,
            decoder_type="legacy_single_layer",
            vision_unfreeze_last_n_blocks=-1,
        )
        output = model(torch.randn(1, 3, 8, 8))
        self.assertEqual(tuple(output.shape), (1, 2, 8, 8))
        output.square().mean().backward()
        self.assertFalse(model.vision_encoder.embeddings.mask_token.requires_grad)
        self.assertIsNotNone(model.vision_encoder.embeddings.patch_embeddings.weight.grad)
        self.assertIsNotNone(model.vision_encoder.norm.weight.grad)

    def test_full_parameter_forward_accepts_register_tokens(self):
        model = Dinov3RoadSegmentationModel(
            FakeDinoV3Encoder(),
            input_size=8,
            hidden_state_indices=[4],
            projection_channels=16,
            decoder_type="legacy_single_layer",
            vision_unfreeze_last_n_blocks=-1,
        )
        output = model(torch.randn(2, 3, 8, 8))
        self.assertEqual(tuple(output.shape), (2, 2, 8, 8))
        output.square().mean().backward()
        self.assertEqual(model.num_register_tokens, 4)
        self.assertIsNone(model.trainable_vision_block_indices)
        self.assertTrue(all(parameter.requires_grad for parameter in model.vision_encoder.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in model.vision_encoder.parameters()))

    def test_dinov2_recipe_still_rejects_register_tokens(self):
        with self.assertRaisesRegex(ValueError, "register tokens"):
            Dinov2RoadSegmentationModel(
                FakeDinoV3Encoder(),
                input_size=8,
                hidden_state_indices=[4],
                projection_channels=16,
                decoder_type="legacy_single_layer",
            )

    def test_mllm_wrapper_layer_24_uses_final_normalized_output(self):
        tower = DINOv3VisionTower.__new__(DINOv3VisionTower)
        nn.Module.__init__(tower)
        tower.select_feature = "patch"
        tower.skip_tokens = 5
        tower.select_layer_idx = 24
        tower.vision_layer_fusion = None
        tower.deepstack_mergers = None
        hidden_states = tuple(torch.zeros(1, 9, 8) for _ in range(25))
        final_output = torch.full((1, 9, 8), 7.0)
        selected, deepstack = tower.feature_select(
            SimpleNamespace(hidden_states=hidden_states, last_hidden_state=final_output)
        )
        self.assertIsNone(deepstack)
        self.assertEqual(tuple(selected.shape), (1, 4, 8))
        self.assertTrue(torch.equal(selected, final_output[:, 5:]))

    def test_formal_di_recipe_is_32_card_full_parameter_and_jiang_compatible(self):
        script = (
            Path(__file__).parents[1]
            / "scripts"
            / "npu"
            / "train"
            / "train_dinov3_private_seg_full_finetune_di_npu.sh"
        ).read_text(encoding="utf-8")
        expected_fragments = (
            "TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-128}",
            "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}",
            "VISION_LEARNING_RATE=${VISION_LEARNING_RATE:-2e-5}",
            "DECODER_LEARNING_RATE=${DECODER_LEARNING_RATE:-1e-4}",
            "EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE:-32}",
            "BEST_METRIC=${BEST_METRIC:-lane_iou}",
            "--hidden_state_indices 24",
            "--vision_unfreeze_last_n_blocks -1",
            "--decoder_type legacy_single_layer",
            "--normalization_mode minus_half",
            "--select-layer 24",
            '"mm_vision_tower_type": "dinov3"',
        )
        for fragment in expected_fragments:
            self.assertIn(fragment, script)


if __name__ == "__main__":
    unittest.main()
