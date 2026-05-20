from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from mllm import conversation as conversation_lib
from mllm.constants import DEFAULT_IMAGE_TOKEN
from mllm.coord_utils import COORD_MODE_PIXEL, record_coord_config
from mllm.mm_utils import process_anyres_image, tokenizer_image_token


def load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list in {path}")
        return [row for row in payload if isinstance(row, dict)]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def ensure_prompt_has_image_token(prompt: str) -> str:
    prompt = str(prompt or "").strip()
    if DEFAULT_IMAGE_TOKEN in prompt:
        return prompt
    return f"{DEFAULT_IMAGE_TOKEN}\n{prompt}".strip()


def build_generation_prompt(user_message: str, conv_template: str) -> str:
    if conv_template not in conversation_lib.conv_templates:
        raise KeyError(f"Unknown conversation template: {conv_template}")
    conv = conversation_lib.conv_templates[conv_template].copy()
    conv.messages = []
    conv.append_message(conv.roles[0], ensure_prompt_has_image_token(user_message))
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


@dataclass
class RLPromptBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    images: torch.Tensor | list[torch.Tensor]
    image_sizes: list[tuple[int, int]]
    image_paths: list[str]
    prompts: list[str]
    ground_truths: list[str]
    sample_ids: list[str]
    coord_configs: list[dict[str, Any]]


class RLSFTJsonlDataset(Dataset):
    """Prompt-only multimodal dataset for post-SFT RL.

    The source rows keep the same JSONL shape as SFT data. Only the human turn
    is used as the rollout prompt; the assistant turn remains the reward target.
    """

    def __init__(
        self,
        data_paths: Sequence[str],
        image_folders: Sequence[str],
        tokenizer,
        image_processor,
        conv_template: str,
        image_aspect_ratio: str = "pad",
        image_grid_pinpoints: str | None = None,
        sample_limit: int | None = None,
        sample_seed: int = 42,
        default_coord_mode: str = COORD_MODE_PIXEL,
        default_patch_size: int = 256,
        default_coord_range: int = 1000,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_template = conv_template
        self.image_aspect_ratio = image_aspect_ratio
        self.image_grid_pinpoints = image_grid_pinpoints
        self.default_coord_mode = default_coord_mode
        self.default_patch_size = default_patch_size
        self.default_coord_range = default_coord_range

        if len(image_folders) == 1 and len(data_paths) > 1:
            image_folders = list(image_folders) * len(data_paths)
        if len(data_paths) != len(image_folders):
            raise ValueError("data_path and image_folder must have the same length, or image_folder must contain one path.")

        records: list[dict[str, Any]] = []
        for path_idx, data_path in enumerate(data_paths):
            for row in load_json_or_jsonl(data_path):
                row = dict(row)
                row["_img_path_idx"] = path_idx
                records.append(row)

        if sample_limit is not None and sample_limit > 0 and len(records) > sample_limit:
            rng = random.Random(sample_seed)
            keep = sorted(rng.sample(range(len(records)), sample_limit))
            records = [records[idx] for idx in keep]

        self.records = records
        self.image_folders = list(image_folders)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, row: dict[str, Any]) -> str:
        image_file = row.get("image")
        if not image_file:
            raise ValueError(f"Sample {row.get('id', '<missing id>')} has no image field")
        folder = self.image_folders[int(row["_img_path_idx"])]
        return os.path.join(folder, image_file)

    def _load_image(self, row: dict[str, Any]) -> tuple[torch.Tensor, tuple[int, int], str]:
        image_path = self._resolve_image_path(row)
        image = Image.open(image_path).convert("RGB")
        image_size = image.size
        processor = self.image_processor

        if self.image_aspect_ratio == "pad":
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                if width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            return processor.preprocess(image, return_tensors="pt")["pixel_values"][0], image_size, image_path
        if self.image_aspect_ratio == "anyres" or "anyres_max" in self.image_aspect_ratio:
            return process_anyres_image(image, processor, self.image_grid_pinpoints), image_size, image_path
        return processor.preprocess(image, return_tensors="pt")["pixel_values"][0], image_size, image_path

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        conversations = copy.deepcopy(row.get("conversations") or [])
        if len(conversations) < 2:
            raise ValueError(f"Sample {row.get('id', index)} must contain human and gpt conversations")
        human = conversations[0]
        assistant = conversations[1]
        if not isinstance(human, dict) or not isinstance(assistant, dict):
            raise ValueError(f"Sample {row.get('id', index)} has invalid conversations")

        prompt = build_generation_prompt(str(human.get("value") or ""), self.conv_template)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
        image, image_size, image_path = self._load_image(row)
        coord_config = record_coord_config(
            row,
            default_mode=self.default_coord_mode,
            default_patch_size=self.default_patch_size,
            default_coord_range=self.default_coord_range,
        )
        return {
            "input_ids": input_ids,
            "image": image,
            "image_size": image_size,
            "image_path": image_path,
            "prompt": prompt,
            "ground_truth": str(assistant.get("value") or ""),
            "sample_id": str(row.get("id") or f"sample_{index}"),
            "coord_config": coord_config,
        }


class RLDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[dict[str, Any]]) -> RLPromptBatch:
        input_ids = [item["input_ids"] for item in instances]
        padded = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = padded.ne(self.tokenizer.pad_token_id)
        images = [item["image"] for item in instances]
        if all(image is not None and image.shape == images[0].shape for image in images):
            image_batch: torch.Tensor | list[torch.Tensor] = torch.stack(images)
        else:
            image_batch = images
        return RLPromptBatch(
            input_ids=padded,
            attention_mask=attention_mask,
            images=image_batch,
            image_sizes=[item["image_size"] for item in instances],
            image_paths=[item["image_path"] for item in instances],
            prompts=[item["prompt"] for item in instances],
            ground_truths=[item["ground_truth"] for item in instances],
            sample_ids=[item["sample_id"] for item in instances],
            coord_configs=[item["coord_config"] for item in instances],
        )
