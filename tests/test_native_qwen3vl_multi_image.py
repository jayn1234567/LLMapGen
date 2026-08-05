import json
from pathlib import Path

from PIL import Image
import torch

from mllm.native_qwen3vl.data import (
    IGNORE_INDEX,
    NativeQwen3VLDataCollator,
    NativeQwen3VLDataset,
    ROAD_MAP_SYSTEM_PROMPT,
    build_qwen3vl_messages,
)
from mllm.data_sampling import deterministic_sample_indices


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2


class _Processor:
    tokenizer = _Tokenizer()

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        image_count = sum(
            item.get("type") == "image"
            for message in messages
            for item in message.get("content", [])
            if isinstance(item, dict)
        )
        has_answer = any(message.get("role") == "assistant" for message in messages)
        return f"images={image_count};answer={int(has_answer)};generation={int(add_generation_prompt)}"

    def __call__(self, *, text, images, return_tensors, **kwargs):
        assert return_tensors == "pt"
        self.calls.append({"text": text[0], "num_images": len(images), **kwargs})
        has_answer = "answer=1" in text[0]
        token_count = 12 if has_answer else 8
        return {
            "input_ids": torch.arange(1, token_count + 1).unsqueeze(0),
            "attention_mask": torch.ones(1, token_count, dtype=torch.long),
            "pixel_values": torch.zeros(len(images), 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 16, 16]] * len(images)),
        }


def _write_three_image_record(tmp_path: Path, sample_id: str = "sample-1") -> Path:
    paths = []
    for index, folder in enumerate(("images", "raw_lane_images", "pose_images")):
        relative = Path(folder) / "train" / "group" / f"{sample_id}.png"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(index * 50, 0, 0)).save(path)
        paths.append(relative.as_posix())

    record = {
        "id": sample_id,
        "image": paths[0],
        "images": paths,
        "conversations": [
            {"from": "human", "value": "<image>\n<image>\n<image>\nDescribe the map."},
            {"from": "gpt", "value": '{"lines":[]}'},
        ],
    }
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return jsonl


def test_build_messages_preserves_three_image_order():
    paths = [Path("clean.png"), Path("raw.png"), Path("pose.png")]
    messages = build_qwen3vl_messages(
        "<image>\n<image>\n<image>\nPrompt", paths, "answer", ROAD_MAP_SYSTEM_PROMPT
    )

    assert messages[0] == {"role": "system", "content": ROAD_MAP_SYSTEM_PROMPT}
    content = messages[1]["content"]
    assert [item["image"] for item in content[:3]] == [str(path) for path in paths]
    assert content[3] == {"type": "text", "text": "Prompt"}
    assert messages[2] == {"role": "assistant", "content": "answer"}


def test_dataset_and_collator_encode_all_three_images(tmp_path):
    jsonl = _write_three_image_record(tmp_path)
    dataset = NativeQwen3VLDataset([str(jsonl)], [str(tmp_path)])
    feature = dataset[0]

    assert [path.parts[-4] for path in feature["image_paths"]] == [
        "images",
        "raw_lane_images",
        "pose_images",
    ]

    processor = _Processor()
    batch = NativeQwen3VLDataCollator(processor, model_max_length=6144)([feature])

    assert [call["num_images"] for call in processor.calls] == [3, 3]
    assert batch["pixel_values"].shape[0] == 3
    assert batch["image_grid_thw"].shape == (3, 3)
    assert torch.all(batch["labels"][0, :8] == IGNORE_INDEX)
    assert torch.all(batch["labels"][0, 8:12] != IGNORE_INDEX)


def test_collator_concatenates_three_image_grids_across_samples(tmp_path):
    jsonl = _write_three_image_record(tmp_path)
    dataset = NativeQwen3VLDataset([str(jsonl)], [str(tmp_path)])
    processor = _Processor()

    batch = NativeQwen3VLDataCollator(processor, model_max_length=6144)([dataset[0], dataset[0]])

    assert batch["input_ids"].shape == (2, 12)
    assert batch["pixel_values"].shape[0] == 6
    assert batch["image_grid_thw"].shape == (6, 3)


def test_matched_subset_indices_are_stable_and_source_ordered():
    first = deterministic_sample_indices(total=20, limit=7, seed=42)
    second = deterministic_sample_indices(total=20, limit=7, seed=42)

    assert first == second
    assert first == sorted(first)
    assert len(first) == 7
