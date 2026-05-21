#!/usr/bin/env python3
import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer

from mllm import conversation as conversation_lib
from mllm.train.train_qwen import preprocess_multimodal
from mllm.mm_utils import tokenizer_image_token


def build_conversation(source, conv_template):
    conv = conversation_lib.conv_templates[conv_template].copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    if roles[source[0]["from"]] != conv.roles[0]:
        source = source[1:]

    conv.messages = []
    for idx, sentence in enumerate(source):
        role = roles[sentence["from"]]
        assert role == conv.roles[idx % 2], f"Unexpected role order at message {idx}"
        conv.append_message(role, sentence["value"])
    return conv.get_prompt()


def analyze_qwen2_mismatch(prompt, tokenizer, conv_template):
    conv = conversation_lib.conv_templates[conv_template].copy()
    split_sep = conv.sep + conv.roles[1]

    input_ids = tokenizer_image_token(prompt, tokenizer)
    total_len = len(input_ids)

    rounds_before = prompt.split(conv.sep)
    if rounds_before[0] == conv.system:
        rounds_before[1] = conv.sep.join([rounds_before[0], rounds_before[1]])
        rounds_before = rounds_before[1:]

    rounds = []
    for idx in range(0, len(rounds_before), 2):
        if idx < len(rounds_before) - 1:
            rounds.append(conv.sep.join([rounds_before[idx], rounds_before[idx + 1]]))
        else:
            rounds.append(rounds_before[idx])

    cur_len = 0
    round_stats = []
    for rou in rounds:
        if rou == "":
            break

        parts = rou.split(split_sep)
        if len(parts) != 2:
            round_stats.append(
                {
                    "round_text_preview": rou[:120],
                    "status": "split_error",
                }
            )
            break

        parts[0] += split_sep
        round_ids = tokenizer_image_token(rou, tokenizer)
        instruction_ids = tokenizer_image_token(parts[0], tokenizer)
        equal_parts = [x == y for x, y in zip(round_ids, instruction_ids)]
        instruction_len = equal_parts.index(False) if False in equal_parts else len(equal_parts)
        round_len = len(round_ids) + 2
        cur_len += round_len

        round_stats.append(
            {
                "round_text_preview": rou[:120],
                "round_len": round_len,
                "instruction_len": instruction_len,
            }
        )

    return {
        "total_len": total_len,
        "cur_len": cur_len,
        "mismatch": cur_len != total_len,
        "round_stats": round_stats,
        "prompt_preview": prompt[:240],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="checkpoints/llava-fastvithd_1.5b_stage2")
    parser.add_argument("--data-path", default="data/train.jsonl")
    parser.add_argument("--image-folder", default="data/images")
    parser.add_argument("--conv-template", default="qwen_2_centerline_coord")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--show-examples", type=int, default=10)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.legacy = False

    if args.conv_template not in conversation_lib.conv_templates:
        raise KeyError(f"Unknown conv template: {args.conv_template}")
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_template]

    raw_text = Path(args.data_path).read_text(encoding="utf-8")
    if raw_text.strip().startswith('['):
        data = json.loads(raw_text)
    else:
        data = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    data = data[: args.limit] if args.limit > 0 else data

    total = 0
    mismatch_count = 0
    mismatch_examples = []

    for sample in data:
        conversations = copy.deepcopy(sample["conversations"])
        sources = preprocess_multimodal([conversations], argparse.Namespace(
            is_multimodal=True,
            mm_use_im_start_end=False,
        ))
        prompt = build_conversation(sources[0], args.conv_template)
        stats = analyze_qwen2_mismatch(prompt, tokenizer, args.conv_template)
        total += 1
        if stats["mismatch"]:
            mismatch_count += 1
            if len(mismatch_examples) < args.show_examples:
                mismatch_examples.append(
                    {
                        "id": sample.get("id"),
                        "image": sample.get("image"),
                        "total_len": stats["total_len"],
                        "cur_len": stats["cur_len"],
                        "prompt_preview": stats["prompt_preview"],
                        "ground_truth_preview": sample["conversations"][-1]["value"][:240],
                        "round_stats": stats["round_stats"][:4],
                    }
                )

    ratio = (mismatch_count / total) if total else 0.0
    summary = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "conv_template": args.conv_template,
        "samples_checked": total,
        "mismatch_count": mismatch_count,
        "mismatch_ratio": ratio,
        "examples": mismatch_examples,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
