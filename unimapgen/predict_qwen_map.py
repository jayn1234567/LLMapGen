import argparse

import torch
from torch.utils.data import DataLoader

from unimapgen.qwen_map_pipeline import (
    build_qwen_map_components,
    build_qwen_map_dataset,
    lines_to_jsonable,
    maybe_load_model_checkpoint,
    save_json,
)
from unimapgen.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="outputs/qwen_map_predictions.json")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--min_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_set = build_qwen_map_dataset(
        cfg,
        split=str(cfg["data"]["train_split"]),
        max_samples=cfg["data"].get("max_train_samples"),
        train_augment=False,
    )
    ds = build_qwen_map_dataset(
        cfg,
        split=str(args.split),
        max_samples=None if args.max_samples is None else int(args.max_samples),
        train_augment=False,
    )
    qwen_map_tokenizer, collator, model = build_qwen_map_components(cfg, train_set=train_set)
    maybe_load_model_checkpoint(model, args.checkpoint)

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dec_cfg = cfg.get("decode", {})
    max_new_tokens = int(args.max_new_tokens if args.max_new_tokens is not None else dec_cfg.get("max_new_tokens", 256))
    min_new_tokens = int(args.min_new_tokens if args.min_new_tokens is not None else dec_cfg.get("min_new_tokens", 0))
    temperature = float(args.temperature if args.temperature is not None else dec_cfg.get("temperature", 1.0))
    top_k = int(args.top_k if args.top_k is not None else dec_cfg.get("top_k", 1))
    repetition_penalty = float(
        args.repetition_penalty if args.repetition_penalty is not None else dec_cfg.get("repetition_penalty", 1.0)
    )
    use_grammar_constraint = bool(dec_cfg.get("use_grammar_constraint", True))
    grammar_min_points_per_line = int(dec_cfg.get("grammar_min_points_per_line", 2))
    grammar_max_lines = int(dec_cfg.get("grammar_max_lines", cfg["serialization"]["max_lines"]))

    outputs = []
    with torch.no_grad():
        for batch in loader:
            pred_qwen_ids = model.generate(
                image=batch["image"].to(device),
                prompt_input_ids=batch["prompt_input_ids"].to(device),
                prompt_attention_mask=batch["prompt_attention_mask"].to(device),
                state_input_ids=batch["state_input_ids"].to(device),
                state_attention_mask=batch["state_attention_mask"].to(device),
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                grammar_helper=qwen_map_tokenizer if use_grammar_constraint else None,
                grammar_min_points_per_line=grammar_min_points_per_line,
                grammar_max_lines=grammar_max_lines,
            )[0].detach().cpu().tolist()
            pred_custom_ids = qwen_map_tokenizer.decode_qwen_map_ids_to_custom_ids(pred_qwen_ids)
            gt_custom_ids = batch["gt_map_token_ids"][0].tolist()
            gt_custom_ids = [int(x) for x in gt_custom_ids if int(x) != qwen_map_tokenizer.map_tokenizer.pad_id]

            pred_lines = train_set.map_tokenizer.decode_to_lines(pred_custom_ids)
            gt_lines = train_set.map_tokenizer.decode_to_lines(gt_custom_ids)
            outputs.append(
                {
                    "token": batch["token_strs"][0],
                    "prompt_text": batch["prompt_texts"][0],
                    "gt_state_custom_ids": [
                        int(x)
                        for x in batch["gt_state_token_ids"][0].tolist()
                        if int(x) != qwen_map_tokenizer.map_tokenizer.pad_id
                    ],
                    "pred_qwen_ids": pred_qwen_ids,
                    "pred_custom_ids": pred_custom_ids,
                    "gt_custom_ids": gt_custom_ids,
                    "pred_lines": lines_to_jsonable(pred_lines),
                    "gt_lines": lines_to_jsonable(gt_lines),
                }
            )

    save_json(args.output, outputs)
    print(f"saved {len(outputs)} samples to {args.output}")


if __name__ == "__main__":
    main()
