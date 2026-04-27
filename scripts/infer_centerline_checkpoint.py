#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer

try:
    from safetensors.torch import load_file as safe_load_file
except ImportError:  # pragma: no cover
    safe_load_file = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava import conversation as conversation_lib
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_qwen import LlavaConfig, LlavaQwen2ForCausalLM

DEFAULT_PROMPT = DEFAULT_IMAGE_TOKEN


def normalize_prediction_text(text: str) -> str:
    cleaned = text.strip()
    for token in ("<|im_end|>", "<|endoftext|>", "</s>"):
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned


def read_manifest(checkpoint_dir: Path) -> dict:
    search_dirs = [checkpoint_dir]
    if checkpoint_dir.parent != checkpoint_dir:
        search_dirs.append(checkpoint_dir.parent)

    for search_dir in search_dirs:
        manifest_path = search_dir / "inference_manifest.json"
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["checkpoint_dir"] = str(checkpoint_dir)
            payload["manifest_source"] = str(manifest_path)
            return payload

        run_config_path = search_dir / "run_config.json"
        if run_config_path.exists():
            payload = json.loads(run_config_path.read_text(encoding="utf-8"))
            model_args = payload.get("model_args", {})
            data_args = payload.get("data_args", {})
            training_args = payload.get("training_args", {})
            return {
                "checkpoint_dir": str(checkpoint_dir),
                "manifest_source": str(run_config_path),
                "model_name_or_path": model_args.get("model_name_or_path"),
                "version": model_args.get("version"),
                "image_aspect_ratio": data_args.get("image_aspect_ratio"),
                "full_model_finetune": training_args.get("full_model_finetune"),
                "lora_enable": training_args.get("lora_enable"),
            }

    return {"checkpoint_dir": str(checkpoint_dir)}


def ensure_prompt_has_image_token(prompt: str) -> str:
    if DEFAULT_IMAGE_TOKEN in prompt:
        return prompt
    return DEFAULT_IMAGE_TOKEN + "\n" + prompt


def build_prompt(user_message: str, conv_template: str) -> str:
    if conv_template not in conversation_lib.conv_templates:
        raise KeyError(f"Unknown conversation template: {conv_template}")
    conv = conversation_lib.conv_templates[conv_template].copy()
    conv.messages = []
    conv.append_message(conv.roles[0], ensure_prompt_has_image_token(user_message))
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def _load_state_dict(checkpoint_dir: Path):
    safetensors_path = checkpoint_dir / "model.safetensors"
    bin_path = checkpoint_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        if safe_load_file is None:
            raise ImportError("safetensors is required to load model.safetensors")
        return safe_load_file(str(safetensors_path), device="cpu")
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"No model weights found under {checkpoint_dir}")


def _resolve_base_model_path(base_model_path: str, checkpoint_dir: Path) -> Path:
    candidate = Path(base_model_path)
    if candidate.is_absolute():
        return candidate

    rel_to_ckpt = (checkpoint_dir / candidate).resolve()
    if rel_to_ckpt.exists():
        return rel_to_ckpt

    rel_to_repo = (REPO_ROOT / candidate).resolve()
    if rel_to_repo.exists():
        return rel_to_repo

    return candidate


def _read_llava_checkpoint_metadata(checkpoint_dir: Path) -> dict:
    metadata_path = checkpoint_dir / "llava_checkpoint.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _load_full_finetune_model(checkpoint_dir: Path, device: str):
    checkpoint_dir_str = str(checkpoint_dir.resolve())
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir_str, use_fast=False, local_files_only=True)
    config = LlavaConfig.from_pretrained(checkpoint_dir_str, local_files_only=True)
    config.fastvit_pretrained = False
    config.fastvit_pretrained_path = None

    model = LlavaQwen2ForCausalLM(config)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if vision_tower is not None and not vision_tower.is_loaded:
        vision_tower.load_model()

    metadata = _read_llava_checkpoint_metadata(checkpoint_dir)
    if metadata and not metadata.get("bundled_vision_tower", True):
        base_model_path = config._name_or_path
        if not base_model_path:
            raise ValueError("Missing config._name_or_path for non-bundled vision tower checkpoint")
        base_checkpoint_dir = _resolve_base_model_path(base_model_path, checkpoint_dir)
        base_state_dict = _load_state_dict(base_checkpoint_dir)
        base_vision_tower_state = {
            key: value for key, value in base_state_dict.items()
            if key.startswith("model.vision_tower.")
        }
        missing, unexpected = model.load_state_dict(base_vision_tower_state, strict=False)
        if unexpected:
            print(f"[WARN] Unexpected base vision-tower keys: {unexpected[:20]}")
        missing = [key for key in missing if not key.startswith("model.vision_tower.")]
        if missing:
            print(f"[WARN] Missing keys after base vision-tower load: {missing[:20]}")

    state_dict = _load_state_dict(checkpoint_dir)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if "lm_head.weight" in missing and getattr(model.config, "tie_word_embeddings", False):
        model.tie_weights()
        missing = [key for key in missing if key != "lm_head.weight"]
    if missing:
        print(f"[WARN] Missing keys after full-finetune load: {missing[:20]}")
    if unexpected:
        print(f"[WARN] Unexpected keys after full-finetune load: {unexpected[:20]}")

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = model.to(dtype=dtype)
    if device == "cuda":
        model = model.to("cuda")
    else:
        model = model.to(device)
    model.eval()

    image_processor = model.get_model().get_vision_tower().image_processor
    return tokenizer, model, image_processor


def load_model_components(checkpoint_dir: Path, manifest: dict, device: str):
    if manifest.get("full_model_finetune") or (checkpoint_dir / "llava_checkpoint.json").exists():
        return _load_full_finetune_model(checkpoint_dir, device)

    model_base = manifest.get("model_name_or_path")
    model_name = f"llava_{checkpoint_dir.name}"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=str(checkpoint_dir),
        model_base=model_base,
        model_name=model_name,
        device_map="auto" if device == "cuda" else {"": device},
        device=device,
    )
    model.eval()
    return tokenizer, model, image_processor


def parse_centerline_json(prediction_text: str):
    parsed_items = json.loads(prediction_text)
    if not isinstance(parsed_items, list):
        raise ValueError("prediction is not a JSON list")

    for item in parsed_items:
        if not isinstance(item, dict):
            raise ValueError("prediction item is not an object")
        if item.get("category") != "CenterLine":
            raise ValueError("non-CenterLine category found")
        points = item.get("points")
        if not isinstance(points, list) or not points:
            raise ValueError("missing points")
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("invalid point format")
    return parsed_items


def sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--image", default="")
    parser.add_argument("--test-json", default="")
    parser.add_argument("--image-folder", default="")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--prompt-mode", choices=["default", "dataset"], default="default")
    parser.add_argument("--conv-template", default="")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--print-full-output", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    manifest = read_manifest(checkpoint_dir)

    conv_template = args.conv_template or manifest.get("version") or "qwen_2_centerline_coord"
    if conv_template not in conversation_lib.conv_templates:
        conv_template = "qwen_2_centerline_coord"

    tokenizer, model, image_processor = load_model_components(checkpoint_dir, manifest, args.device)

    if args.test_json:
        with open(args.test_json, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                records = json.load(f)
            else:
                records = [json.loads(line) for line in f if line.strip()]
        if not args.image_folder:
            raise ValueError("--image-folder is required when using --test-json")
        start = max(0, args.sample_offset)
        end = start + max(1, args.num_samples)
        records = records[start:end]
    else:
        if not args.image:
            raise ValueError("Provide either --image or --test-json")
        records = [{"id": "single_image", "image": args.image, "conversations": [{"value": args.prompt}]}]

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, record in enumerate(records):
        image_path = Path(record["image"])
        if args.test_json:
            image_path = Path(args.image_folder) / image_path
        image_path = image_path.resolve()

        image = Image.open(image_path).convert("RGB")
        images_tensor = process_images([image], image_processor, model.config)
        dtype = next(model.parameters()).dtype
        if isinstance(images_tensor, list):
            images_tensor = [img.to(dtype=dtype, device=model.device) for img in images_tensor]
        else:
            images_tensor = images_tensor.to(dtype=dtype, device=model.device)

        if args.prompt_mode == "dataset" and record.get("conversations"):
            prompt_text = record["conversations"][0]["value"]
        else:
            prompt_text = args.prompt
        prompt = build_prompt(prompt_text, conv_template)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(model.device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=images_tensor,
                image_sizes=[image.size],
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Keep the full generated sequence. For this task the model is expected
        # to output the JSON directly, and length-based slicing can remove valid
        # prefixes when generate() already returns completion-only tokens.
        decoded_ids = output_ids

        raw_prediction = tokenizer.batch_decode(
            decoded_ids,
            skip_special_tokens=False,
        )[0].strip()
        prediction = normalize_prediction_text(raw_prediction)

        parse_ok = False
        parsed_items = []
        parse_error = ""
        try:
            parsed_items = parse_centerline_json(prediction)
            parse_ok = True
        except Exception as exc:
            parse_error = str(exc)

        result = {
            "checkpoint_dir": str(checkpoint_dir),
            "image": str(image_path),
            "record_id": record.get("id", f"sample_{idx}"),
            "prompt": prompt,
            "conv_template": conv_template,
            "raw_prediction": raw_prediction,
            "prediction": prediction,
            "parse_ok": parse_ok,
            "num_items": len(parsed_items) if parse_ok else 0,
            "parse_error": parse_error,
            "input_token_len": int(input_ids.shape[1]),
            "output_token_len": int(output_ids.shape[1]),
            "decoded_token_len": int(decoded_ids.shape[1]),
            "decoded_mode": "full_output",
            "manifest": manifest,
        }
        if len(record.get("conversations", [])) > 1:
            result["ground_truth"] = record["conversations"][1]["value"]
        results.append(result)

        if output_dir is not None:
            sample_path = output_dir / f"{idx:03d}_{sanitize_filename(str(result['record_id']))}.json"
            sample_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        print(
            json.dumps(
                {
                    "idx": idx,
                    "record_id": result["record_id"],
                    "image": result["image"],
                    "parse_ok": parse_ok,
                    "num_items": result["num_items"],
                    "parse_error": parse_error,
                    "decoded_mode": result["decoded_mode"],
                    "input_token_len": result["input_token_len"],
                    "output_token_len": result["output_token_len"],
                    "decoded_token_len": result["decoded_token_len"],
                },
                ensure_ascii=False,
            )
        )
        if args.print_full_output:
            print("RAW_PREDICTION_START")
            print(raw_prediction)
            print("RAW_PREDICTION_END")
            print("NORMALIZED_PREDICTION_START")
            print(prediction)
            print("NORMALIZED_PREDICTION_END")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
