import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unimapgen.compare_metrics import _sample_metrics


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def select_items(items: List[Dict[str, Any]], sample_ids: List[str], max_samples: int) -> List[Dict[str, Any]]:
    if sample_ids:
        wanted = set(sample_ids)
        selected = [item for item in items if item.get("id") in wanted]
        missing = [sid for sid in sample_ids if sid not in {item.get("id") for item in selected}]
        if missing:
            raise ValueError(f"sample ids not found: {missing}")
        return selected

    preferred_suffixes = ["_p00", "_p01", "_p04", "_p05", "_p10", "_p15"]
    selected: List[Dict[str, Any]] = []
    seen = set()
    for suffix in preferred_suffixes:
        for item in items:
            sample_id = str(item.get("id", ""))
            if sample_id.endswith(suffix) and sample_id not in seen:
                selected.append(item)
                seen.add(sample_id)
                break
    if len(selected) < max_samples:
        for item in items:
            sample_id = str(item.get("id", ""))
            if sample_id in seen:
                continue
            selected.append(item)
            seen.add(sample_id)
            if len(selected) >= max_samples:
                break
    return selected[:max_samples]


def build_user_text(raw_text: str) -> str:
    return raw_text.replace("<image>\n", "", 1).replace("<image>", "", 1).strip()


def build_conversation(sample: Dict[str, Any], image_path: str) -> List[Dict[str, Any]]:
    messages = sample["messages"]
    conversation: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        if role == "assistant":
            continue
        content = msg["content"]
        if role == "system":
            conversation.append({"role": "system", "content": [{"type": "text", "text": content}]})
        elif role == "user":
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": build_user_text(content)},
                    ],
                }
            )
        else:
            conversation.append({"role": role, "content": [{"type": "text", "text": content}]})
    return conversation


def build_lf_messages_and_system(sample: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Optional[str]]:
    system_text: Optional[str] = None
    messages: List[Dict[str, str]] = []
    for msg in sample["messages"]:
        role = msg["role"]
        if role == "system":
            system_text = str(msg["content"])
        elif role == "user":
            messages.append({"role": "user", "content": build_user_text(str(msg["content"]))})
    return messages, system_text


def extract_state_lines(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    for msg in sample["messages"]:
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content", ""))
        marker = "Previous state:\n"
        idx = content.rfind(marker)
        if idx < 0:
            return []
        raw = content[idx + len(marker) :].strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return list(obj.get("lines", []))
    return []


def extract_gt_lines(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    for msg in sample["messages"]:
        if msg.get("role") == "assistant":
            return list(json.loads(msg["content"]).get("lines", []))
    return []


def _repair_truncated_lines_json(cleaned: str) -> Optional[str]:
    prefix = '{"lines":['
    start = cleaned.find(prefix)
    if start < 0:
        return None

    content = cleaned[start + len(prefix) :]
    last_complete_line = content.rfind("]}")
    if last_complete_line < 0:
        return prefix + "]}"

    kept = content[: last_complete_line + 2].rstrip(", \n\r\t")
    return prefix + kept + "]}"


def parse_generated_json(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        obj = json.loads(cleaned)
        if not isinstance(obj, dict):
            return None, cleaned
        obj.setdefault("lines", [])
        if not isinstance(obj["lines"], list):
            obj["lines"] = []
        return obj, cleaned
    except json.JSONDecodeError:
        repaired = _repair_truncated_lines_json(cleaned)
        if repaired is not None:
            try:
                obj = json.loads(repaired)
                if not isinstance(obj, dict):
                    return None, cleaned
                obj.setdefault("lines", [])
                if not isinstance(obj["lines"], list):
                    obj["lines"] = []
                return obj, repaired
            except json.JSONDecodeError:
                pass
        return None, cleaned


def get_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 20)
    except OSError:
        return ImageFont.load_default()


def draw_polyline(draw: ImageDraw.ImageDraw, points: List[List[int]], color: Tuple[int, int, int], width: int) -> None:
    if len(points) < 2:
        return
    draw.line([tuple(map(int, pt)) for pt in points], fill=color, width=width)


def render_panel(
    image: Image.Image,
    lines: List[Dict[str, Any]],
    state_lines: List[Dict[str, Any]],
    title: str,
    line_color: Tuple[int, int, int],
    size: int = 896,
) -> Image.Image:
    panel = image.copy().convert("RGB")
    draw = ImageDraw.Draw(panel)
    for line in lines:
        draw_polyline(draw, line.get("points", []), line_color, width=4)
    for line in state_lines:
        draw_polyline(draw, line.get("points", []), (255, 165, 0), width=3)
    draw.rectangle([(0, 0), (size - 1, size - 1)], outline=(180, 0, 255), width=3)
    font = get_font()
    draw.rectangle([(8, 8), (300, 42)], fill=(0, 0, 0))
    draw.text((16, 14), title, fill=(255, 255, 255), font=font)
    return panel


def stack_panels(left: Image.Image, right: Image.Image) -> Image.Image:
    width, height = left.size
    canvas = Image.new("RGB", (width * 2, height), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (width, 0))
    return canvas


def generate_with_custom_engine(
    sample: Dict[str, Any],
    image: Image.Image,
    image_path: Path,
    processor: AutoProcessor,
    model: torch.nn.Module,
    max_new_tokens: int,
) -> str:
    conversation = build_conversation(sample, str(image_path))
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            use_cache=True,
        )
    prompt_len = int(inputs["input_ids"].shape[1])
    gen_ids = generated[:, prompt_len:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def generate_with_llamafactory_engine(
    sample: Dict[str, Any],
    image: Image.Image,
    chat_model: Any,
    max_new_tokens: int,
) -> str:
    messages, system_text = build_lf_messages_and_system(sample)
    responses = chat_model.chat(
        messages=messages,
        system=system_text,
        images=[image],
        do_sample=False,
        max_new_tokens=int(max_new_tokens),
        temperature=0.0,
    )
    return responses[0].response_text if responses else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--processor-path", type=str, default="")
    parser.add_argument("--engine", type=str, default="custom", choices=["custom", "llamafactory"])
    parser.add_argument("--template", type=str, default="qwen2_vl")
    parser.add_argument("--image-max-pixels", type=int, default=802816)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--sample-ids", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    dataset_jsonl = Path(args.dataset_jsonl)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(dataset_jsonl)
    sample_ids = [s.strip() for s in args.sample_ids.split(",") if s.strip()]
    selected = select_items(items, sample_ids=sample_ids, max_samples=int(args.max_samples))

    processor = None
    model = None
    chat_model = None
    if args.engine == "custom":
        processor_path = args.processor_path or args.adapter
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto" if args.device.startswith("cuda") else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, args.adapter)
        if not args.device.startswith("cuda"):
            model = model.to(args.device)
        model.eval()
    else:
        from llamafactory.chat import ChatModel

        infer_args = {
            "model_name_or_path": args.base_model,
            "adapter_name_or_path": args.adapter,
            "finetuning_type": "lora",
            "stage": "sft",
            "template": args.template,
            "infer_backend": "huggingface",
            "infer_dtype": "bfloat16",
            "trust_remote_code": True,
            "image_max_pixels": args.image_max_pixels,
        }
        chat_model = ChatModel(infer_args)

    results: List[Dict[str, Any]] = []
    agg_metrics: Dict[str, List[float]] = {}

    for sample in selected:
        sample_id = str(sample["id"])
        rel_image = sample["images"][0]
        image_path = (dataset_root / rel_image).resolve()
        image = Image.open(image_path).convert("RGB")
        if args.engine == "custom":
            assert processor is not None and model is not None
            pred_text = generate_with_custom_engine(
                sample=sample,
                image=image,
                image_path=image_path,
                processor=processor,
                model=model,
                max_new_tokens=int(args.max_new_tokens),
            )
        else:
            pred_text = generate_with_llamafactory_engine(
                sample=sample,
                image=image,
                chat_model=chat_model,
                max_new_tokens=int(args.max_new_tokens),
            )
        pred_obj, cleaned_pred_text = parse_generated_json(pred_text)
        gt_lines = extract_gt_lines(sample)
        state_lines = extract_state_lines(sample)
        pred_lines = list((pred_obj or {"lines": []}).get("lines", []))
        parse_ok = pred_obj is not None

        metrics = _sample_metrics(pred_lines=pred_lines, gt_lines=gt_lines, thresholds=[2.0, 4.0, 8.0])
        for key, value in metrics.items():
            agg_metrics.setdefault(key, []).append(float(value))

        gt_panel = render_panel(image=image, lines=gt_lines, state_lines=state_lines, title=f"{sample_id} | GT", line_color=(0, 120, 255))
        pred_panel = render_panel(
            image=image,
            lines=pred_lines,
            state_lines=state_lines,
            title=f"{sample_id} | Pred",
            line_color=(255, 60, 60),
        )
        combined = stack_panels(gt_panel, pred_panel)
        combined.save(viz_dir / f"{sample_id}.png")

        results.append(
            {
                "id": sample_id,
                "image": rel_image,
                "engine": args.engine,
                "parse_ok": parse_ok,
                "pred_text": pred_text,
                "pred_json_text": cleaned_pred_text,
                "pred_lines": pred_lines,
                "gt_lines": gt_lines,
                "state_lines": state_lines,
                "metrics": metrics,
            }
        )

    summary = {
        "num_samples": len(results),
        "num_parse_ok": sum(1 for item in results if item["parse_ok"]),
        "mean_metrics": {key: sum(values) / max(1, len(values)) for key, values in agg_metrics.items()},
    }

    with (output_dir / "predictions.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
