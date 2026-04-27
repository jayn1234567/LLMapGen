#!/usr/bin/env python3
"""Convert cloud server data format to LLaVA training format (JSONL).

Cloud format:
{
  "id": "...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image>\n..."},
    {"role": "assistant", "content": "[{points:[[x,y],...]},...]"}
  ],
  "images": ["img/xxx.png"]
}

Target format:
{
  "id": "...",
  "image": "img/xxx.png",
  "conversations": [
    {"from": "human", "value": "<image>\n..."},
    {"from": "gpt", "value": "[{\"points\":[[x,y],...],\"category\":\"CenterLine\"},...]"}
  ]
}
"""

import argparse
import json
from pathlib import Path


def fix_assistant_json(text: str, add_category: str = "CenterLine") -> str:
    """Add category wrapper and fix unquoted keys if needed."""
    text = text.strip()
    if not text:
        return text

    # Quick check: does it already have "category"?
    # If not, try to parse and inject
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try fixing unquoted keys: points -> "points"
        import re
        fixed = re.sub(r'(?<=\{)\s*points\s*:', '"points":', text)
        fixed = re.sub(r'(?<=,)\s*points\s*:', '"points":', fixed)
        try:
            data = json.loads(fixed)
        except json.JSONDecodeError:
            return text  # give up, return as-is

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "category" not in item and add_category:
                item["category"] = add_category

    return json.dumps(data, ensure_ascii=False)


def convert_record(record: dict, add_category: str = "CenterLine", skip_system: bool = True) -> dict:
    images = record.get("images", [])
    image_path = images[0] if images else ""

    conversations = []
    for msg in record.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            if skip_system:
                continue
            from_role = "system"
        elif role == "user":
            from_role = "human"
        elif role == "assistant":
            from_role = "gpt"
            content = fix_assistant_json(content, add_category)
        else:
            from_role = role

        conversations.append({"from": from_role, "value": content})

    return {
        "id": record.get("id", ""),
        "image": image_path,
        "conversations": conversations,
    }


def load_json_or_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    if raw.strip().startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Convert cloud data to LLaVA JSONL format")
    parser.add_argument("--input", required=True, help="Input file (JSON array or JSONL)")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--add-category", default="CenterLine",
                        help="Inject 'category' key into assistant predictions (default: CenterLine). Use empty string to skip.")
    parser.add_argument("--keep-system", action="store_true",
                        help="Keep system messages in conversations (default: skip)")
    args = parser.parse_args()

    records = load_json_or_jsonl(args.input)
    add_cat = args.add_category or None

    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            converted = convert_record(rec, add_category=args.add_category, skip_system=not args.keep_system)
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")

    print(f"Done: {len(records)} records -> {args.output}")


if __name__ == "__main__":
    main()
