"""通用 helper。

这个文件放的是和具体数据阶段无关的基础能力：
- 目录创建
- json/jsonl 读写
- ShareGPT 样本组装
- system prompt / images 目录处理
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def ensure_dir(path: Path) -> None:
    """确保目录存在，不存在时递归创建。"""
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    """读取一个 json 文件并返回字典。

    使用 `utf-8-sig`，这样即使文件带 BOM 也能正常读取。
    """
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """把字典写成格式化 json 文件。"""
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 jsonl 文件，返回由多行对象组成的列表。"""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """写出 jsonl 文件，并返回写入的行数。"""
    count = 0
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def sanitize_name(name: str) -> str:
    """把任意名字清洗成适合出现在数据集名里的安全字符串。"""
    out = []
    for ch in str(name):
        out.append(ch if ch.isalnum() or ch in ("_", "-") else "_")
    return "".join(out).strip("_") or "dataset"


def extract_message_content(row: Dict[str, Any], role: str) -> str:
    """从一条 ShareGPT 样本中提取指定 role 的文本内容。

    如果不存在目标 role，就返回空字符串。
    """
    want = str(role).strip().lower()
    for msg in row.get("messages", []):
        if str(msg.get("role", "")).strip().lower() == want:
            return str(msg.get("content", ""))
    return ""


def resolve_optional_text(*, inline_text: str = "", file_path: str = "", fallback: str = "") -> str:
    """统一处理一段可选文本的优先级。

    优先级顺序是：
    1. `file_path`
    2. `inline_text`
    3. `fallback`
    """
    if str(file_path).strip():
        return Path(str(file_path)).read_text(encoding="utf-8").strip()
    if str(inline_text).strip():
        return str(inline_text).strip()
    return str(fallback).strip()


def link_or_copy_images(input_root: Path, output_root: Path, mode: str) -> str:
    """把输入数据集里的 `images/` 暴露到输出根目录下。

    支持三种模式：
    - `symlink`
    - `copy`
    - `none`

    返回值描述最终实际采用的方式。
    """
    src = input_root / "images"
    dst = output_root / "images"
    if not src.exists() or str(mode) == "none":
        return "none"
    if dst.exists() or dst.is_symlink():
        return "existing"
    if str(mode) == "symlink":
        try:
            dst.symlink_to(src, target_is_directory=True)
            return "symlink"
        except OSError:
            shutil.copytree(src, dst)
            return "copy_fallback"
    shutil.copytree(src, dst)
    return "copy"


def build_sharegpt_dataset_info(output_root: Path, splits: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """生成 LLaMAFactory 可读取的 `dataset_info.json` 内容。"""
    base = sanitize_name(output_root.name)
    info: Dict[str, Dict[str, Any]] = {}
    for split in splits:
        info[f"unimapgen_{base}_{split}"] = {
            "file_name": str((output_root / f"{split}.jsonl").resolve()),
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    return info


def make_sharegpt_record(
    *,
    sample_id: str,
    image_rel_path: str,
    user_text: str,
    assistant_payload: Any,
    system_prompt: str = "",
) -> Dict[str, Any]:
    """组装一条 ShareGPT 格式样本。

    参数含义：
    - `sample_id`: 样本主键
    - `image_rel_path`: 图像相对路径
    - `user_text`: user 消息内容
    - `assistant_payload`: assistant 内容，可以是字符串，也可以是对象
    - `system_prompt`: 可选的 system 消息
    """
    if isinstance(assistant_payload, str):
        assistant_text = assistant_payload
    else:
        assistant_text = json.dumps(assistant_payload, ensure_ascii=False, separators=(",", ":"))

    messages: List[Dict[str, str]] = []
    if str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": str(user_text)})
    messages.append({"role": "assistant", "content": assistant_text})
    return {
        "id": str(sample_id),
        "messages": messages,
        "images": [str(image_rel_path)],
    }
