#!/usr/bin/env python3
import argparse
import heapq
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer
from transformers import AutoConfig
import os

try:
    from safetensors.torch import load_file as safe_load_file
except ImportError:  # pragma: no cover
    safe_load_file = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mllm.torch_runtime import maybe_disable_cudnn_from_env

maybe_disable_cudnn_from_env(torch)

from mllm import conversation as conversation_lib
from mllm.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from mllm.coord_utils import (
    COORD_MODE_PIXEL,
    COORD_MODE_NORM1000,
    DEFAULT_COORD_RANGE,
    convert_items,
    convert_payload_text,
    payload_to_text,
    record_coord_config,
)
from mllm.coordinate_tokens import (
    COORDINATE_TOKEN_MODE_ANGLE,
    append_coordinate_token_instruction,
    decode_coordinate_tokens,
    normalize_coordinate_token_mode,
    tokenizer_has_coordinate_vocabulary,
)
from mllm.mm_utils import process_images, tokenizer_image_token
from mllm.model.builder import load_pretrained_model
from mllm.model.language_model.qwen_family import (
    is_qwen3_or_newer_family,
    qwen_family_from_config,
    qwen_family_from_text,
    qwen_multimodal_model_class,
)
from mllm.model.qwen_token_utils import qwen_tokenizer_kwargs, sync_qwen_token_config
from mllm.reward.map_schema import parse_map_json as parse_map_schema_json
from scripts.tools.map_visualization import offset_lines, record_origin

DEFAULT_PROMPT = DEFAULT_IMAGE_TOKEN


def _env_flag(name, default="0"):
    return str(os.environ.get(name, default)).lower() in ("1", "true", "yes", "on")


if _env_flag("MLLM_DISABLE_CUDNN", "0"):
    torch.backends.cudnn.enabled = False


def silence_non_primary_rank_output():
    """Keep normal inference logs on global rank 0 only."""
    if not _env_flag("MLLM_LOG_RANK0_ONLY", "1"):
        return
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank == 0:
        return
    sys.stdout.flush()
    sys.stdout = open(os.devnull, "w")
    if _env_flag("MLLM_SUPPRESS_NONZERO_STDERR", "0"):
        sys.stderr.flush()
        sys.stderr = open(os.devnull, "w")


def rank_json_path(path: Path, rank: int) -> Path:
    return path.with_name(f"{path.stem}_rank{rank:05d}{path.suffix}")


def resolve_record_image(record: dict) -> str:
    """Resolve the single image consumed by the Stage-A inference engine."""
    image = record.get("image")
    if image is None:
        images = record.get("images")
        if isinstance(images, str):
            image = images
        elif isinstance(images, list):
            if len(images) != 1:
                raise ValueError(
                    "Stage-A checkpoint inference expects one image per record; "
                    f"got {len(images)} in record {record.get('id', '<unknown>')}."
                )
            image = images[0]
    if not isinstance(image, str) or not image.strip():
        raise ValueError(
            "Inference record has no usable image path in 'image' or single-item 'images': "
            f"{record.get('id', '<unknown>')}"
        )
    return image.strip()


def iter_json_records(path: Path):
    """Yield indexed JSON records without materializing a JSONL file."""
    with path.open("r", encoding="utf-8-sig") as handle:
        first = handle.read(1)
        while first and first.isspace():
            first = handle.read(1)
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"Expected a JSON array in {path}")
            for index, record in enumerate(payload):
                if not isinstance(record, dict):
                    raise ValueError(f"Record {index} in {path} is not an object")
                yield index, record
            return

        index = 0
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number} in {path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Record {index} in {path} is not an object")
            yield index, record
            index += 1


def load_completed_jsonl_indices(path: Path) -> set[int]:
    """Read completed stream rows and truncate only an incomplete final row."""
    if not path.is_file():
        return set()
    completed: set[int] = set()
    with path.open("rb+") as handle:
        while True:
            offset = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if not raw_line.endswith(b"\n"):
                    handle.seek(offset)
                    handle.truncate()
                    break
                raise ValueError(f"Malformed completed inference row in {path}") from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("idx"), int):
                raise ValueError(f"Completed inference row has no integer idx: {path}")
            index = int(payload["idx"])
            if index in completed:
                raise ValueError(f"Duplicate completed inference idx={index}: {path}")
            completed.add(index)
    return completed


def merge_rank_jsonl(output_path: Path, world_size: int) -> int:
    """Merge rank-local streams in input order with bounded memory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handles = []
    heap = []

    def read_next(handle, rank: int):
        for raw_line in handle:
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed rank inference row: {handle.name}") from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("idx"), int):
                raise ValueError(f"Rank inference row has no integer idx: {handle.name}")
            return int(payload["idx"]), rank, payload
        return None

    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    count = 0
    last_idx = None
    try:
        for rank in range(world_size):
            rank_path = rank_json_path(output_path, rank)
            if not rank_path.is_file():
                raise FileNotFoundError(f"Missing rank inference output: {rank_path}")
            handle = rank_path.open("r", encoding="utf-8")
            handles.append(handle)
            first = read_next(handle, rank)
            if first is not None:
                heapq.heappush(heap, first)

        with temporary.open("w", encoding="utf-8") as destination:
            while heap:
                index, rank, payload = heapq.heappop(heap)
                if last_idx is not None and index <= last_idx:
                    raise ValueError(f"Duplicate or unsorted inference idx={index}")
                destination.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                destination.write("\n")
                count += 1
                last_idx = index
                next_item = read_next(handles[rank], rank)
                if next_item is not None:
                    heapq.heappush(heap, next_item)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(output_path)
    finally:
        for handle in handles:
            handle.close()
        if temporary.exists():
            temporary.unlink()
    return count


def write_json_atomic(path: Path, payload) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def wait_for_rank_jsons(
    output_json_path: Path,
    world_size: int,
    timeout_seconds: int,
) -> list[Path]:
    rank_paths = [rank_json_path(output_json_path, rank) for rank in range(world_size)]
    deadline = time.monotonic() + timeout_seconds
    missing = [path for path in rank_paths if not path.is_file()]
    while missing and time.monotonic() < deadline:
        time.sleep(2)
        missing = [path for path in rank_paths if not path.is_file()]
    if missing:
        raise TimeoutError(
            "Timed out waiting for inference rank summaries: "
            + ", ".join(str(path) for path in missing)
        )
    return rank_paths


def normalize_prediction_text(text: str, max_coordinate: int = 1000) -> str:
    cleaned = decode_coordinate_tokens(text, max_coordinate=max_coordinate).strip()
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


def resolve_coordinate_token_settings(tokenizer, config) -> tuple[str, int]:
    max_coordinate = int(getattr(config, "coordinate_token_max", 1000) or 1000)
    mode = normalize_coordinate_token_mode(
        getattr(config, "coordinate_token_mode", "none")
    )
    if mode != COORDINATE_TOKEN_MODE_ANGLE and tokenizer_has_coordinate_vocabulary(
        tokenizer,
        max_coordinate=max_coordinate,
    ):
        mode = COORDINATE_TOKEN_MODE_ANGLE
    return mode, max_coordinate


def extract_json_array(text: str) -> str:
    start = text.find("[")
    if start < 0:
        return text.strip()

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1].strip()

    return text[start:].strip()


def extract_json_payload(text: str) -> str:
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not starts:
        return text.strip()
    start = min(starts)

    depth_stack = []
    in_string = False
    escape = False
    pairs = {"{": "}", "[": "]"}
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in pairs:
            depth_stack.append(pairs[ch])
        elif depth_stack and ch == depth_stack[-1]:
            depth_stack.pop()
            if not depth_stack:
                return text[start:idx + 1].strip()

    return text[start:].strip()


def prediction_preview(text: str, limit: int = 240) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def completion_token_ids(
    output_ids: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask=None,
    row_index: int = 0,
):
    """Return one completion from batched HF or completion-only generation."""
    output = output_ids[row_index] if output_ids.ndim == 2 else output_ids
    prompt_padded = input_ids[row_index] if input_ids.ndim == 2 else input_ids
    if attention_mask is None:
        prompt = prompt_padded
    else:
        mask = attention_mask[row_index] if attention_mask.ndim == 2 else attention_mask
        prompt = prompt_padded[mask.bool()]
    for candidate, mode in (
        (prompt_padded, "completion_sliced_padded"),
        (prompt, "completion_sliced"),
    ):
        if output.numel() >= candidate.numel() and torch.equal(output[:candidate.numel()], candidate):
            return output[candidate.numel():], mode
    return output, "completion_only"


def iter_batches(items, batch_size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def left_pad_token_sequences(sequences: list[torch.Tensor], pad_token_id: int, device):
    if not sequences:
        raise ValueError("Cannot pad an empty inference batch.")
    max_length = max(int(sequence.numel()) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), max_length),
        int(pad_token_id),
        dtype=sequences[0].dtype,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(sequences), max_length),
        dtype=torch.long,
        device=device,
    )
    for row_index, sequence in enumerate(sequences):
        length = int(sequence.numel())
        input_ids[row_index, -length:] = sequence.to(device=device)
        attention_mask[row_index, -length:] = 1
    return input_ids, attention_mask


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
    safe_index_path = checkpoint_dir / "model.safetensors.index.json"
    bin_index_path = checkpoint_dir / "pytorch_model.bin.index.json"
    if safe_index_path.exists() or bin_index_path.exists():
        index_path = safe_index_path if safe_index_path.exists() else bin_index_path
        with index_path.open("r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        if not weight_map:
            raise ValueError(f"Checkpoint index {index_path} has no weight_map entries")
        shard_names = sorted(set(weight_map.values()))
        merged_state = {}
        for shard_name in shard_names:
            shard_path = checkpoint_dir / shard_name
            shard_state = _load_checkpoint_shard(shard_path)
            merged_state.update(shard_state)
            del shard_state
        return merged_state
    shard_paths = _sharded_model_weight_files(checkpoint_dir)
    if shard_paths:
        merged_state = {}
        for shard_path in shard_paths:
            shard_state = _load_checkpoint_shard(shard_path)
            merged_state.update(shard_state)
            del shard_state
        return merged_state
    raise FileNotFoundError(f"No model weights found under {checkpoint_dir}")


def _sharded_model_weight_files(checkpoint_dir: Path) -> list[Path]:
    safetensor_shards = sorted(checkpoint_dir.glob("model-*-of-*.safetensors"))
    if safetensor_shards:
        return safetensor_shards
    return sorted(checkpoint_dir.glob("pytorch_model-*-of-*.bin"))


def _load_checkpoint_shard(shard_path: Path):
    if not shard_path.exists():
        raise FileNotFoundError(f"Checkpoint shard listed in index is missing: {shard_path}")
    if shard_path.suffix == ".safetensors":
        if safe_load_file is None:
            raise ImportError("safetensors is required to load sharded safetensors checkpoints")
        return safe_load_file(str(shard_path), device="cpu")
    return torch.load(shard_path, map_location="cpu")


def _resolve_base_model_path(base_model_path: str, checkpoint_dir: Path) -> Path:
    override = os.environ.get("QWEN_BASE_MODEL_PATH") or os.environ.get("MODEL_BASE") or os.environ.get("QWEN3VL_EXTRACTED_LLM_PATH")
    if override:
        override_path = Path(override).expanduser()
        if override_path.exists():
            return override_path

    candidate = Path(base_model_path)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        repaired = _repair_missing_extracted_base_path(candidate, checkpoint_dir)
        if repaired is not None:
            return repaired
        return candidate

    rel_to_ckpt = (checkpoint_dir / candidate).resolve()
    if rel_to_ckpt.exists():
        return rel_to_ckpt

    rel_to_repo = (REPO_ROOT / candidate).resolve()
    if rel_to_repo.exists():
        return rel_to_repo

    return candidate


def _repair_missing_extracted_base_path(candidate: Path, checkpoint_dir: Path) -> Path | None:
    name = candidate.name
    is_legacy_extracted = name.startswith(".qwen3_llm_extracted_")
    is_stable_extracted = name.endswith("_llm_extracted")
    if not (is_legacy_extracted or is_stable_extracted):
        return None

    roots = []
    for root_env in ("QWEN3VL_EXTRACTED_LLM_ROOT",):
        root = os.environ.get(root_env)
        if root:
            roots.append(Path(root).expanduser())
    roots.append(candidate.parent)
    roots.extend([checkpoint_dir, *list(checkpoint_dir.parents)[:5]])
    roots.extend([REPO_ROOT, REPO_ROOT / "checkpoints", REPO_ROOT / "outputs"])

    seen = set()
    deduped_roots = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            deduped_roots.append(root)

    if is_stable_extracted:
        for root in deduped_roots:
            repaired = root / name
            if repaired.exists():
                print(f"[WARN] Repaired missing base_model_name_or_path to {repaired}")
                return repaired

    for root in deduped_roots:
        if not root.exists() or not root.is_dir():
            continue
        matches = sorted(root.glob("*_llm_extracted"))
        if not matches and is_legacy_extracted:
            matches = sorted(root.glob(".qwen3_llm_extracted_*"))
        for repaired in matches:
            if repaired.exists():
                print(f"[WARN] Repaired missing extracted LLM base path {candidate} -> {repaired}")
                return repaired
    return None


def _read_checkpoint_metadata(checkpoint_dir: Path) -> dict:
    qwen_metadata_path = checkpoint_dir / "qwen_multimodal_checkpoint.json"
    if qwen_metadata_path.exists():
        payload = json.loads(qwen_metadata_path.read_text(encoding="utf-8"))
        payload["_metadata_source"] = qwen_metadata_path.name
        return payload

    legacy_metadata_path = checkpoint_dir / "llava_checkpoint.json"
    if legacy_metadata_path.exists():
        payload = json.loads(legacy_metadata_path.read_text(encoding="utf-8"))
        payload["_metadata_source"] = legacy_metadata_path.name
        print("[WARN] Using legacy llava_checkpoint.json metadata for a Qwen multimodal checkpoint.")
        return payload

    return {}


def _has_model_weights(checkpoint_dir: Path) -> bool:
    if any(
        (checkpoint_dir / filename).exists()
        for filename in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
    ):
        return True
    return bool(_sharded_model_weight_files(checkpoint_dir))


def _has_adapter_weights(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "adapter_model.safetensors").exists() or (checkpoint_dir / "adapter_model.bin").exists()


def _looks_like_full_mllm_checkpoint(checkpoint_dir: Path) -> bool:
    if not _has_model_weights(checkpoint_dir):
        return False
    try:
        config = AutoConfig.from_pretrained(str(checkpoint_dir), local_files_only=True)
    except Exception:
        return False

    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type.startswith(("mllm_", "llava_")) or "mllm" in model_type or "llava" in model_type:
        return True

    return any(
        getattr(config, attr, None) is not None
        for attr in ("mm_vision_tower", "vision_tower", "mm_projector_type")
    )


def _config_overrides_from_args(args) -> dict:
    return {
        "mm_vision_tower": args.vision_tower or None,
        "vision_tower": args.vision_tower or None,
        "vision_tower_checkpoint": getattr(args, "vision_tower_checkpoint", "") or None,
        "mm_vision_tower_type": getattr(args, "mm_vision_tower_type", "") or None,
        "input_image_size": args.input_image_size,
        "disable_deepstack": args.disable_deepstack,
        "deepstack_visual_indexes": args.deepstack_visual_indexes,
        "multi_vision_towers": getattr(args, "multi_vision_towers", "") or None,
        "multi_vision_tower_types": getattr(args, "multi_vision_tower_types", "") or None,
        "multi_vision_input_image_sizes": getattr(args, "multi_vision_input_image_sizes", "") or None,
        "multi_vision_primary_index": getattr(args, "multi_vision_primary_index", None),
        "multi_vision_hidden_size": getattr(args, "multi_vision_hidden_size", None),
        "multi_vision_target_grid": getattr(args, "multi_vision_target_grid", None),
        "multi_vision_fusion": getattr(args, "multi_vision_fusion", "") or None,
        "multi_vision_router_temperature": getattr(args, "multi_vision_router_temperature", None),
        "multi_vision_router_hidden_ratio": getattr(args, "multi_vision_router_hidden_ratio", None),
        "multi_vision_router_use_diff": getattr(args, "multi_vision_router_use_diff", None),
        "multi_vision_dropout": getattr(args, "multi_vision_dropout", None),
        "vision_layer_fusion_indexes": getattr(args, "vision_layer_fusion_indexes", None),
        "vision_layer_fusion_type": getattr(args, "vision_layer_fusion_type", "") or None,
    }


def _apply_config_overrides(config, overrides: dict):
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)


def _apply_checkpoint_metadata_defaults(config, metadata: dict):
    if not metadata:
        return
    for key in (
        "mm_vision_tower",
        "vision_tower",
        "vision_tower_checkpoint",
        "mm_vision_tower_type",
        "input_image_size",
        "disable_deepstack",
        "deepstack_visual_indexes",
        "multi_vision_towers",
        "multi_vision_tower_types",
        "multi_vision_input_image_sizes",
        "multi_vision_primary_index",
        "multi_vision_hidden_size",
        "multi_vision_target_grid",
        "multi_vision_fusion",
        "multi_vision_router_temperature",
        "multi_vision_router_hidden_ratio",
        "multi_vision_router_use_diff",
        "multi_vision_dropout",
    ):
        value = metadata.get(key)
        if value is not None and getattr(config, key, None) is None:
            setattr(config, key, value)


def _same_path_or_value(left, right) -> bool:
    if left is None or right is None:
        return False
    left_s = str(left)
    right_s = str(right)
    if left_s == right_s:
        return True
    try:
        return Path(left_s).expanduser().resolve() == Path(right_s).expanduser().resolve()
    except Exception:
        return False


def _has_tokenizer_files(path: Path) -> bool:
    return any((path / name).exists() for name in ("tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt"))


def _load_lora_tokenizer(checkpoint_dir: Path, resolved_model_base: Path, config):
    tokenizer_source = checkpoint_dir if _has_tokenizer_files(checkpoint_dir) else resolved_model_base
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source),
        use_fast=False,
        local_files_only=True,
        **qwen_tokenizer_kwargs(str(tokenizer_source), config=config),
    )
    return tokenizer, tokenizer_source


def _is_dinov3_config(config) -> bool:
    values = [
        getattr(config, "mm_vision_tower_type", ""),
        getattr(config, "mm_vision_tower", ""),
        getattr(config, "vision_tower", ""),
    ]
    return any("dinov3" in str(value).lower() for value in values if value is not None)


def _runtime_dtype(config, device: str):
    if not str(device).startswith(("cuda", "npu")):
        return torch.float32
    if _is_dinov3_config(config):
        return torch.bfloat16
    return torch.float16


def _load_full_finetune_model(checkpoint_dir: Path, device: str, config_overrides=None):
    checkpoint_dir_str = str(checkpoint_dir.resolve())
    config = AutoConfig.from_pretrained(checkpoint_dir_str, local_files_only=True)
    metadata = _read_checkpoint_metadata(checkpoint_dir)
    _apply_checkpoint_metadata_defaults(config, metadata)
    _apply_config_overrides(config, config_overrides or {})
    if not getattr(config, "mm_vision_tower", None) and not getattr(config, "vision_tower", None):
        raise RuntimeError(
            "Full Qwen multimodal checkpoint is missing a vision tower path in config. "
            "Pass --vision_tower explicitly; do not rely on legacy generic fallback loading."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir_str,
        use_fast=False,
        local_files_only=True,
        **qwen_tokenizer_kwargs(checkpoint_dir_str, config=config),
    )
    sync_qwen_token_config(
        tokenizer=tokenizer,
        config=config,
        model_name_or_path=checkpoint_dir_str,
    )
    config.unfreeze_mm_vision_tower = False
    config.tune_vision_tower = False
    config.fastvit_pretrained = False
    config.fastvit_pretrained_path = None

    qwen_family = qwen_family_from_config(config) or "qwen2"
    model = qwen_multimodal_model_class(qwen_family)(config)
    model.resize_token_embeddings(len(tokenizer))
    sync_qwen_token_config(
        tokenizer=tokenizer,
        model=model,
        model_name_or_path=checkpoint_dir_str,
    )

    vision_tower = model.get_vision_tower()
    if vision_tower is not None and not vision_tower.is_loaded:
        vision_tower.load_model()
    if vision_tower is not None and hasattr(vision_tower, 'set_llm_hidden_size'):
        vision_tower.set_llm_hidden_size(model.config.hidden_size)

    state_dict = _load_state_dict(checkpoint_dir)
    _report_full_checkpoint_coverage(model, state_dict, model.config, metadata)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if "lm_head.weight" in missing and getattr(model.config, "tie_word_embeddings", False):
        model.tie_weights()
        missing = [key for key in missing if key != "lm_head.weight"]
    if missing:
        print(f"[WARN] Missing keys after full-finetune load: {missing[:20]}")
    if unexpected:
        print(f"[WARN] Unexpected keys after full-finetune load: {unexpected[:20]}")

    dtype = _runtime_dtype(model.config, device)
    model = model.to(dtype=dtype)
    model = model.to(device)
    model.eval()

    image_processor = model.get_model().get_vision_tower().image_processor
    return tokenizer, model, image_processor


def _normalize_non_lora_state_dict(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("base_model."):
            key = key[len("base_model."):]
        if key.startswith("model.model."):
            key = key[len("model."):]
        normalized[key] = value
    return normalized


def _state_dict_has_prefix(state_dict, prefix: str) -> bool:
    return any(key.startswith(prefix) for key in state_dict)


def _report_full_checkpoint_coverage(model, state_dict, config, metadata):
    model_state = model.state_dict()
    matched = []
    mismatched = []
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            continue
        if tuple(value.shape) == tuple(target.shape):
            matched.append(key)
        else:
            mismatched.append(f"{key}:{tuple(value.shape)}->{tuple(target.shape)}")

    print(
        f"Loaded {len(matched)}/{len(model_state)} model tensors from full-finetune checkpoint "
        f"({len(state_dict)} checkpoint tensors)."
    )
    if mismatched:
        raise RuntimeError(f"Shape-mismatched full checkpoint tensors: {mismatched[:20]}")

    critical_missing = []
    if not _state_dict_has_prefix(state_dict, "model.mm_projector."):
        critical_missing.append("model.mm_projector.*")

    deepstack_enabled = (
        not getattr(config, "disable_deepstack", False)
        and getattr(config, "deepstack_visual_indexes", None) is not None
    )
    if deepstack_enabled and not _state_dict_has_prefix(state_dict, "model.vision_tower.deepstack_mergers."):
        critical_missing.append("model.vision_tower.deepstack_mergers.*")

    has_vit_weights = (
        _state_dict_has_prefix(state_dict, "model.vision_tower.vision_tower.")
        or _state_dict_has_prefix(state_dict, "model.vision_tower.vision_towers.")
    )
    explicit_external_vit = bool(metadata) and metadata.get("bundled_vision_tower") is False
    if not has_vit_weights and not explicit_external_vit:
        critical_missing.append("model.vision_tower.vision_tower.*")
    if explicit_external_vit:
        print("Checkpoint metadata declares external ViT weights; using the configured vision_tower weights.")

    if critical_missing:
        raise RuntimeError(
            "Full-finetune checkpoint is missing critical multimodal weights: "
            + ", ".join(critical_missing)
        )


def _add_multimodal_tokens(tokenizer, config):
    if getattr(config, "mm_use_im_patch_token", True):
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if getattr(config, "mm_use_im_start_end", False):
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)


def _load_compatible_state_dict(model, state_dict, *, skip_prefixes=(), source_name="checkpoint"):
    model_state = model.state_dict()
    compatible = {}
    partial = []
    skipped_shape = []

    for key, value in state_dict.items():
        if skip_prefixes and any(key.startswith(prefix) for prefix in skip_prefixes):
            continue
        target = model_state.get(key)
        if target is None:
            continue
        if tuple(value.shape) == tuple(target.shape):
            compatible[key] = value
            continue
        if (
            key in {"model.embed_tokens.weight", "lm_head.weight"}
            and value.ndim == target.ndim == 2
            and value.shape[1] == target.shape[1]
        ):
            merged = target.detach().cpu().clone()
            rows = min(value.shape[0], target.shape[0])
            merged[:rows].copy_(value[:rows].to(dtype=merged.dtype))
            compatible[key] = merged
            partial.append(f"{key}:{tuple(value.shape)}->{tuple(target.shape)}")
            continue
        skipped_shape.append(f"{key}:{tuple(value.shape)}->{tuple(target.shape)}")

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if partial:
        print(f"Partially loaded {len(partial)} resized tensors from {source_name}: {partial[:8]}")
    if skipped_shape:
        print(f"[WARN] Skipped {len(skipped_shape)} shape-mismatched tensors from {source_name}: {skipped_shape[:8]}")
    return missing, unexpected, len(compatible)


def _load_lora_finetune_model(checkpoint_dir: Path, device: str, config_overrides=None):
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    non_lora_path = checkpoint_dir / "non_lora_trainables.bin"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {checkpoint_dir}")
    if not non_lora_path.exists():
        raise FileNotFoundError(f"non_lora_trainables.bin not found in {checkpoint_dir}")

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    model_base = adapter_config.get("base_model_name_or_path")
    if not model_base:
        raise ValueError(f"Missing base_model_name_or_path in {adapter_config_path}")

    checkpoint_dir_str = str(checkpoint_dir.resolve())
    resolved_model_base = _resolve_base_model_path(model_base, checkpoint_dir)
    config = AutoConfig.from_pretrained(checkpoint_dir_str, local_files_only=True)
    metadata = _read_checkpoint_metadata(checkpoint_dir)
    _apply_checkpoint_metadata_defaults(config, metadata)
    _apply_config_overrides(config, config_overrides or {})
    config.unfreeze_mm_vision_tower = False
    config.tune_vision_tower = False
    tokenizer, tokenizer_source = _load_lora_tokenizer(checkpoint_dir, resolved_model_base, config)
    _add_multimodal_tokens(tokenizer, config)

    dtype = _runtime_dtype(config, device)
    qwen_family = qwen_family_from_config(config) or "qwen2"
    model = qwen_multimodal_model_class(qwen_family)(config)
    model.resize_token_embeddings(len(tokenizer))
    sync_qwen_token_config(
        tokenizer=tokenizer,
        model=model,
        model_name_or_path=str(tokenizer_source),
    )

    vision_tower = model.get_vision_tower()
    if vision_tower is not None and not vision_tower.is_loaded:
        vision_tower.load_model()
    if vision_tower is not None and hasattr(vision_tower, "set_llm_hidden_size"):
        vision_tower.set_llm_hidden_size(model.config.hidden_size)

    base_metadata = _read_checkpoint_metadata(resolved_model_base) if resolved_model_base.is_dir() else {}
    base_vision_tower = base_metadata.get("mm_vision_tower") or base_metadata.get("vision_tower")
    requested_vision_tower = getattr(config, "mm_vision_tower", None) or getattr(config, "vision_tower", None)
    skip_prefixes = []
    if base_vision_tower and requested_vision_tower and not _same_path_or_value(base_vision_tower, requested_vision_tower):
        skip_prefixes.append("model.vision_tower.")
        print(
            "[WARN] LoRA base checkpoint vision tower differs from requested vision_tower; "
            "skipping base model.vision_tower.* tensors."
        )

    base_state = _load_state_dict(resolved_model_base)
    _, _, loaded_base = _load_compatible_state_dict(
        model,
        base_state,
        skip_prefixes=tuple(skip_prefixes),
        source_name=f"base model {resolved_model_base}",
    )
    print(f"Loaded {loaded_base} compatible base tensors for LoRA inference.")

    non_lora_state = torch.load(non_lora_path, map_location="cpu")
    non_lora_state = _normalize_non_lora_state_dict(non_lora_state)
    missing, unexpected = model.load_state_dict(non_lora_state, strict=False)
    unexpected = [key for key in unexpected if "lora_" not in key]
    if unexpected:
        print(f"[WARN] Unexpected non-LoRA keys: {unexpected[:20]}")
    loaded = len(non_lora_state)
    print(f"Loaded {loaded} non-LoRA trainable tensors from LoRA checkpoint.")

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, checkpoint_dir_str, local_files_only=True)
    model = model.merge_and_unload()
    model = model.to(dtype=dtype)
    model = model.to(device)
    model.eval()

    image_processor = model.get_model().get_vision_tower().image_processor
    return tokenizer, model, image_processor


def load_model_components(checkpoint_dir: Path, manifest: dict, device: str, config_overrides=None):
    if _has_adapter_weights(checkpoint_dir):
        return _load_lora_finetune_model(checkpoint_dir, device, config_overrides=config_overrides)

    if _has_model_weights(checkpoint_dir):
        if not _looks_like_full_mllm_checkpoint(checkpoint_dir):
            raise RuntimeError(
                f"{checkpoint_dir} contains full model weights but does not look like a Qwen multimodal checkpoint. "
                "Refusing to use the legacy generic loader because it can silently skip projector/ViT weights. "
                "Check config.json for mm_vision_tower/vision_tower/mm_projector_type or use the correct checkpoint dir."
            )
        return _load_full_finetune_model(checkpoint_dir, device, config_overrides=config_overrides)

    model_base = manifest.get("model_name_or_path")
    if not model_base:
        raise RuntimeError(
            f"{checkpoint_dir} has neither LoRA adapter weights nor full model weights. "
            "Legacy base+projector loading requires an inference_manifest.json/run_config.json with model_name_or_path."
        )
    model_name = f"mllm_{checkpoint_dir.name}"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=str(checkpoint_dir),
        model_base=model_base,
        model_name=model_name,
        device_map={"": device},
        device=device,
        model_config_overrides=config_overrides,
    )
    model.eval()
    return tokenizer, model, image_processor



def _validate_points(points):
    if not isinstance(points, list) or not points:
        raise ValueError("missing points")
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("invalid point format")
        if not all(isinstance(v, (int, float)) for v in point):
            raise ValueError("point coordinates must be numeric")


def resolve_coord_config(record, args):
    cfg = record_coord_config(
        record,
        default_mode=COORD_MODE_PIXEL,
        default_patch_size=args.patch_size,
        default_coord_range=args.coord_range,
    )
    if args.coord_mode != "auto":
        cfg["coord_mode"] = args.coord_mode
        cfg["coord_range"] = args.coord_range
    return cfg


def parse_map_json(
    prediction_text: str,
    map_task: str = "lane_intersection",
    patch_size: int = 256,
    coord_mode: str = COORD_MODE_PIXEL,
    coord_range: int = DEFAULT_COORD_RANGE,
):
    parse_result = parse_map_schema_json(
        prediction_text,
        map_task=map_task,
        patch_size=patch_size,
        coord_mode=coord_mode,
        coord_range=coord_range,
    )
    if not parse_result.ok:
        raise ValueError(parse_result.error or "prediction parse failed")
    return parse_result.items


def parse_centerline_json(
    prediction_text: str,
    map_task: str = "lane_intersection",
    patch_size: int = 256,
    coord_mode: str = COORD_MODE_PIXEL,
    coord_range: int = DEFAULT_COORD_RANGE,
):
    return parse_map_json(
        prediction_text,
        map_task=map_task,
        patch_size=patch_size,
        coord_mode=coord_mode,
        coord_range=coord_range,
    )


def sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def build_eval_payload(records, args, evaluate_records_fn, evaluate_lane_intersection_records_fn):
    eval_kwargs = dict(
        meter_per_pixel=args.eval_meter_per_pixel,
        buffer_size=args.eval_buffer_size,
        match_threshold=args.eval_match_threshold,
    )
    if args.map_task == "lane_intersection":
        map_eval = evaluate_lane_intersection_records_fn(records, **eval_kwargs)
        return {
            "centerline_eval": map_eval["lane"],
            "intersection_eval": map_eval["intersection"],
            "lane_intersection_eval": map_eval["lane_intersection"],
            "map_eval": map_eval,
        }
    return evaluate_records_fn(records, **eval_kwargs)


def print_eval_payload(eval_payload, args, print_eval_table_fn, print_lane_intersection_eval_tables_fn):
    if args.map_task == "lane_intersection":
        print_lane_intersection_eval_tables_fn(eval_payload["map_eval"])
    else:
        print_eval_table_fn(eval_payload)


def eval_console_payload(eval_path, eval_payload, args):
    payload = {"eval_json": str(eval_path), "centerline_eval_json": str(eval_path)}
    if args.map_task == "lane_intersection":
        payload.update({
            "centerline_eval": eval_payload["centerline_eval"],
            "intersection_eval": eval_payload["intersection_eval"],
            "lane_intersection_eval": eval_payload["lane_intersection_eval"],
        })
    else:
        payload["centerline_eval"] = eval_payload
    return payload


def main():
    import os
    import torch
    local_rank = int(os.environ.get("LOCAL_RANK",-1))
    rank = int(os.environ.get("RANK", local_rank if local_rank >= 0 else 0))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    silence_non_primary_rank_output()
    if local_rank >= 0:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device_str = f"cuda:{local_rank}"
        else:
            if hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.set_device(local_rank)
            device_str = f"npu:{local_rank}"
    else:
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.set_device(0)
            device_str = "npu:0"
        else:
            device_str = "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--image", default="")
    parser.add_argument("--test-json", default="")
    parser.add_argument("--image-folder", default="")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of records to infer from --test-json. 0 or negative means all records.",
    )
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--prompt-mode", choices=["default", "dataset"], default="default")
    parser.add_argument("--conv-template", default="")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--per-device-infer-batch-size",
        type=int,
        default=int(os.environ.get("PER_DEVICE_INFER_BATCH_SIZE", "1")),
        help="Number of samples passed to one generate() call on each device.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--output-jsonl",
        default="",
        help="Stream one result object per line; rank files are merged on rank 0.",
    )
    parser.add_argument(
        "--resume-output-jsonl",
        action="store_true",
        help="Resume a JSONL stream by skipping indices already written by this rank.",
    )
    parser.add_argument(
        "--stream-fsync-every",
        type=int,
        default=100,
        help="fsync interval for streamed output rows; 0 disables periodic fsync.",
    )
    parser.add_argument(
        "--distributed-merge-timeout",
        type=int,
        default=int(os.environ.get("INFER_DISTRIBUTED_MERGE_TIMEOUT", "1800")),
        help="Seconds rank 0 waits for filesystem rank summaries; no HCCL collective is used.",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--sample-json-dir", default="", help="Directory for per-sample JSON files. Defaults to output-dir.")
    parser.add_argument(
        "--no-sample-json",
        action="store_true",
        help="Do not write one auxiliary JSON file per sample.",
    )
    parser.add_argument("--print-full-output", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print one progress line every N newly processed records.",
    )
    parser.add_argument(
        "--quiet-samples",
        action="store_true",
        help="Print compact progress lines instead of prediction previews.",
    )
    parser.add_argument("--vision_tower", default="")
    parser.add_argument("--vision_tower_checkpoint", default="")
    parser.add_argument("--mm_vision_tower_type", default="")
    parser.add_argument("--multi_vision_towers", default="")
    parser.add_argument("--multi_vision_tower_types", default="")
    parser.add_argument("--multi_vision_input_image_sizes", default="")
    parser.add_argument("--multi_vision_primary_index", type=int, default=None)
    parser.add_argument("--multi_vision_hidden_size", type=int, default=None)
    parser.add_argument("--multi_vision_target_grid", type=int, default=None)
    parser.add_argument("--multi_vision_fusion", default="")
    parser.add_argument("--multi_vision_router_temperature", type=float, default=None)
    parser.add_argument("--multi_vision_router_hidden_ratio", type=float, default=None)
    parser.add_argument("--multi_vision_router_use_diff", type=lambda x: str(x).lower() in ("1", "true", "yes", "on"), default=None)
    parser.add_argument("--multi_vision_dropout", type=float, default=None)
    parser.add_argument("--vision_layer_fusion_indexes", type=int, nargs="*", default=None)
    parser.add_argument("--vision_layer_fusion_type", default="")
    parser.add_argument("--input_image_size", type=int, default=None)
    parser.add_argument("--disable_deepstack", action="store_true", default=None)
    parser.add_argument("--deepstack_visual_indexes", type=int, nargs="*", default=None)
    parser.add_argument("--eval-centerline", action="store_true")
    parser.add_argument("--eval-output-json", default="", help="Path for aggregate centerline metrics JSON.")
    parser.add_argument("--eval-meter-per-pixel", type=float, default=0.2)
    parser.add_argument("--eval-buffer-size", type=float, default=1.0)
    parser.add_argument("--eval-match-threshold", type=float, default=0.33)
    parser.add_argument("--map-task", choices=["lane", "lane_intersection"], default="lane_intersection")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--coord-mode", choices=["auto", COORD_MODE_PIXEL, COORD_MODE_NORM1000], default="auto")
    parser.add_argument("--coord-range", type=int, default=DEFAULT_COORD_RANGE)
    args = parser.parse_args()
    if args.per_device_infer_batch_size < 1:
        raise ValueError("--per-device-infer-batch-size must be >= 1")
    if args.distributed_merge_timeout < 1:
        raise ValueError("--distributed-merge-timeout must be >= 1")
    if args.stream_fsync_every < 0:
        raise ValueError("--stream-fsync-every must be non-negative")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if args.output_jsonl and args.eval_centerline:
        raise ValueError("--eval-centerline is not supported with streaming --output-jsonl")
    args.device = device_str
    if distributed and args.output_json:
        rank_json_path(Path(args.output_json), rank).unlink(missing_ok=True)
        if rank == 0:
            Path(args.output_json).unlink(missing_ok=True)

    evaluate_one_sample = evaluate_records = print_eval_table = None
    evaluate_lane_intersection_records = print_lane_intersection_eval_tables = None
    if args.eval_centerline:
        from infer_index.line_eval import (
            evaluate_one_sample,
            evaluate_records,
            evaluate_lane_intersection_records,
            print_eval_table,
            print_lane_intersection_eval_tables,
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    manifest = read_manifest(checkpoint_dir)

    conv_template = args.conv_template or manifest.get("version") or ""
    if not conv_template or conv_template not in conversation_lib.conv_templates:
        qwen_family = qwen_family_from_text(
            manifest.get("model_type"),
            json.dumps(manifest, ensure_ascii=False),
        )
        if is_qwen3_or_newer_family(qwen_family):
            conv_template = "conv_qwen_3_Dinov2_huawei"
        else:
            conv_template = "conv_qwen_2_Dinov2_huawei"

   
    config_overrides = _config_overrides_from_args(args)
    print(
        "[infer] config overrides: "
        f"vision_tower={config_overrides.get('mm_vision_tower')}, "
        f"vision_tower_checkpoint={config_overrides.get('vision_tower_checkpoint')}, "
        f"input_image_size={config_overrides.get('input_image_size')}, "
        f"disable_deepstack={config_overrides.get('disable_deepstack')}, "
        f"deepstack_visual_indexes={config_overrides.get('deepstack_visual_indexes')}"
    )
    tokenizer, model, image_processor = load_model_components(
        checkpoint_dir, manifest, args.device, config_overrides=config_overrides)
    coordinate_token_mode, coordinate_token_max = resolve_coordinate_token_settings(
        tokenizer,
        model.config,
    )
    print(
        "[infer] coordinate tokens: "
        f"mode={coordinate_token_mode}, max={coordinate_token_max}"
    )
    if args.per_device_infer_batch_size > 1:
        tokenizer.padding_side = "left"
        model.config.tokenizer_padding_side = "left"
    print(f"[infer] per-device batch size: {args.per_device_infer_batch_size}")
    if args.test_json:
        if not args.image_folder:
            raise ValueError("--image-folder is required when using --test-json")
        start = max(0, args.sample_offset)
        end = start + args.num_samples if args.num_samples > 0 else None

        def selected_records():
            selected_position = 0
            for index, record in iter_json_records(Path(args.test_json)):
                if index < start:
                    continue
                if end is not None and index >= end:
                    break
                if selected_position % world_size == rank:
                    yield index, record
                selected_position += 1

        indexed_records = selected_records()
    else:
        if not args.image:
            raise ValueError("Provide either --image or --test-json")
        indexed_records = iter([(0, {"id": "single_image", "image": args.image, "conversations": [{"value": args.prompt}]})])

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    sample_json_dir = None
    if not args.no_sample_json:
        sample_json_dir = Path(args.sample_json_dir) if args.sample_json_dir else output_dir
    if sample_json_dir is not None:
        sample_json_dir.mkdir(parents=True, exist_ok=True)
    rank_suffix = ""
    if distributed:
        rank_suffix = f"rank{rank}_"

    stream_output_path = Path(args.output_jsonl) if args.output_jsonl else None
    stream_rank_handle = None
    stream_completed = set()
    if stream_output_path is not None:
        stream_output_path.parent.mkdir(parents=True, exist_ok=True)
        stream_rank_path = rank_json_path(stream_output_path, rank)
        if args.resume_output_jsonl:
            stream_completed = load_completed_jsonl_indices(stream_rank_path)
            stream_rank_handle = stream_rank_path.open("a", encoding="utf-8")
            print(
                f"[infer] resume stream rank={rank} completed={len(stream_completed)} "
                f"path={stream_rank_path}"
            )
        else:
            stream_rank_handle = stream_rank_path.open("w", encoding="utf-8")
            if rank == 0 and stream_output_path.exists():
                stream_output_path.unlink()
        if distributed:
            torch.distributed.barrier()

    generation_config = getattr(model, "generation_config", None)
    pad_token_id = getattr(generation_config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    eos_token_id = getattr(generation_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer/model does not define pad_token_id or eos_token_id.")

    vision_tower = model.get_vision_tower()
    dtype = vision_tower.dtype if vision_tower is not None else next(model.parameters()).dtype
    image_device = vision_tower.device if vision_tower is not None else model.device

    pending_records = (
        (idx, record)
        for idx, record in indexed_records
        if stream_rank_handle is None or idx not in stream_completed
    )
    results = []
    processed_count = 0
    stream_write_count = 0
    for batch_entries in iter_batches(pending_records, args.per_device_infer_batch_size):
        contexts = []
        images = []
        token_sequences = []
        for idx, record in batch_entries:
            image_relpath = resolve_record_image(record)
            image_path = Path(image_relpath)
            if args.test_json:
                image_path = Path(args.image_folder) / image_path
            image_path = image_path.resolve()
            with Image.open(image_path) as image_handle:
                image = image_handle.convert("RGB")
            images.append(image)

            if args.prompt_mode == "dataset" and record.get("conversations"):
                prompt_text = record["conversations"][0]["value"]
            else:
                prompt_text = args.prompt
            if coordinate_token_mode == COORDINATE_TOKEN_MODE_ANGLE:
                prompt_text = append_coordinate_token_instruction(
                    prompt_text,
                    max_coordinate=coordinate_token_max,
                )
            prompt = build_prompt(prompt_text, conv_template)
            token_sequence = tokenizer_image_token(
                prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            token_sequences.append(token_sequence)
            contexts.append((idx, record, image_path, image_relpath, image.size, prompt))

        images_tensor = process_images(images, image_processor, model.config)
        if isinstance(images_tensor, list):
            images_tensor = [img.to(dtype=dtype, device=image_device) for img in images_tensor]
        else:
            images_tensor = images_tensor.to(dtype=dtype, device=image_device)
        input_ids, attention_mask = left_pad_token_sequences(
            token_sequences,
            pad_token_id,
            model.device,
        )

        generate_kwargs = {
            "attention_mask": attention_mask,
            "images": images_tensor,
            "image_sizes": [context[4] for context in contexts],
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
            "do_sample": args.temperature > 0,
            "num_beams": 1,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
        }
        if args.temperature > 0:
            generate_kwargs["temperature"] = args.temperature

        with torch.inference_mode():
            output_ids = model.generate(input_ids, **generate_kwargs)
        if output_ids.ndim == 1:
            output_ids = output_ids.unsqueeze(0)
        if int(output_ids.shape[0]) != len(contexts):
            raise RuntimeError(
                "generate() returned an unexpected batch dimension: "
                f"expected={len(contexts)}, actual={int(output_ids.shape[0])}"
            )

        for batch_row, (idx, record, image_path, image_relpath, _, prompt) in enumerate(contexts):
            decoded_ids, decoded_mode = completion_token_ids(
                output_ids,
                input_ids,
                attention_mask=attention_mask,
                row_index=batch_row,
            )
            raw_prediction = tokenizer.batch_decode(
                decoded_ids.unsqueeze(0),
                skip_special_tokens=False,
            )[0].strip()
            prediction = normalize_prediction_text(
                raw_prediction,
                max_coordinate=coordinate_token_max,
            )
            prediction_json = extract_json_payload(prediction)
            coord_cfg = resolve_coord_config(record, args)

            parse_ok = False
            parsed_items = []
            parsed_items_pixel = []
            parse_error = ""
            try:
                parsed_items = parse_centerline_json(
                    prediction_json,
                    map_task=args.map_task,
                    patch_size=coord_cfg["patch_size"],
                    coord_mode=coord_cfg["coord_mode"],
                    coord_range=coord_cfg["coord_range"],
                )
                parsed_items_pixel = convert_items(
                    parsed_items,
                    coord_cfg["coord_mode"],
                    COORD_MODE_PIXEL,
                    coord_cfg["patch_width"],
                    coord_cfg["patch_height"],
                    coord_range=coord_cfg["coord_range"],
                    clamp=True,
                )
                prediction_json = payload_to_text({"lines": parsed_items})
                parse_ok = True
            except Exception as exc:
                parse_error = str(exc)
            prediction_json_pixel = payload_to_text({"lines": parsed_items_pixel}) if parse_ok else ""
            origin_record = {
                "meta": record.get("meta", {}),
                "patch_size": coord_cfg["patch_size"],
                "patch_width": coord_cfg["patch_width"],
                "patch_height": coord_cfg["patch_height"],
            }
            x0, y0 = record_origin(origin_record)
            meta = record.get("meta", {})
            row = int(meta.get("row", meta.get("patch_row", y0 // max(coord_cfg["patch_height"], 1))))
            col = int(meta.get("col", meta.get("patch_col", x0 // max(coord_cfg["patch_width"], 1))))
            tile_id = meta.get("tile_id", record.get("tile_id", "tile"))
            lines_global = offset_lines(parsed_items_pixel, x0, y0) if parse_ok else []

            result = {
                "idx": idx,
                "checkpoint_dir": str(checkpoint_dir),
                "image": str(image_path),
                "image_relpath": image_relpath,
                "record_id": record.get("id", f"sample_{idx}"),
                "tile_id": tile_id,
                "row": row,
                "col": col,
                "x0": x0,
                "y0": y0,
                "meta": record.get("meta", {}),
                "coord_mode": coord_cfg["coord_mode"],
                "coord_range": coord_cfg["coord_range"],
                "patch_size": coord_cfg["patch_size"],
                "patch_width": coord_cfg["patch_width"],
                "patch_height": coord_cfg["patch_height"],
                "prompt": prompt,
                "conv_template": conv_template,
                "raw_prediction": raw_prediction,
                "prediction": prediction,
                "prediction_json": prediction_json,
                "prediction_json_pixel": prediction_json_pixel,
                "parse_ok": parse_ok,
                "num_items": len(parsed_items) if parse_ok else 0,
                "parse_error": parse_error,
                "lines_local": parsed_items_pixel,
                "lines_local_model": parsed_items,
                "lines_global": lines_global,
                "input_token_len": int(attention_mask[batch_row].sum().item()),
                "output_token_len": int(output_ids[batch_row].numel()),
                "decoded_token_len": int(decoded_ids.numel()),
                "decoded_mode": decoded_mode,
                "per_device_infer_batch_size": args.per_device_infer_batch_size,
            }
            if stream_rank_handle is None:
                # Preserve the legacy per-sample payload for small JSON-array runs.
                result["manifest"] = manifest
            else:
                # Store the manifest once in the launcher output tree instead of
                # repeating it in every large streamed row.
                result["manifest_source"] = manifest.get("manifest_source", "")
            if len(record.get("conversations", [])) > 1:
                result["ground_truth"] = record["conversations"][1]["value"]
                try:
                    result["ground_truth_pixel"] = convert_payload_text(
                        result["ground_truth"],
                        coord_cfg["coord_mode"],
                        COORD_MODE_PIXEL,
                        coord_cfg["patch_width"],
                        coord_cfg["patch_height"],
                        coord_range=coord_cfg["coord_range"],
                        clamp=True,
                    )
                except Exception:
                    result["ground_truth_pixel"] = result["ground_truth"]
                if args.eval_centerline:
                    result["centerline_eval"] = vars(evaluate_one_sample(
                        result["ground_truth_pixel"],
                        prediction_json_pixel or prediction_json,
                        parse_ok=parse_ok,
                        meter_per_pixel=args.eval_meter_per_pixel,
                        buffer_size=args.eval_buffer_size,
                        match_threshold=args.eval_match_threshold,
                    ))
            if stream_rank_handle is None:
                results.append(result)

            if stream_rank_handle is not None:
                stream_rank_handle.write(
                    json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                stream_rank_handle.flush()
                stream_write_count += 1
                if args.stream_fsync_every and stream_write_count % args.stream_fsync_every == 0:
                    os.fsync(stream_rank_handle.fileno())

            if sample_json_dir is not None:
                sample_path = sample_json_dir / f"{rank_suffix}{idx:03d}_{sanitize_filename(str(result['record_id']))}.json"
                sample_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

            processed_count += 1
            if processed_count == 1 or processed_count % args.progress_every == 0:
                if args.quiet_samples:
                    print(
                        json.dumps(
                            {
                                "progress": processed_count,
                                "last_idx": idx,
                                "last_record_id": result["record_id"],
                                "parse_ok": parse_ok,
                                "num_items": result["num_items"],
                            },
                            ensure_ascii=False,
                        )
                    )
                else:
                    print(
                        json.dumps(
                            {
                                "idx": idx,
                                "record_id": result["record_id"],
                                "image": result["image"],
                                "parse_ok": parse_ok,
                                "num_items": result["num_items"],
                                "parse_error": parse_error,
                                "prediction_preview": prediction_preview(prediction),
                                "decoded_mode": result["decoded_mode"],
                                "input_token_len": result["input_token_len"],
                                "output_token_len": result["output_token_len"],
                                "decoded_token_len": result["decoded_token_len"],
                                "per_device_infer_batch_size": args.per_device_infer_batch_size,
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

    if stream_rank_handle is not None:
        stream_rank_handle.flush()
        os.fsync(stream_rank_handle.fileno())
        stream_rank_handle.close()
        print(f"[infer] rank processed new rows={processed_count}")
        if distributed:
            torch.distributed.barrier()
            if rank == 0:
                merged_count = merge_rank_jsonl(
                    stream_output_path,
                    world_size,
                )
                print(f"[infer] merged streamed JSONL rows={merged_count}: {stream_output_path}")
            torch.distributed.barrier()
        else:
            merged_count = merge_rank_jsonl(stream_output_path, 1)
            print(f"[infer] merged streamed JSONL rows={merged_count}: {stream_output_path}")

    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        if distributed:
            rank_output_json_path = rank_json_path(output_json_path, rank)
            write_json_atomic(rank_output_json_path, results)
            if rank == 0:
                merged_results = []
                rank_paths = wait_for_rank_jsons(
                    output_json_path,
                    world_size,
                    args.distributed_merge_timeout,
                )
                for rank_path in rank_paths:
                    rank_payload = json.loads(rank_path.read_text(encoding="utf-8"))
                    if isinstance(rank_payload, list):
                        merged_results.extend(item for item in rank_payload if isinstance(item, dict))
                merged_results.sort(key=lambda item: (item.get("idx", 0), str(item.get("record_id", ""))))
                write_json_atomic(output_json_path, merged_results)
                if args.eval_centerline:
                    eval_summary = build_eval_payload(
                        merged_results,
                        args,
                        evaluate_records,
                        evaluate_lane_intersection_records,
                    )
                    eval_path = Path(args.eval_output_json) if args.eval_output_json else output_json_path.with_name("eval.json")
                    eval_path.parent.mkdir(parents=True, exist_ok=True)
                    write_json_atomic(eval_path, eval_summary)
                    print_eval_payload(eval_summary, args, print_eval_table, print_lane_intersection_eval_tables)
                    print(json.dumps(eval_console_payload(eval_path, eval_summary, args), ensure_ascii=False))
        else:
            write_json_atomic(output_json_path, results)
            if args.eval_centerline:
                eval_summary = build_eval_payload(
                    results,
                    args,
                    evaluate_records,
                    evaluate_lane_intersection_records,
                )
                eval_path = Path(args.eval_output_json) if args.eval_output_json else output_json_path.with_name("eval.json")
                eval_path.parent.mkdir(parents=True, exist_ok=True)
                write_json_atomic(eval_path, eval_summary)
                print_eval_payload(eval_summary, args, print_eval_table, print_lane_intersection_eval_tables)
                print(json.dumps(eval_console_payload(eval_path, eval_summary, args), ensure_ascii=False))

if __name__ == "__main__":
    main()
