"""Discrete coordinate tokens for map-geometry SFT and inference."""

from __future__ import annotations

import json
import re
from numbers import Integral, Real
from typing import Any, Iterable


COORDINATE_TOKEN_MODE_NONE = "none"
COORDINATE_TOKEN_MODE_ANGLE = "angle"
_COORDINATE_TOKEN_RE = re.compile(r"<(\d+)>")
_QUOTED_COORDINATE_TOKEN_RE = re.compile(r'"<(\d+)>"')


def normalize_coordinate_token_mode(mode: str | None) -> str:
    normalized = str(mode or COORDINATE_TOKEN_MODE_NONE).strip().lower()
    aliases = {
        "": COORDINATE_TOKEN_MODE_NONE,
        "off": COORDINATE_TOKEN_MODE_NONE,
        "false": COORDINATE_TOKEN_MODE_NONE,
        "disabled": COORDINATE_TOKEN_MODE_NONE,
        "discrete": COORDINATE_TOKEN_MODE_ANGLE,
        "special": COORDINATE_TOKEN_MODE_ANGLE,
        "angle_bracket": COORDINATE_TOKEN_MODE_ANGLE,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {COORDINATE_TOKEN_MODE_NONE, COORDINATE_TOKEN_MODE_ANGLE}:
        raise ValueError(
            f"Unsupported coordinate_token_mode={mode!r}; expected none or angle."
        )
    return normalized


def coordinate_token(value: int) -> str:
    return f"<{int(value)}>"


def build_coordinate_vocabulary(max_coordinate: int = 1000) -> list[str]:
    if max_coordinate < 0:
        raise ValueError("max_coordinate must be non-negative.")
    return [coordinate_token(value) for value in range(max_coordinate + 1)]


def coordinate_token_instruction(max_coordinate: int = 1000) -> str:
    return (
        "In the assistant answer, write every value inside a points array as one "
        f"unquoted discrete coordinate token <n>, where n is 0-{max_coordinate}. "
        "For example, write [[<956>,<42>],[<1000>,<0>]]. Use these tokens only "
        "for coordinates inside points arrays."
    )


def append_coordinate_token_instruction(text: str, max_coordinate: int = 1000) -> str:
    if "unquoted discrete coordinate token <n>" in text:
        return text
    return text.rstrip() + "\n\n" + coordinate_token_instruction(max_coordinate)


def _encode_point_values(value: Any, max_coordinate: int, in_points: bool = False) -> Any:
    if isinstance(value, dict):
        return {
            key: _encode_point_values(
                child,
                max_coordinate,
                in_points=in_points or str(key).lower() == "points",
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _encode_point_values(child, max_coordinate, in_points=in_points)
            for child in value
        ]
    if not in_points:
        return value
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Point coordinate must be numeric, got {value!r}.")
    if isinstance(value, Integral):
        integer_value = int(value)
    else:
        integer_value = int(value)
        if float(value) != float(integer_value):
            raise ValueError(f"Point coordinate must be an integer, got {value!r}.")
    if not 0 <= integer_value <= max_coordinate:
        raise ValueError(
            f"Point coordinate {integer_value} is outside 0-{max_coordinate}."
        )
    return coordinate_token(integer_value)


def encode_map_json_coordinates(text: str, max_coordinate: int = 1000) -> str:
    """Replace numeric values under every ``points`` key with bare ``<n>`` tokens."""
    payload = json.loads(text)
    encoded_payload = _encode_point_values(payload, max_coordinate)
    encoded = json.dumps(encoded_payload, ensure_ascii=False, separators=(",", ":"))
    return _QUOTED_COORDINATE_TOKEN_RE.sub(r"<\1>", encoded)


def encode_coordinate_conversations(
    conversations: Iterable[dict[str, Any]],
    *,
    max_coordinate: int = 1000,
) -> list[dict[str, Any]]:
    """Add the format instruction and encode assistant map coordinates."""
    encoded = [dict(message) for message in conversations]
    instruction_added = False
    for message in encoded:
        role = str(message.get("from", message.get("role", ""))).strip().lower()
        value_key = "value" if "value" in message else "content"
        value = message.get(value_key)
        if not isinstance(value, str):
            continue
        if role in {"human", "user"} and not instruction_added:
            message[value_key] = append_coordinate_token_instruction(value, max_coordinate)
            instruction_added = True
        elif role in {"gpt", "assistant"}:
            message[value_key] = encode_map_json_coordinates(value, max_coordinate)
    return encoded


def decode_coordinate_tokens(text: str, max_coordinate: int = 1000) -> str:
    """Restore ``<n>`` generation tokens to JSON numeric literals."""

    def replace(match: re.Match[str]) -> str:
        value = int(match.group(1))
        return str(value) if 0 <= value <= max_coordinate else match.group(0)

    # Accept both the documented bare form and a model's occasionally quoted form.
    decoded = _QUOTED_COORDINATE_TOKEN_RE.sub(replace, text)
    return _COORDINATE_TOKEN_RE.sub(replace, decoded)


def tokenizer_has_coordinate_vocabulary(tokenizer: Any, max_coordinate: int = 1000) -> bool:
    probes = (coordinate_token(0), coordinate_token(max_coordinate))
    added_vocab = getattr(tokenizer, "get_added_vocab", lambda: {})()
    return all(token in added_vocab for token in probes)


def register_coordinate_vocabulary(
    tokenizer: Any,
    model: Any,
    *,
    max_coordinate: int = 1000,
) -> dict[str, Any]:
    """Register ``<0>`` through ``<max>`` and resize model embeddings once."""
    vocabulary = build_coordinate_vocabulary(max_coordinate)
    baseline_lengths = [
        len(tokenizer(str(value), add_special_tokens=False).input_ids)
        for value in range(max_coordinate + 1)
    ]
    original_tokenizer_size = len(tokenizer)
    # These must stay ordinary added tokens. Marking them as tokenizer control
    # tokens would make generic ``skip_special_tokens=True`` decode paths erase
    # generated coordinates before schema parsing or reward calculation.
    added_tokens = int(tokenizer.add_tokens(vocabulary, special_tokens=False))
    if added_tokens:
        model.resize_token_embeddings(len(tokenizer))

    invalid = []
    for token in vocabulary:
        token_ids = tokenizer(token, add_special_tokens=False).input_ids
        if len(token_ids) != 1:
            invalid.append({"token": token, "token_ids": token_ids})
            if len(invalid) >= 8:
                break
    if invalid:
        raise ValueError(f"Coordinate tokens are not atomic: {invalid}")

    return {
        "mode": COORDINATE_TOKEN_MODE_ANGLE,
        "max_coordinate": int(max_coordinate),
        "vocabulary_size": len(vocabulary),
        "original_tokenizer_size": original_tokenizer_size,
        "final_tokenizer_size": len(tokenizer),
        "added_tokens": added_tokens,
        "baseline_mean_tokens_per_coordinate": (
            sum(baseline_lengths) / max(len(baseline_lengths), 1)
        ),
        "baseline_max_tokens_per_coordinate": max(baseline_lengths, default=0),
        "baseline_single_token_coordinates": sum(length == 1 for length in baseline_lengths),
        "discrete_tokens_per_coordinate": 1,
    }
