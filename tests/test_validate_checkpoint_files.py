import json
import sys
from types import ModuleType

import pytest

from scripts.tools.validate_checkpoint_files import validate_checkpoint


class _FakeSafeOpen:
    def __init__(self, path, **_kwargs):
        self.path = path

    def __enter__(self):
        if open(self.path, "rb").read() == b"corrupt":
            raise RuntimeError("incomplete metadata, file not fully covered")
        return self

    def __exit__(self, *_args):
        return False

    def keys(self):
        return ["weight"]


@pytest.fixture(autouse=True)
def fake_safetensors(monkeypatch):
    module = ModuleType("safetensors")
    module.safe_open = _FakeSafeOpen
    monkeypatch.setitem(sys.modules, "safetensors", module)


def test_validate_checkpoint_accepts_complete_indexed_safetensors(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"valid")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": shard.name}}),
        encoding="utf-8",
    )

    summary = validate_checkpoint(tmp_path)

    assert summary["num_weight_files"] == 1
    assert summary["weight_files"] == [str(shard.resolve())]


def test_validate_checkpoint_rejects_corrupt_safetensors(tmp_path):
    model_path = tmp_path / "model.safetensors"
    model_path.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="Unreadable safetensors file"):
        validate_checkpoint(tmp_path)


def test_validate_checkpoint_rejects_missing_index_shard(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing files"):
        validate_checkpoint(tmp_path)


def test_validate_checkpoint_accepts_complete_lora_layout(tmp_path):
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(b"valid")
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/cache/base-model"}),
        encoding="utf-8",
    )
    (tmp_path / "non_lora_trainables.bin").write_bytes(b"vision-and-projector")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    summary = validate_checkpoint(tmp_path, expected_kind="lora")

    assert summary["checkpoint_kind"] == "lora"
    assert summary["expected_kind"] == "lora"
    assert summary["weight_files"] == [str(adapter.resolve())]


@pytest.mark.parametrize(
    "missing_name",
    ("adapter_config.json", "adapter_model.safetensors", "non_lora_trainables.bin", "config.json"),
)
def test_validate_checkpoint_rejects_incomplete_lora_layout(tmp_path, missing_name):
    files = {
        "adapter_config.json": json.dumps({"base_model_name_or_path": "/cache/base-model"}).encode(),
        "adapter_model.safetensors": b"valid",
        "non_lora_trainables.bin": b"vision-and-projector",
        "config.json": b"{}",
    }
    for name, payload in files.items():
        if name != missing_name:
            (tmp_path / name).write_bytes(payload)

    with pytest.raises(FileNotFoundError, match="LoRA checkpoint is missing required files"):
        validate_checkpoint(tmp_path, expected_kind="lora")


def test_validate_checkpoint_rejects_lora_without_base_model(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"valid")
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "non_lora_trainables.bin").write_bytes(b"vision-and-projector")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="base_model_name_or_path"):
        validate_checkpoint(tmp_path, expected_kind="lora")
