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
