import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import torch


def file_stat_payload(path: Optional[str]):
    if not path or not os.path.isfile(path):
        return None
    st = os.stat(path)
    return {
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


class PretokenizedSplitCache:
    def __init__(self, cache_dir: str, split: str, meta: Dict) -> None:
        self.cache_dir = str(cache_dir)
        self.split = str(split)
        self.meta = dict(meta)
        self.shard_size = int(meta.get("shard_size", 4096))
        self.num_samples = int(meta.get("num_samples", 0))
        self.shards = list(meta.get("shards", []))
        self._loaded_shard_idx = None
        self._loaded_shard_samples = None

    @property
    def is_valid(self) -> bool:
        if self.num_samples < 0 or self.shard_size <= 0:
            return False
        if self.num_samples == 0:
            return False
        if not self.shards:
            return False
        for name in self.shards:
            if not os.path.isfile(os.path.join(self.cache_dir, str(name))):
                return False
        return True

    def get_sample(self, index: int) -> Dict:
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)
        shard_idx = int(index) // self.shard_size
        local_idx = int(index) % self.shard_size
        if self._loaded_shard_idx != shard_idx or self._loaded_shard_samples is None:
            shard_name = str(self.shards[shard_idx])
            shard_path = os.path.join(self.cache_dir, shard_name)
            obj = torch.load(shard_path, map_location="cpu", weights_only=False)
            samples = obj.get("samples", obj) if isinstance(obj, dict) else obj
            if not isinstance(samples, list):
                raise ValueError(f"Invalid cache shard: {shard_path}")
            self._loaded_shard_idx = shard_idx
            self._loaded_shard_samples = samples
        return dict(self._loaded_shard_samples[local_idx])


def build_cache_dir(base_dir: str, split: str, signature: str) -> str:
    return os.path.join(str(base_dir), f"{str(split)}_{str(signature)}")


def load_split_cache(cache_dir: str, split: str, signature: str) -> Optional[PretokenizedSplitCache]:
    meta_path = os.path.join(str(cache_dir), "meta.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    if str(meta.get("split", "")) != str(split):
        return None
    if str(meta.get("signature", "")) != str(signature):
        return None
    cache = PretokenizedSplitCache(cache_dir=cache_dir, split=split, meta=meta)
    return cache if cache.is_valid else None


def save_split_cache(
    cache_dir: str,
    split: str,
    signature: str,
    samples: List[Dict],
    shard_size: int,
    extra_meta: Optional[Dict] = None,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    shard_size = max(1, int(shard_size))
    shard_names: List[str] = []
    for shard_idx, start in enumerate(range(0, len(samples), shard_size)):
        shard_name = f"{str(split)}-{int(shard_idx):05d}.pt"
        shard_path = os.path.join(cache_dir, shard_name)
        tmp_path = shard_path + ".tmp"
        torch.save({"samples": samples[start : start + shard_size]}, tmp_path)
        os.replace(tmp_path, shard_path)
        shard_names.append(shard_name)

    meta = {
        "schema_version": "unimapgen.pretokenized_cache.v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": str(split),
        "signature": str(signature),
        "num_samples": int(len(samples)),
        "shard_size": int(shard_size),
        "num_shards": int(len(shard_names)),
        "shards": shard_names,
    }
    if isinstance(extra_meta, dict):
        meta.update(extra_meta)

    meta_path = os.path.join(cache_dir, "meta.json")
    tmp_meta_path = meta_path + ".tmp"
    with open(tmp_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp_meta_path, meta_path)
