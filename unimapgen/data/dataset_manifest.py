import copy
import json
import os
from typing import Dict


def _resolve_path(base_dir: str, value):
    if not isinstance(value, str) or not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(base_dir, value))


def load_dataset_manifest(manifest_path: str) -> Dict:
    path = os.path.abspath(str(manifest_path))
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Dataset manifest must be a JSON object: {path}")
    obj["_manifest_path"] = path
    return obj


def resolve_view_data_config(data_cfg: Dict, dataset_name: str) -> Dict:
    resolved = copy.deepcopy(dict(data_cfg))
    manifest_path = str(resolved.get("dataset_manifest_path", "") or "").strip()
    if not manifest_path:
        raise ValueError("dataset_manifest_path is required to resolve a manifest dataset view.")
    manifest = load_dataset_manifest(manifest_path)
    manifest_dir = os.path.dirname(str(manifest["_manifest_path"]))

    views = manifest.get("training_views", {})
    if dataset_name not in views or not isinstance(views[dataset_name], dict):
        raise KeyError(f"Dataset view not found in manifest: {dataset_name}")
    view = dict(views[dataset_name])
    snapshot_name = str(view.get("snapshot", "") or "").strip()
    snapshots = manifest.get("snapshots", {})
    if snapshot_name not in snapshots or not isinstance(snapshots[snapshot_name], dict):
        raise KeyError(f"Snapshot not found in manifest: {snapshot_name}")
    snapshot = dict(snapshots[snapshot_name])

    snapshot_fields = {
        "source": snapshot.get("source", "opensatmap"),
        "opensatmap_root": _resolve_path(manifest_dir, snapshot.get("root")),
        "opensatmap_ann_json": _resolve_path(manifest_dir, snapshot.get("ann_json")),
        "opensatmap_split_dir": _resolve_path(manifest_dir, snapshot.get("split_dir")),
        "splits_meta_path": _resolve_path(manifest_dir, snapshot.get("splits_meta_path")),
        "patch_geometry_json": _resolve_path(manifest_dir, snapshot.get("patch_geometry_json")),
    }
    view_fields = {
        k: v
        for k, v in view.items()
        if k not in {"snapshot", "intended_configs"}
    }

    resolved.update(snapshot_fields)
    resolved.update(view_fields)
    resolved["dataset_name"] = str(dataset_name)
    resolved["dataset_snapshot_name"] = snapshot_name
    resolved["dataset_manifest_path"] = str(manifest["_manifest_path"])
    return resolved


def load_mix_profile(data_cfg: Dict, mix_profile_name: str) -> Dict:
    manifest_path = str(data_cfg.get("dataset_manifest_path", "") or "").strip()
    if not manifest_path:
        raise ValueError("dataset_manifest_path is required to resolve a mix profile.")
    manifest = load_dataset_manifest(manifest_path)
    profiles = manifest.get("mix_profiles", {})
    if mix_profile_name not in profiles or not isinstance(profiles[mix_profile_name], dict):
        raise KeyError(f"Mix profile not found in manifest: {mix_profile_name}")
    out = dict(profiles[mix_profile_name])
    out["_manifest_path"] = str(manifest["_manifest_path"])
    out["name"] = str(mix_profile_name)
    return out


def resolve_data_config(data_cfg: Dict) -> Dict:
    resolved = copy.deepcopy(dict(data_cfg))
    dataset_name = str(resolved.get("dataset_name", "") or "").strip()
    if not dataset_name:
        return resolved
    return resolve_view_data_config(resolved, dataset_name=dataset_name)
