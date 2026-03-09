import argparse
import glob
import json
import os
import pickle
import sys
from typing import Dict, List, Tuple

from unimapgen.utils import load_yaml


def _norm_nuscenes_rel_path(rel_path: str) -> str:
    if rel_path.startswith("./data/nuscenes/"):
        return rel_path[len("./data/nuscenes/") :]
    return rel_path


def _load_infos(pkl_path: str) -> Tuple[List[Dict], str]:
    if not os.path.exists(pkl_path):
        return [], f"missing pkl: {pkl_path}"
    try:
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
    except Exception as e:  # pragma: no cover
        return [], f"failed to read pkl: {pkl_path} ({e})"
    if not isinstance(raw, dict) or "infos" not in raw:
        return [], f"invalid pkl format (expect dict with 'infos'): {pkl_path}"
    infos = raw.get("infos", [])
    if not isinstance(infos, list):
        return [], f"invalid pkl field type (infos should be list): {pkl_path}"
    return infos, ""


def _load_ann_json(path: str) -> Tuple[Dict, str]:
    if not os.path.exists(path):
        return {}, f"missing annotation json: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            x = json.load(f)
    except Exception as e:  # pragma: no cover
        return {}, f"failed to read json: {path} ({e})"
    if not isinstance(x, dict):
        return {}, f"invalid annotation json format (expect dict): {path}"
    return x, ""


def _check_nuscenes_maptr_split(cfg: Dict, split: str, max_scan: int) -> Dict:
    dcfg = cfg["data"]
    scfg = cfg["serialization"]
    pkl_path = os.path.join(dcfg["nuscenes_map_pkl_dir"], f"nuscenes_map_infos_temporal_{split}.pkl")
    infos, err = _load_infos(pkl_path)
    out = {
        "kind": "nuscenes_maptr",
        "split": split,
        "path": pkl_path,
        "total_items": len(infos),
        "scanned_items": 0,
        "sat_exists": 0,
        "sat_missing": 0,
        "ann_exists": 0,
        "ann_nonempty": 0,
        "pv_exists": 0,
        "pv_missing": 0,
        "errors": [],
        "missing_sat_examples": [],
        "missing_pv_examples": [],
    }
    if err:
        out["errors"].append(err)
        return out
    if len(infos) == 0:
        out["errors"].append(f"empty infos in: {pkl_path}")
        return out

    use_pv = bool(dcfg.get("use_pv", False))
    pv_camera = str(dcfg.get("pv_camera", "CAM_FRONT"))
    categories = [str(x) for x in scfg.get("categories", [])]
    sat_root = dcfg["satmap_root"]
    nuscenes_root = dcfg["nuscenes_root"]

    n_scan = len(infos) if max_scan <= 0 else min(len(infos), max_scan)
    out["scanned_items"] = n_scan
    for info in infos[:n_scan]:
        token = str(info.get("token", ""))
        sat_path = os.path.join(sat_root, f"{token}_satellite.png")
        if token and os.path.exists(sat_path):
            out["sat_exists"] += 1
        else:
            out["sat_missing"] += 1
            if len(out["missing_sat_examples"]) < 5:
                out["missing_sat_examples"].append(sat_path)

        ann = info.get("annotation", None)
        if isinstance(ann, dict):
            out["ann_exists"] += 1
            has_any = False
            for cat in categories:
                arr = ann.get(cat, [])
                if isinstance(arr, list) and len(arr) > 0:
                    has_any = True
                    break
            if has_any:
                out["ann_nonempty"] += 1

        if use_pv:
            cams = info.get("cams", {}) if isinstance(info.get("cams", {}), dict) else {}
            cam = cams.get(pv_camera, {}) if isinstance(cams.get(pv_camera, {}), dict) else {}
            rel = str(cam.get("data_path", ""))
            if rel:
                pv_path = os.path.join(nuscenes_root, _norm_nuscenes_rel_path(rel))
                if os.path.exists(pv_path):
                    out["pv_exists"] += 1
                else:
                    out["pv_missing"] += 1
                    if len(out["missing_pv_examples"]) < 5:
                        out["missing_pv_examples"].append(pv_path)
            else:
                out["pv_missing"] += 1

    return out


def _check_opensatmap_split(cfg: Dict, split: str, max_scan: int) -> Dict:
    dcfg = cfg["data"]
    root = str(dcfg["opensatmap_root"])
    split_root = str(dcfg.get("opensatmap_split_dir", os.path.join(root, "picuse20trainvaltest")))
    split_dir = os.path.join(split_root, split)
    ann_path = str(dcfg.get("opensatmap_ann_json", os.path.join(root, "annotrainval20.json")))
    ann, err = _load_ann_json(ann_path)

    out = {
        "kind": "opensatmap",
        "split": split,
        "path": split_dir,
        "ann_path": ann_path,
        "total_items": 0,
        "scanned_items": 0,
        "sat_exists": 0,
        "sat_missing": 0,
        "ann_exists": 0,
        "ann_nonempty": 0,
        "pv_exists": 0,
        "pv_missing": 0,
        "errors": [],
        "missing_sat_examples": [],
        "missing_pv_examples": [],
    }
    if err:
        out["errors"].append(err)
        return out
    if not os.path.isdir(split_dir):
        out["errors"].append(f"missing split dir: {split_dir}")
        return out

    names = sorted(os.listdir(split_dir))
    out["total_items"] = len(names)
    if len(names) == 0:
        out["errors"].append(f"empty split dir: {split_dir}")
        return out

    n_scan = len(names) if max_scan <= 0 else min(len(names), max_scan)
    out["scanned_items"] = n_scan
    for name in names[:n_scan]:
        p = os.path.join(split_dir, name)
        if os.path.isfile(p):
            out["sat_exists"] += 1
        else:
            out["sat_missing"] += 1
            if len(out["missing_sat_examples"]) < 5:
                out["missing_sat_examples"].append(p)
        rec = ann.get(name)
        if isinstance(rec, dict):
            out["ann_exists"] += 1
            lines = rec.get("lines", [])
            if isinstance(lines, list) and len(lines) > 0:
                out["ann_nonempty"] += 1

    return out


def _build_sdmap_token_set(sdmap_root: str) -> set:
    toks = set()
    paths = glob.glob(os.path.join(sdmap_root, "**", "*.pkl"), recursive=True)
    for p in paths:
        name = os.path.basename(p)
        if len(name) == 36:
            toks.add(name[:-4])
    return toks


def _check_nuscenes_sdmap_split(cfg: Dict, split: str, max_scan: int, sdmap_token_set: set) -> Dict:
    dcfg = cfg["data"]
    temporal_pkl_dir = str(dcfg.get("nuscenes_temporal_pkl_dir", dcfg.get("nuscenes_root", "")))
    temporal_prefix = str(dcfg.get("nuscenes_temporal_pkl_prefix", "vad_nuscenes_infos_temporal_"))
    temporal_pkl_path = str(
        dcfg.get("nuscenes_temporal_pkl_path", os.path.join(temporal_pkl_dir, f"{temporal_prefix}{split}.pkl"))
    )
    infos, err = _load_infos(temporal_pkl_path)
    out = {
        "kind": "nuscenes_sdmap",
        "split": split,
        "path": temporal_pkl_path,
        "total_items": len(infos),
        "scanned_items": 0,
        "sat_exists": 0,
        "sat_missing": 0,
        "ann_exists": 0,
        "ann_nonempty": 0,
        "pv_exists": 0,
        "pv_missing": 0,
        "errors": [],
        "missing_sat_examples": [],
        "missing_pv_examples": [],
    }
    if err:
        out["errors"].append(err)
        return out
    if len(infos) == 0:
        out["errors"].append(f"empty infos in: {temporal_pkl_path}")
        return out

    use_pv = bool(dcfg.get("use_pv", False))
    pv_camera = str(dcfg.get("pv_camera", "CAM_FRONT"))
    nuscenes_root = str(dcfg.get("nuscenes_root", ""))
    sat_root = str(dcfg.get("satmap_root", ""))
    n_scan = len(infos) if max_scan <= 0 else min(len(infos), max_scan)
    out["scanned_items"] = n_scan

    for info in infos[:n_scan]:
        token = str(info.get("token", ""))
        sat_path = os.path.join(sat_root, f"{token}_satellite.png")
        if token and os.path.exists(sat_path):
            out["sat_exists"] += 1
        else:
            out["sat_missing"] += 1
            if len(out["missing_sat_examples"]) < 5:
                out["missing_sat_examples"].append(sat_path)

        if token in sdmap_token_set:
            out["ann_exists"] += 1
            # sd_map pkl may contain empty class lists; count as non-empty if token file exists.
            out["ann_nonempty"] += 1

        if use_pv:
            cams = info.get("cams", {}) if isinstance(info.get("cams", {}), dict) else {}
            cam = cams.get(pv_camera, {}) if isinstance(cams.get(pv_camera, {}), dict) else {}
            rel = str(cam.get("data_path", ""))
            if rel:
                pv_path = os.path.join(nuscenes_root, _norm_nuscenes_rel_path(rel))
                if os.path.exists(pv_path):
                    out["pv_exists"] += 1
                else:
                    out["pv_missing"] += 1
                    if len(out["missing_pv_examples"]) < 5:
                        out["missing_pv_examples"].append(pv_path)
            else:
                out["pv_missing"] += 1
    return out


def _print_split_report(rep: Dict) -> None:
    if rep["kind"] == "opensatmap":
        print(f"[Split:{rep['split']}] split_dir={rep['path']}")
        print(f"  ann_json={rep.get('ann_path', '')}")
    else:
        print(f"[Split:{rep['split']}] pkl={rep['path']}")
    print(
        f"  items={rep['total_items']} scanned={rep['scanned_items']} "
        f"image_exists={rep['sat_exists']} image_missing={rep['sat_missing']}"
    )
    print(
        f"  ann_exists={rep['ann_exists']} ann_nonempty={rep['ann_nonempty']} "
        f"pv_exists={rep['pv_exists']} pv_missing={rep['pv_missing']}"
    )
    for e in rep["errors"]:
        print(f"  ERROR: {e}")
    if rep["missing_sat_examples"]:
        print("  missing_image_examples:")
        for p in rep["missing_sat_examples"]:
            print(f"    - {p}")
    if rep["missing_pv_examples"]:
        print("  missing_pv_examples:")
        for p in rep["missing_pv_examples"]:
            print(f"    - {p}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_scan_per_split", type=int, default=512)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if "data" not in cfg or "serialization" not in cfg:
        print("ERROR: config must contain both 'data' and 'serialization'")
        sys.exit(1)
    dcfg = cfg["data"]
    source = str(dcfg.get("source", "nuscenes_maptr")).lower()

    print(f"[Config] {args.config}")
    print(f"[Source] {source}")

    fatal = False
    splits = [str(dcfg.get("train_split", "train"))]
    val_split = str(dcfg.get("val_split", "val"))
    if val_split not in splits:
        splits.append(val_split)

    reports = []
    if source == "opensatmap":
        print(f"[Path] opensatmap_root={dcfg.get('opensatmap_root', '')}")
        print(f"[Path] opensatmap_ann_json={dcfg.get('opensatmap_ann_json', '')}")
        print(f"[Path] opensatmap_split_dir={dcfg.get('opensatmap_split_dir', '')}")
        for p in [dcfg.get("opensatmap_root", ""), dcfg.get("opensatmap_ann_json", "")]:
            if not p or not os.path.exists(p):
                print(f"ERROR: path not found -> {p}")
                fatal = True
        reports = [_check_opensatmap_split(cfg, split=s, max_scan=int(args.max_scan_per_split)) for s in splits]
    elif source == "nuscenes_sdmap":
        print(f"[Path] nuscenes_root={dcfg.get('nuscenes_root', '')}")
        print(f"[Path] satmap_root={dcfg.get('satmap_root', '')}")
        print(f"[Path] nuscenes_sdmap_root={dcfg.get('nuscenes_sdmap_root', '')}")
        print(f"[Path] nuscenes_temporal_pkl_dir={dcfg.get('nuscenes_temporal_pkl_dir', '')}")
        print(f"[Mode] use_pv={bool(dcfg.get('use_pv', False))} pv_camera={dcfg.get('pv_camera', 'CAM_FRONT')}")
        for p in [
            dcfg.get("nuscenes_root", ""),
            dcfg.get("satmap_root", ""),
            dcfg.get("nuscenes_sdmap_root", ""),
            dcfg.get("nuscenes_temporal_pkl_dir", ""),
        ]:
            if not p or not os.path.exists(p):
                print(f"ERROR: path not found -> {p}")
                fatal = True
        sdmap_token_set = _build_sdmap_token_set(str(dcfg.get("nuscenes_sdmap_root", "")))
        reports = [
            _check_nuscenes_sdmap_split(cfg, split=s, max_scan=int(args.max_scan_per_split), sdmap_token_set=sdmap_token_set)
            for s in splits
        ]
    else:
        print(f"[Path] nuscenes_root={dcfg.get('nuscenes_root', '')}")
        print(f"[Path] satmap_root={dcfg.get('satmap_root', '')}")
        print(f"[Path] pkl_dir={dcfg.get('nuscenes_map_pkl_dir', '')}")
        print(f"[Mode] use_pv={bool(dcfg.get('use_pv', False))} pv_camera={dcfg.get('pv_camera', 'CAM_FRONT')}")
        for p in [
            dcfg.get("nuscenes_root", ""),
            dcfg.get("satmap_root", ""),
            dcfg.get("nuscenes_map_pkl_dir", ""),
        ]:
            if not p or not os.path.exists(p):
                print(f"ERROR: path not found -> {p}")
                fatal = True
        reports = [_check_nuscenes_maptr_split(cfg, split=s, max_scan=int(args.max_scan_per_split)) for s in splits]

    for rep in reports:
        _print_split_report(rep)
        if rep["errors"]:
            fatal = True
        if rep["scanned_items"] > 0 and rep["sat_exists"] == 0:
            print(f"  ERROR: zero matched images for split={rep['split']}")
            fatal = True
        if rep["scanned_items"] > 0 and rep["ann_exists"] == 0:
            print(f"  ERROR: zero matched annotations for split={rep['split']}")
            fatal = True
        if source in {"nuscenes_maptr", "nuscenes_sdmap"} and bool(dcfg.get("use_pv", False)) and rep["scanned_items"] > 0 and rep["pv_exists"] == 0:
            print(f"  ERROR: zero matched PV images for split={rep['split']} and camera={dcfg.get('pv_camera')}")
            fatal = True

    if fatal:
        print("[Result] FAILED (dataset/config is not ready)")
        sys.exit(1)
    print("[Result] OK (dataset/config looks ready)")


if __name__ == "__main__":
    main()
