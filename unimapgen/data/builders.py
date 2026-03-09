import os

from unimapgen.data.dataset import DatasetConfig, NuScenesSatelliteMapDataset
from unimapgen.data.nuscenes_sdmap_dataset import NuScenesSDMapDataset, NuScenesSDMapDatasetConfig
from unimapgen.data.opensatmap_dataset import OpenSatMapDataset, OpenSatMapDatasetConfig


def _build_nuscenes_maptr_dataset(cfg, split: str, max_samples, train_augment: bool):
    dcfg = cfg["data"]
    scfg = cfg["serialization"]
    common = dict(
        nuscenes_root=dcfg["nuscenes_root"],
        pkl_dir=dcfg["nuscenes_map_pkl_dir"],
        satmap_root=dcfg["satmap_root"],
        image_size=dcfg["image_size"],
        use_pv=bool(dcfg.get("use_pv", False)),
        pv_camera=dcfg.get("pv_camera", "CAM_FRONT"),
        pv_num_frames=int(dcfg.get("pv_num_frames", 1)),
        pv_image_size=list(dcfg.get("pv_image_size", [224, 400])),
        sample_interval_meter=scfg["sample_interval_meter"],
        max_lines=scfg["max_lines"],
        max_points_per_line=scfg["max_points_per_line"],
        categories=scfg["categories"],
        max_seq_len=scfg["max_seq_len"],
        coord_num_bins=scfg.get("coord_num_bins"),
        angle_num_bins=int(scfg.get("angle_num_bins", 360)),
        use_state_update=bool(dcfg.get("use_state_update", False)),
        state_update_mode=dcfg.get("state_update_mode", "sample_prev"),
        state_prefix_mode=dcfg.get("state_prefix_mode", "all"),
        use_text_prompt=bool(dcfg.get("use_text_prompt", False)),
        text_prompt_mode=dcfg.get("text_prompt_mode", "full_map"),
        text_num_trace_points=int(dcfg.get("text_num_trace_points", 8)),
    )
    return NuScenesSatelliteMapDataset(
        DatasetConfig(
            split=split,
            max_samples=max_samples,
            train_augment=bool(train_augment),
            aug_rot90_prob=float(dcfg.get("aug_rot90_prob", 0.0)) if train_augment else 0.0,
            aug_hflip_prob=float(dcfg.get("aug_hflip_prob", 0.0)) if train_augment else 0.0,
            aug_vflip_prob=float(dcfg.get("aug_vflip_prob", 0.0)) if train_augment else 0.0,
            **common,
        )
    )


def _build_opensatmap_dataset(cfg, split: str, max_samples, train_augment: bool):
    dcfg = cfg["data"]
    scfg = cfg["serialization"]
    opensatmap_root = str(dcfg["opensatmap_root"])
    ann_json_path = str(dcfg.get("opensatmap_ann_json", os.path.join(opensatmap_root, "annotrainval20.json")))
    split_dir = dcfg.get("opensatmap_split_dir")
    return OpenSatMapDataset(
        OpenSatMapDatasetConfig(
            opensatmap_root=opensatmap_root,
            ann_json_path=ann_json_path,
            split=split,
            image_size=int(dcfg["image_size"]),
            max_samples=max_samples,
            sample_interval_meter=float(scfg["sample_interval_meter"]),
            meter_per_pixel=float(dcfg.get("meter_per_pixel", 0.15)),
            max_lines=int(scfg["max_lines"]),
            max_points_per_line=int(scfg["max_points_per_line"]),
            categories=list(scfg["categories"]),
            line_types=list(scfg.get("line_types", [])),
            max_seq_len=int(scfg["max_seq_len"]),
            coord_num_bins=scfg.get("coord_num_bins"),
            angle_num_bins=int(scfg.get("angle_num_bins", 360)),
            train_augment=bool(train_augment),
            aug_rot90_prob=float(dcfg.get("aug_rot90_prob", 0.0)) if train_augment else 0.0,
            aug_hflip_prob=float(dcfg.get("aug_hflip_prob", 0.0)) if train_augment else 0.0,
            aug_vflip_prob=float(dcfg.get("aug_vflip_prob", 0.0)) if train_augment else 0.0,
            split_dir=str(split_dir) if split_dir else None,
        )
    )


def _build_nuscenes_sdmap_dataset(cfg, split: str, max_samples, train_augment: bool):
    dcfg = cfg["data"]
    scfg = cfg["serialization"]
    temporal_pkl_dir = str(dcfg.get("nuscenes_temporal_pkl_dir", dcfg.get("nuscenes_root", "")))
    temporal_prefix = str(dcfg.get("nuscenes_temporal_pkl_prefix", "vad_nuscenes_infos_temporal_"))
    temporal_pkl_path = str(dcfg.get("nuscenes_temporal_pkl_path", os.path.join(temporal_pkl_dir, f"{temporal_prefix}{split}.pkl")))
    return NuScenesSDMapDataset(
        NuScenesSDMapDatasetConfig(
            nuscenes_root=str(dcfg["nuscenes_root"]),
            temporal_pkl_path=temporal_pkl_path,
            sdmap_root=str(dcfg["nuscenes_sdmap_root"]),
            satmap_root=str(dcfg["satmap_root"]),
            image_size=int(dcfg["image_size"]),
            use_pv=bool(dcfg.get("use_pv", False)),
            pv_camera=str(dcfg.get("pv_camera", "CAM_FRONT")),
            pv_num_frames=int(dcfg.get("pv_num_frames", 1)),
            pv_image_size=list(dcfg.get("pv_image_size", [224, 400])),
            max_samples=max_samples,
            sample_interval_meter=float(scfg["sample_interval_meter"]),
            meter_range_half=float(dcfg.get("meter_range_half", 180.0)),
            max_lines=int(scfg["max_lines"]),
            max_points_per_line=int(scfg["max_points_per_line"]),
            categories=list(scfg["categories"]),
            max_seq_len=int(scfg["max_seq_len"]),
            coord_num_bins=scfg.get("coord_num_bins"),
            angle_num_bins=int(scfg.get("angle_num_bins", 360)),
            train_augment=bool(train_augment),
            aug_rot90_prob=float(dcfg.get("aug_rot90_prob", 0.0)) if train_augment else 0.0,
            aug_hflip_prob=float(dcfg.get("aug_hflip_prob", 0.0)) if train_augment else 0.0,
            aug_vflip_prob=float(dcfg.get("aug_vflip_prob", 0.0)) if train_augment else 0.0,
        )
    )


def build_dataset_from_cfg(cfg, split: str, max_samples=None, train_augment: bool = False):
    source = str(cfg.get("data", {}).get("source", "nuscenes_maptr")).lower()
    if source == "opensatmap":
        return _build_opensatmap_dataset(cfg, split=split, max_samples=max_samples, train_augment=train_augment)
    if source == "nuscenes_sdmap":
        return _build_nuscenes_sdmap_dataset(cfg, split=split, max_samples=max_samples, train_augment=train_augment)
    return _build_nuscenes_maptr_dataset(cfg, split=split, max_samples=max_samples, train_augment=train_augment)
