def build_model_from_cfg(cfg, vocab_size: int, pad_id: int):
    model_cfg = cfg.get("model", {})
    arch = model_cfg.get("arch", "baseline_v1")
    if arch == "paper":
        from .unimapgen_paper import UniMapGenPaper

        return UniMapGenPaper(
            vocab_size=vocab_size,
            pad_id=pad_id,
            bev_encoder_name=model_cfg.get("bev_encoder_name", "facebook/dinov2-vitl14"),
            llm_name=model_cfg.get("llm_name", "Qwen/Qwen2.5-1.5B-Instruct"),
            local_files_only=bool(model_cfg.get("local_files_only", False)),
            use_fallback=bool(model_cfg.get("use_fallback", False)),
            use_pv=bool(cfg.get("data", {}).get("use_pv", False)),
            pv_cnn_channels=tuple(model_cfg.get("pv_cnn_channels", model_cfg.get("cnn_channels", [32, 64, 128]))),
            pv_memory_tokens_hw=tuple(model_cfg.get("pv_memory_tokens_hw", [2, 4])),
            use_text_prompt=bool(cfg.get("data", {}).get("use_text_prompt", False)),
            num_prompt_types=int(model_cfg.get("num_prompt_types", 4)),
            d_model_fallback=int(model_cfg.get("d_model_fallback", 256)),
            bev_token_hw=tuple(model_cfg.get("bev_token_hw", [8, 8])),
            bev_patch_size=int(model_cfg.get("bev_patch_size", 14)),
            bev_drop_cls_token=bool(model_cfg.get("bev_drop_cls_token", True)),
            bev_normalize_input=bool(model_cfg.get("bev_normalize_input", True)),
        )

    from .unimapgen_v1 import UniMapGenV1

    return UniMapGenV1(
        vocab_size=vocab_size,
        d_model=int(model_cfg["d_model"]),
        num_heads=int(model_cfg["num_heads"]),
        num_decoder_layers=int(model_cfg["num_decoder_layers"]),
        ff_dim=int(model_cfg["ff_dim"]),
        dropout=float(model_cfg["dropout"]),
        cnn_channels=tuple(model_cfg["cnn_channels"]),
        memory_tokens_hw=tuple(model_cfg["memory_tokens_hw"]),
        use_pv=bool(cfg["data"].get("use_pv", False)),
        pv_cnn_channels=tuple(model_cfg.get("pv_cnn_channels", model_cfg["cnn_channels"])),
        pv_memory_tokens_hw=tuple(model_cfg.get("pv_memory_tokens_hw", [2, 4])),
        use_text_prompt=bool(cfg["data"].get("use_text_prompt", False)),
        num_prompt_types=int(model_cfg.get("num_prompt_types", 4)),
        pad_id=pad_id,
    )


def __getattr__(name):
    if name == "DINOv2LaneSeg":
        from .dino_lane_seg import DINOv2LaneSeg

        return DINOv2LaneSeg
    if name == "QwenSatelliteMapGenerator":
        from .qwen_map_generator import QwenSatelliteMapGenerator

        return QwenSatelliteMapGenerator
    if name == "UniMapGenPaper":
        from .unimapgen_paper import UniMapGenPaper

        return UniMapGenPaper
    if name == "UniMapGenV1":
        from .unimapgen_v1 import UniMapGenV1

        return UniMapGenV1
    raise AttributeError(name)


__all__ = ["UniMapGenV1", "UniMapGenPaper", "DINOv2LaneSeg", "QwenSatelliteMapGenerator", "build_model_from_cfg"]
