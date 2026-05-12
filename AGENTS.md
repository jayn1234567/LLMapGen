# MLLM_project — 工作文档 (branch: qwen3vl_dinov3)

## 项目概述

基于 LLaVA 架构的 BEV 道路几何理解多模态大模型（VLM），使用 DINOv2/DINOv3 作为视觉编码器 + Qwen2/Qwen3 作为语言模型，完成 BEV 图像的道路中心线重建。

本分支实现 **Qwen3-VL DeepStack 真实架构**：每层 ViT 特征通过独立 merger 注入 LLM 不同层。

- **分支**: `qwen3vl_dinov3`（基于 `qwen3vl_Dinov2` 分出，加入 DINOv3 支持）
- **环境**: conda `fastvlm`
- **依赖**: transformers>=4.51.0

---

## 目录结构

```
MLLM_project/
├── AGENTS.md                          # 本文档
├── README.md
├── pyproject.toml
├── configs/                           # DeepSpeed 配置
│   ├── deepspeed_zero2.json
│   └── deepspeed_zero3.json
│
├── llava/
│   ├── conversation.py                # 对话模板
│   ├── constants.py
│   ├── mm_utils.py
│   ├── model/
│   │   ├── builder.py                 # load_pretrained_model()
│   │   ├── llava_arch.py              # 多模态融合核心
│   │   ├── qwen3vl_extractor.py       # Qwen3-VL → LLM 提取
│   │   ├── language_model/
│   │   │   ├── llava_qwen.py          # Qwen2 LLM
│   │   │   └── llava_qwen3.py         # Qwen3 LLM
│   │   ├── multimodal_encoder/
│   │   │   ├── builder.py             # 路由到 DINOv2/DINOv3/CLIP
│   │   │   ├── dino_config.py         # DINO 变体注册表 ★
│   │   │   ├── dinov2_encoder.py      # DINOv2 编码器
│   │   │   ├── dinov3_encoder.py      # DINOv3 编码器 ★
│   │   │   ├── deepstack.py           # DeepStack (per-layer merger) ★
│   │   │   └── clip_encoder.py
│   │   └── multimodal_projector/
│   └── train/
│       ├── train_qwen.py              # 主训练脚本
│   └── llava_trainer.py           # LLaVATrainer — HF Trainer 子类 (含分组 LR)
│
└── scripts/
    ├── gpu/                           # GPU 训练脚本 ★ 7 个
    │   ├── train_dinov2_qwen2-1.5b.sh
    │   ├── train_dinov2_qwen3-8b.sh
    │   ├── train_dinov2_qwen3vl-2b.sh
    │   ├── train_dinov2_qwen3vl-8b.sh
    │   ├── train_dinov3_qwen2-1.5b.sh     # 新增
    │   ├── train_dinov3_qwen3vl-2b.sh     # 新增
    │   └── train_dinov3_qwen3vl-8b.sh     # 新增
    ├── npu/                           # NPU 训练+推理脚本 ★ 4 个
    │   ├── train_dinov2_qwen2-1.5b_npu.sh
    │   ├── train_dinov2_qwen3vl-8b_npu.sh
    │   ├── train_dinov3_qwen2-1.5b_npu.sh  # 新增
    │   └── train_dinov3_qwen3vl-8b_npu.sh  # 新增
    ├── qwen2/
    ├── qwen3vl/
    ├── infer_centerline_checkpoint.py # 推理引擎
    ├── visualize_centerline.py
    ├── summarize_centerline_eval.py
    ├── deepspeed_zero2.json
    ├── deepspeed_zero3.json
    └── deepspeed_zero3_no_merge.json  # 训练不合并，结束后脚本合并
```

---

## 模型架构

### 完整数据流

```
DINOv2/DINOv3 ViT (frozen)
    │ output_hidden_states=True
    ├── main layer → mm_projector (MLP) → 替换 <image> token → LLM embedding 层
    └── deepstack layers → 独立 merger MLP → 残差注入 LLM layer 0, 1, 2, 3
```

### DeepStack 注入

```
DINOv3 Layer 23 → main feature → mm_projector → 替换 <image> token
DINOv3 Layer 18 → Merger[0] → 残差加到 LLM Layer 0
DINOv3 Layer 12 → Merger[1] → 残差加到 LLM Layer 1
DINOv3 Layer 6  → Merger[2] → 残差加到 LLM Layer 2
```

### DeepStackMerger

```
LayerNorm(vit_dim) → Linear(vit_dim → llm_dim) → GELU → Linear(llm_dim → llm_dim)
```

每层独立的 merger，不共享权重。

---

## DINO 编码器配置

### 自动检测

| 路径关键字 | 编码器 | 默认 DeepStack | 默认 image_size |
|-----------|--------|---------------|-----------------|
| `dinov2` | DINOv2VisionTower | [6,12,18,23] | 518 |
| `dinov3-vitl16` | DINOv3VisionTower | [6,12,18,23] | 224 |
| `dinov3-vitb16` | DINOv3VisionTower | [3,6,9,11] | 224 |

自动检测由 `dino_config.py` 实现，`builder.py` 调用。可通过 `--deepstack_visual_indexes` 和 `--input_image_size` 覆盖。

### DINO 路径别名

| 别名 | 对应 config key |
|------|----------------|
| `dinov2_large`, `dinov2_l` | `dinov2-large` |
| `dinov3_small`, `dinov3_s` | `dinov3-vits16` |
| `dinov3_base`, `dinov3_b` | `dinov3-vitb16` |
| `dinov3_large`, `dinov3_l` | `dinov3-vitl16` |
| `dinov3_huge`, `dinov3_h` | `dinov3-vith16plus` |

优先级：`mm_vision_tower_type` → `vision_tower` 路径关键字。路径应包含明确的 DINO key 或别名；如果只能看出 `dinov...` 但无法判断具体类型，会直接报错。

### DINOv2 vs DINOv3

| | DINOv2-L | DINOv3-L |
|---|---|---|
| 模型类 | Dinov2Model | DINOv3ViTModel |
| patch_size | 14 | 16 |
| 位置编码 | 绝对嵌入 | RoPE |
| register tokens | 0 | 4 |
| skip_tokens | 1 (CLS) | 5 (CLS+4) |

---

## 训练参数

### ModelArguments (新增参数以 ★ 标注)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_name_or_path` | str | `facebook/opt-125m` | LLM checkpoint 路径 |
| `--version` | str | `v0` | 对话模板: `conv_qwen_2/3_Dinov2_huawei` |
| `--vision_tower` | str | None | 视觉编码器路径 |
| `--mm_vision_select_layer` | int | -1 | ViT 层选择 (-2=倒数第二层) |
| `--mm_projector_type` | str | `linear` | 投影器: `mlp2x_gelu` |
| `--mm_vision_select_feature` | str | `patch` | `patch`(去CLS) / `cls_patch` |
| `--mm_patch_merge_type` | str | `flat` | `flat` / `spatial` / `spatial_unpad` |
| `--unfreeze_mm_vision_tower` | bool | False | 解冻 Vision Tower |
| `--freeze_llm` | bool | False | 冻结 LLM, 仅训 ViT + Projector |
| ★ `--deepstack_visual_indexes` | List[int] | None | DeepStack 选择层，如 `6 12 18 23` |
| ★ `--input_image_size` | int | None | ViT 输入尺寸 (不传则用默认) |
| ★ `--mm_vision_tower_type` | str | None | 可选视觉塔类型: `dinov2` / `dinov3`，通常由 checkpoint 或路径自动识别 |

### DeepSpeed 配置

| 配置文件 | gather_16bit_weights | 用途 |
|---------|---------------------|------|
| `deepspeed_zero3.json` | true | 训练时自动合并权重 (可能 OOM) |
| `deepspeed_zero3_no_merge.json` | false | 训练时保持分片 (推荐) |

使用 no_merge 时，训练结束后 NPU 脚本会自动运行 `zero_to_fp32.py` 将每个 checkpoint 的分片合并为 `model.safetensors`。

### 推荐训练命令

```bash
# Qwen2 + DINOv2 + DeepStack
python -m llava.train.train_qwen \
    --model_name_or_path checkpoints/llava-fastvithd_7b_stage2 \
    --version conv_qwen_2_Dinov2_huawei \
    --vision_tower checkpoints/facebook_dinov2-large \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --deepstack_visual_indexes 6 12 18 23 \
    --data_path data/train.jsonl \
    --image_folder data/img \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir outputs/my_exp \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --mm_projector_lr 5e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --save_steps 1000 \
    --logging_steps 10
```

---

## 验证矩阵

| 视觉编码器 | LLM | DeepStack | 状态 |
|---|---|---|---|
| DINOv2-L | Qwen2.5-1.5B | ✅ [6,12,18,23] | ✅ |
| DINOv2-L | Qwen3-VL-2B | ✅ [6,12,18,23] | ✅ |
| DINOv3-L | Qwen2.5-1.5B | ✅ [6,12,18,23] | ✅ |
| DINOv3-L | Qwen3-VL-2B | ✅ [6,12,18,23] | ✅ |
| DINOv3-B | Qwen2.5-1.5B | ✅ [3,6,9,11] | ✅ |
