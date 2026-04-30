# MLLM_project — 工作文档

## 项目概述

基于 LLaVA 架构的 BEV 道路几何理解多模态大模型（VLM），使用 DINOv2 作为视觉编码器 + Qwen2/Qwen3 作为语言模型，完成 BEV (Bird's Eye View) 图像的道路中心线重建任务。

- **分支**: `qwen3vl_Dinov2`（主工作分支，不合并到 main）
- **环境**: conda `fastvlm`
- **依赖**: transformers>=4.51.0 (Qwen3 支持), tokenizers>=0.21, huggingface-hub>=0.25.1

---

## 目录结构

```
MLLM_project/
├── AGENTS.md                          # 本文档 — 项目工作记录
├── README.md                          # 占位
├── pyproject.toml                     # 包配置 (llava v1.2.2)
├── .gitignore
├── setup_npu.sh                       # NPU (Ascend) 环境配置
│
├── configs/                           # DeepSpeed 配置
│   ├── deepspeed_zero2.json           # ZeRO-2: 优化器状态分片
│   └── deepspeed_zero3.json           # ZeRO-3: 全参数分片
│
├── llava/                             # 核心 Python 包
│   ├── __init__.py                    # 导出 : LlavaLlamaForCausalLM, LlavaQwen2ForCausalLM, LlavaQwen3ForCausalLM
│   ├── constants.py                   # 常量: IMAGE_TOKEN_INDEX=-200, IGNORE_INDEX
│   ├── conversation.py                # 对话模板: QWEN_2, QWEN_3, centerline_coord, Dinov2_huawei
│   ├── mm_utils.py                    # 图像处理、tokenizer 辅助函数
│   ├── utils.py                       # 日志、设备管理
│   │
│   ├── model/                         # 模型核心
│   │   ├── __init__.py                # 导出所有 LM 类
│   │   ├── builder.py                 # load_pretrained_model() — 模型加载入口
│   │   ├── llava_arch.py              # LlavaMetaModel + LlavaMetaForCausalLM — 多模态融合核心
│   │   ├── apply_delta.py             # LoRA delta 应用
│   │   ├── make_delta.py              # Delta 创建
│   │   ├── consolidate.py             # 权重合并
│   │   ├── utils.py
│   │   │
│   │   ├── language_model/            # LLM 后端
│   │   │   ├── llava_qwen.py          # LlavaQwen2ForCausalLM (model_type="llava_qwen2") ★ 主力
│   │   │   ├── llava_qwen3.py         # LlavaQwen3ForCausalLM (model_type="llava_qwen3") ★ Qwen3 适配
│   │   │   ├── llava_llama.py         # LlavaLlamaForCausalLM
│   │   │   ├── llava_mistral.py       # LlavaMistralForCausalLM
│   │   │   └── llava_mpt.py           # LlavaMptForCausalLM
│   │   │
│   │   ├── multimodal_encoder/        # 视觉编码器
│   │   │   ├── builder.py             # build_vision_tower() — 路由到 DINOv2/CLIP/MobileCLIP
│   │   │   ├── deepstack.py           # ★ DeepStack 模块 — Qwen3-VL 多层特征融合
│   │   │   ├── dinov2_encoder.py      # DINOv2VisionTower ★ 主视觉编码器 (含 DeepStack 支持)
│   │   │   ├── clip_encoder.py        # CLIPVisionTower / CLIPVisionTowerS2
│   │   │   ├── mobileclip_encoder.py  # MobileCLIPVisionTower (Apple FastViT)
│   │   │   └── mobileclip/            # MCi/FastViT 实现
│   │   │
│   │   ├── multimodal_projector/
│   │   │   └── builder.py             # build_vision_projector() — linear / mlpNx_gelu / identity
│   │   │
│   │   └── qwen3vl_extractor.py       # ★ Qwen3-VL → 纯 LLM 权重自动提取
│   │
│   ├── train/                         # 训练
│   │   ├── train_qwen.py              # ★ 主训练脚本 (Qwen2/Qwen3 + DINOv2)
│   │   ├── train.py                   # 原始训练脚本 (Llama-based)
│   │   ├── train_mem.py               # 内存优化训练
│   │   ├── llava_trainer.py           # LLaVATrainer — HF Trainer 子类
│   │   └── llama_flash_attn_monkey_patch.py
│   │
│   └── serve/                         # 模型服务 (gradio/FastAPI)
│       ├── cli.py, controller.py, model_worker.py
│       └── gradio_web_server.py
│
├── data_process/                      # 数据处理
│   └── convert_cloud_format.py        # 云端数据 → LLaVA JSONL 格式转换
│
├── model_export/                      # Apple Silicon 导出
│   ├── README.md
│   ├── export_vision_encoder.py       # CoreML 视觉编码器导出
│   └── fastvlm_mlx-vlm.patch          # mlx-vlm 补丁
│
├── scripts/                           # 训练 & 评估脚本
│   ├── train_dinov2_qwen2-1.5b.sh      # 单卡: DINOv2 + Qwen2-1.5B + DeepStack
│   ├── train_dinov2_qwen3-8b.sh        # 多卡: DINOv2 + Qwen3-8B + DeepStack + ZeRO-3
│   ├── train_dinov2_qwen3vl-2b.sh      # 单卡: DINOv2 + Qwen3-VL-2B LLM(自动提取) + DeepStack
│   ├── train_dinov2_qwen3vl-8b.sh      # 多卡: DINOv2 + Qwen3-VL-8B LLM(自动提取) + DeepStack + ZeRO-3
│   ├── train_dinov2_qwen2-1.5b_npu.sh  # NPU: DINOv2 + Qwen2-1.5B + DeepStack + ZeRO-3, 训练后自动推理
│   ├── train_dinov2_qwen3vl-8b_npu.sh  # NPU: DINOv2 + Qwen3-VL-8B LLM(自动提取) + DeepStack + ZeRO-3, 训练后自动推理
│   ├── infer_dinov2_centerline.sh      # 推理: 加载 checkpoint → 中心线 JSON + 可视化
│   ├── infer_centerline_checkpoint.py  # 推理引擎 (被 infer_dinov2_centerline.sh 调用)
│   ├── summarize_centerline_eval.py    # 评估汇总
│   ├── visualize_centerline.py         # 中心线可视化
│   ├── deepspeed_zero2.json            # ZeRO-2 配置
│   └── deepspeed_zero3.json            # ZeRO-3 配置
│
├── predict.py                         # 推理 CLI
├── test_batch.py                      # 批量测试
└── test_predict.py                    # 推理测试
```

---

## 模型架构

```
Image → DINOv2VisionTower (ViT, frozen) → DeepStack (可选) → mm_projector (MLP) → LLM (Qwen2/Qwen3)
                                                                ↓
                                              image embeddings interleaved with text embeddings
```

### 推理流程

1. **图像处理**: DINOv2ImageProcessor → resize/crop 到固定尺寸 (448×448)
2. **ViT 编码**: Dinov2Model 输出 hidden_states (all layers, `output_hidden_states=True`)
3. **特征选择**:
   - 普通模式: `feature_select()` 选 `select_layer` (默认 -2) 的单层特征
   - DeepStack 模式: 从 `deepstack_visual_indexes` 指定层提取特征 → DeepStack 加权融合
4. **投影**: `mm_projector` (mlp2x_gelu) 将 1024-dim → LLM hidden_size
5. **多模态融合**: `<image>` token 位置替换为投影后的图像特征
6. **LLM 解码**: Qwen2/Qwen3 自回归生成

### 训练流程（两阶段）

| 阶段 | 训练组件 | 冻结组件 |
|------|---------|---------|
| Stage1 (对齐) | mm_projector only | DINOv2, LLM |
| Stage2 (微调) | 全模型 / LoRA | 可选冻结 DINOv2 |

---

## DeepStack 模块

复现自 Qwen3-VL 论文 (arXiv:2511.21631)。

**原理**: 传统 LLaVA 只使用 ViT 最后一层输出，DeepStack 融合多个中间层的特征，捕获更丰富的视觉细节。Qwen3-VL-2B 原版使用层 [5, 11, 17] (24 层 ViT)，Qwen3-VL-8B 原版使用层 [8, 16, 24] (27 层 ViT)。

**接入 LLM 的数据流**:

```
DINOv2 ViT (24 层, frozen)
    │
    ├── Layer 6  hidden_states  [B, 1025, 1024] ──┐
    ├── Layer 12 hidden_states  [B, 1025, 1024] ──┤
    ├── Layer 18 hidden_states  [B, 1025, 1024] ──┤
    └── Layer 23 hidden_states  [B, 1025, 1024] ──┤
                                                    ↓
              ┌─────────────────────────────────────┐
              │         DeepStack 融合               │
              │                                     │
              │  去 CLS → [B, 1024, 1024]           │
              │  ↓                                  │
              │  每层独立 LayerNorm(1024)            │
              │  ↓                                  │
              │  可学习权重加权求和                   │
              │  lw = softmax(init=1/n)             │
              │  ↓                                  │
              │  fused = Σ(lw_i × LN_i(hs_i))       │
              │  ↓                                  │
              │  输出 [B, 1024, 1024]                │
              └─────────────────────────────────────┘
                                ↓
                     mm_projector (MLP)
                     1024 → LLM hidden_size
                                ↓
                     LLM (Qwen2/Qwen3)
                   image embeddings 与 text tokens
                   在 embedding 层交错拼接
```

**关键参数**:

| DINOv2-large | Qwen3-VL-2B 原版 | 本项目默认 |
|-------------|------------------|-----------|
| 总层数 | 24 | 24 |
| 选中层 | [5, 11, 17] | [6, 12, 18, 23] |
| 选中层数 | 3 | 4 |
| hidden_size | 1024 | 1024 |

**实现** (`llava/model/multimodal_encoder/deepstack.py`):
1. 从 DINOv2 指定层提取 hidden_states（`output_hidden_states=True`）
2. 去除 CLS token，保留 1024 个 patch token
3. 每层特征通过独立的 `nn.LayerNorm(1024)` 归一化
4. 可学习的层权重 `layer_weights` (初始化为均匀分布 1/n) 加权求和融合
5. 可选输出投影改变维度（当前不使用，由 mm_projector 承担维度映射）

**代码注入路径** (从 ViT → DeepStack → LLM 完整链路):

```
1. 选层 + 去 CLS
   llava/model/multimodal_encoder/dinov2_encoder.py:68-78 (feature_select)
   hidden_states[6] → [B,1025,1024] → 去CLS → [B,1024,1024]
   hidden_states[12] → [B,1025,1024] → 去CLS → [B,1024,1024]
   hidden_states[18] → [B,1025,1024] → 去CLS → [B,1024,1024]
   hidden_states[23] → [B,1025,1024] → 去CLS → [B,1024,1024]

2. DeepStack 融合
   llava/model/multimodal_encoder/deepstack.py:34-50 (forward)
   每层 → LayerNorm → ×层权重 → 4层求和 → [B,1024,1024]

3. 投影
   llava/model/llava_arch.py:165-168 (encode_images)
   [B,1024,1024] → mm_projector (MLP) → [B,1024,LLM_hidden]

4. 注入 LLM
   llava/model/llava_arch.py:288-296 (prepare_inputs_labels_for_multimodal)
   text_embeds[0] + image_embeds + text_embeds[1] + ... → LLM forward
```

**使用方式**:
```bash
--deepstack_visual_indexes 6 12 18 23  # 选择层索引 (0-indexed)
```

当 `deepstack_visual_indexes` 为 None 时，回退到原始单层模式（使用 `select_layer`），保证向后兼容。

---

## Qwen3 适配

新增 `LlavaQwen3ForCausalLM` (`llava/model/language_model/llava_qwen3.py`):
- 继承自 `Qwen3ForCausalLM`（如 transformers 不支持则回退 `Qwen2ForCausalLM`）
- model_type = `"llava_qwen3"`
- 对话模板: `conv_qwen_3_Dinov2_huawei` (version="qwen_v3")
- 训练脚本自动检测: 若 `model_name_or_path` 包含 "qwen3" 则使用 Qwen3 类

### Qwen3-VL 关键技术（仅记录，本次未全部复现）

| 技术 | 说明 | 状态 |
|------|------|------|
| **DeepStack** | 多层 ViT 特征融合 | ✅ 已复现 |
| **Interleaved-MRoPE** | 时间/宽度/高度三维频率分配位置编码 | ❌ 视频相关，暂不需要 |
| **Text-Timestamp Alignment** | 显式文本时间戳对齐 | ❌ 视频相关，暂不需要 |
| **Qwen3 LLM** | Qwen3 语言模型作为解码器 | ✅ 已适配 |

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
| ★ `--deepstack_visual_indexes` | List[int] | None | DeepStack 选择层，如 `6 12 18 23` |

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

## 数据格式

JSONL 格式，每行一个样本:
```json
{
  "id": "sample_001",
  "image": "img/xxx.png",
  "conversations": [
    {"from": "human", "value": "<image>\nPredict the complete road map..."},
    {"from": "gpt", "value": "[{\"points\":[[x,y],...],\"category\":\"CenterLine\"},...]"}
  ]
}
```

---

## 对话模板

| 模板名称 | 用途 |
|---------|------|
| `qwen_2` | 通用 Qwen2 对话 |
| `qwen_2_centerline_coord` | BEV 道路中心线提取 (Qwen2) |
| `conv_qwen_2_Dinov2_huawei` | LiDAR BEV 道路重建 (Qwen2 + DINOv2) |
| `conv_qwen_3_Dinov2_huawei` | LiDAR BEV 道路重建 (Qwen3 + DINOv2) |

---

## 环境依赖

```toml
[project]
name = "llava"
version = "1.2.2.post1"
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.51.0",
    "tokenizers>=0.21",
    "accelerate>=0.20.0",
    "peft>=0.10.0",
    "deepspeed==0.13.1",
    "timm>=0.9.0",
    "einops>=0.6",
    "Pillow",
    "packaging",
    "ninja",
    "wandb",
    "huggingface-hub>=0.25.1",
]
```

> **注意**: 本地 `fastvlm` 环境已升级至 transformers 5.7.0, tokenizers 0.22.2。NPU 云平台脚本中 Qwen2 需 `transformers>=4.48.3`，Qwen3-VL 需 `transformers>=4.51.0`。

---

## Git 工作流

- **主分支**: `main` (仅生产就绪代码)
- **工作分支**: `qwen3vl_Dinov2` (所有开发在此分支进行)
- **规则**: 不合并到 main，在 `qwen3vl_Dinov2` 上开发并推送远程

```bash
git checkout qwen3vl_Dinov2   # 切换到工作分支
# ... 开发 ...
git add <files>
git commit -m "描述改动"
git push origin qwen3vl_Dinov2
```

---

## Qwen2 ↔ Qwen3 切换指南

### 已自动化处理（无需手动改）

| 组件 | 处理方式 |
|------|---------|
| **训练脚本** `train_qwen.py` | 检测 `model_name_or_path` 含 `qwen3`/`qwen-3` → 自动用 `LlavaQwen3ForCausalLM` |
| **模型加载** `builder.py` | 检测 config `model_type` 含 `qwen3` → 自动用对应类 |
| **Qwen3-VL LLM 提取** | 自动检测 Qwen3-VL checkpoint，提取纯 LLM 权重用于训练，保存完整 LLaVA 模型 |
| **Tokenizer** | Qwen3 和 Qwen2 共用 `Qwen2Tokenizer` 类，但各自加载自己的 tokenizer 文件 (vocab/tokenizer.json)，已自动处理 |
| **对话模板** | `conv_qwen_3_Dinov2_huawei` 与 `conv_qwen_2_Dinov2_huawei` 格式相同 |
| **位置编码** | 两者都用 RoPE，LLaVA 架构下为 1D sequential position_ids |
| **Image Token** | `<image>` token 统一使用 IMAGE_TOKEN_INDEX=-200 |

### 需要手动改（仅 2 处）

| 位置 | 改什么 |
|------|--------|
| `llava/conversation.py` `conv_qwen_3_Dinov2_huawei` | **系统提示词** — 按任务需求修改 system prompt |
| 训练脚本顶部 | `MODEL_NAME_OR_PATH` 指向 Qwen3 checkpoint，`VERSION=conv_qwen_3_Dinov2_huawei` |

### 已验证的兼容性矩阵

| 测试 | 状态 | 说明 |
|------|------|------|
| Qwen2.5-1.5B + DINOv2 + DeepStack | ✅ | Loss 0.76→0.58, GPU 0 |
| Qwen2.5-1.5B + DINOv2 + DeepStack + ZeRO-3 | ✅ | Loss 2.23→1.06, 1 GPU |
| Qwen3-VL-2B LLM + DINOv2 + DeepStack | ✅ | Loss 0.69→0.54, GPU 0, auto-extract |
| Qwen2.5-1.5B + DINOv2 + DeepStack (推理) | ✅ | 加载完整模型，正常生成输出 |
| Qwen3-VL-2B LLM + DINOv2 + DeepStack (推理) | ✅ | 加载完整模型，正常生成输出 |
| Qwen3-VL-8B LLM (待测试) | ⏳ | 需下载 Qwen3-VL-8B-Instruct |

### Qwen3-VL LLM 自动提取机制

项目会自动检测 `model_name_or_path` 是否为 Qwen3-VL checkpoint (`model_type: "qwen3_vl"`)，并在首次使用时自动提取 LLM 权重：

```
Qwen3-VL checkpoint/            →  缓存目录 .qwen3_llm_extracted_<hash>/
  model.safetensors                 config.json (model_type="qwen3")
  ├── model.visual.* (315 keys)     model.safetensors (model.*, 310 keys)
  └── model.language_model.*        tokenizer 文件
      → model.* (310 keys)

自动提取流程:
  1. 读取 config.json 的 text_config
  2. 重命名 model.language_model.* → model.*
  3. 丢弃 model.visual.*
  4. 保存到缓存目录，后续直接复用
  5. 训练保存完整 LLaVA 模型，推理无需再提取
```

实现文件: `llava/model/qwen3vl_extractor.py`

### 升级 transformers 以启用原生 Qwen3

```bash
pip install "transformers>=4.51.0"
```

升级后 `LlavaQwen3ForCausalLM` 将使用真正的 `Qwen3ForCausalLM`；不升级则自动回退 `Qwen2ForCausalLM`，功能不变。
