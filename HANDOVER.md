# MLLM_project 交接文档

## 项目简介

基于 LLaVA 架构的 BEV 道路几何理解多模态大模型。DINOv2/DINOv3 作为视觉编码器，Qwen2/Qwen3 作为语言模型，完成 BEV 图像的道路中心线重建。

---

## 分支说明

| 分支 | 说明 |
|------|------|
| `main` | 初始代码，不维护 |
| `qwen3vl_Dinov2` | Qwen2/Qwen3 + DINOv2 + DeepStack（稳定版本） |
| `qwen3vl_deepstack_injection_dinov2` | DeepStack 真实注入架构实验分支 |
| **`qwen3vl_dinov3`** | **当前主力分支**：DINOv2/DINOv3 + Qwen2/Qwen3 + DeepStack |

---

## 环境配置

```bash
conda activate fastvlm
cd /media/q/data2/jjh/project/MLLM_project
pip install -e .
```

关键依赖：`transformers>=4.51.0`, `deepspeed`, `peft`, `safetensors`, `timm`, `einops`

---

## 脚本目录

```
scripts/
├── gpu/     ← GPU 训练（本地 A6000）
├── npu/     ← NPU 训练+推理（华为 Ascend 云平台）
├── qwen2/   ← 预留
└── qwen3vl/ ← 预留
```

### GPU 训练脚本

| 脚本 | 说明 |
|------|------|
| `train_dinov2_qwen2-1.5b.sh` | DINOv2 + Qwen2-1.5B，单卡 |
| `train_dinov2_qwen3-8b.sh` | DINOv2 + Qwen3-8B，多卡 |
| `train_dinov2_qwen3vl-2b.sh` | DINOv2 + Qwen3-VL-2B LLM 自动提取 |
| `train_dinov2_qwen3vl-8b.sh` | DINOv2 + Qwen3-VL-8B LLM 自动提取，多卡 |
| `train_dinov3_qwen2-1.5b.sh` | DINOv3 + Qwen2-1.5B，单卡 |
| `train_dinov3_qwen3vl-2b.sh` | DINOv3 + Qwen3-VL-2B LLM 自动提取 |
| `train_dinov3_qwen3vl-8b.sh` | DINOv3 + Qwen3-VL-8B LLM 自动提取，多卡 |

### NPU 训练脚本

| 脚本 | 说明 |
|------|------|
| `train_dinov2_qwen2-1.5b_npu.sh` | DINOv2 + Qwen2-1.5B |
| `train_dinov2_qwen3vl-8b_npu.sh` | DINOv2 + Qwen3-VL-8B，含 freeze_llm |
| `train_dinov3_qwen2-1.5b_npu.sh` | DINOv3 + Qwen2-1.5B |
| `train_dinov3_qwen3vl-8b_npu.sh` | DINOv3 + Qwen3-VL-8B，含 freeze_llm |

每个 NPU 脚本是自包含的：下载模型/数据 → 训练 → DeepSpeed 合并 → 推理。

---

## 快速开始

### 1. 下载 DINO 权重

```bash
python -c "from modelscope import snapshot_download; snapshot_download('facebook/dinov3-vitl16-pretrain-lvd1689m', cache_dir='checkpoints')"
```

### 2. 训练（GPU 单卡）

```bash
bash scripts/gpu/train_dinov3_qwen2-1.5b.sh
```

### 3. 推理

```bash
python scripts/tools/infer_centerline_checkpoint.py \
    --checkpoint-dir outputs/xxx \
    --test-json data/test.jsonl \
    --image-folder data/images \
    --device cuda
```

---

## 核心参数

| 参数 | 说明 |
|------|------|
| `--vision_tower` | DINO 模型路径，自动检测 dinov2/dinov3 |
| `--input_image_size` | ViT 输入尺寸（DINOv2 默认 518，DINOv3 默认 224） |
| `--deepstack_visual_indexes` | DeepStack 层选择，自动从 dino_config 填充 |
| `--freeze_llm` | 冻结 LLM，仅训 ViT + Projector |
| `--unfreeze_mm_vision_tower` | 解冻 ViT（需配合 freeze_llm 使用） |
| `--mm_projector_lr` | Projector 独立学习率 |
| `--mm_vision_tower_lr` | ViT 独立学习率 |

---

## 常用操作

### 训练 Qwen3-VL LLM + DINOv3

模型路径含 "Qwen3-VL" 时自动提取 LLM 权重，无需手动操作。

### 仅训 ViT + Projector（LLM 冻结）

```bash
--freeze_llm True \
--unfreeze_mm_vision_tower True \
--mm_vision_tower_lr 5e-7 \
--learning_rate 5e-6
```

注意：LLM 冻结后，ViT 特征变化会导致 loss 先升后降，需要较低 LR+足够 step。

### DeepSpeed ZeRO-3 训练

NPU 脚本默认使用 `deepspeed_zero3_no_merge.json`，训练时不合并权重（避免 OOM），训练结束后自动运行 `zero_to_fp32.py` 将分片合并为 `model.safetensors`。

---

## 文件结构速查

| 文件 | 作用 |
|------|------|
| `llava/model/language_model/llava_qwen.py` | Qwen2 LLM 适配 |
| `llava/model/language_model/llava_qwen3.py` | Qwen3 LLM 适配 |
| `llava/model/multimodal_encoder/dinov2_encoder.py` | DINOv2 编码器 |
| `llava/model/multimodal_encoder/dinov3_encoder.py` | DINOv3 编码器 |
| `llava/model/multimodal_encoder/dino_config.py` | DINO 变体配置注册表 |
| `llava/model/multimodal_encoder/deepstack.py` | DeepStack merger 模块 |
| `llava/model/qwen3vl_extractor.py` | Qwen3-VL → LLM 权重提取 |
| `llava/model/llava_arch.py` | 多模态融合核心 |
| `llava/model/builder.py` | 模型加载入口 |
| `llava/train/train_qwen.py` | 训练主脚本 |
| `llava/train/llava_trainer.py` | Trainer 子类（分组 LR） |
| `scripts/tools/infer_centerline_checkpoint.py` | 推理引擎 |
| `configs/` | DeepSpeed 配置 |
| `AGENTS.md` | 详细工作文档 |

---

## 已知问题

1. **Qwen3 训练后 model_type 显示 llava_qwen2**：已修复（`pop("model_type")`），重新训练即可
2. **推理 meta device 错误**：`builder.py` 已修（`str(device).startswith()` 精确路由）
3. **freeze_llm 训练 loss 上升**：正常现象，ViT 需适应冻结 LLM，降低 LR 即可
4. **NPU 多卡推理跨设备错误**：推理脚本已加固 vision tower 设备同步

---

## 联系人

项目路径: `/media/q/data2/jjh/project/MLLM_project`
Git: `github.com/jiangjihua8/MLLM_project`
分支: `qwen3vl_dinov3`
