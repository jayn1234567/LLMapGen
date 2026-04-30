# MLLM_project — 工作文档 (branch: qwen3vl_deepstack_injection_dinov2)

## 项目概述

基于 LLaVA 架构的 BEV 道路几何理解多模态大模型（VLM），使用 DINOv2 作为视觉编码器 + Qwen2/Qwen3 作为语言模型，完成 BEV 图像的道路中心线重建。

本分支实现了 **Qwen3-VL DeepStack 的真实架构**：ViT 不同层的特征通过残差注入到 LLM 的前 N 层。

- **分支**: `qwen3vl_deepstack_injection_dinov2`（从 `qwen3vl_Dinov2` 分出）
- **环境**: conda `fastvlm`
- **依赖**: transformers>=4.51.0 (transformers 5.7.0 本地)

---

## DeepStack 架构（真实实现）

### Qwen3-VL 原版 DeepStack

Qwen3-VL 的 DeepStack **不是**将 ViT 多层特征融合后统一注入，而是：

```
ViT Layer 23 → main merger → pooler_output → 替换 <image> token (LLM 输入层)
ViT Layer 18 → Merger[0]  → 残差加到 LLM Layer 0
ViT Layer 12 → Merger[1]  → 残差加到 LLM Layer 1
ViT Layer 6  → Merger[2]  → 残差加到 LLM Layer 2
```

**核心机制**: 浅层 ViT 特征 → 浅层 LLM，深层 ViT 特征 → 深层 LLM，通过残差加法注入。

### 我们的实现

使用 `register_forward_pre_hook` 在 LLM decoder layer 前注入深栈特征：

```python
# 在标准 model.forward() 运行前注册 hook
for i in range(n_ds):  # n_ds = 4 for layers [6,12,18,23]
    self.model.layers[i].register_forward_pre_hook(make_hook(i))

# 标准 forward 处理所有 RoPE/causal mask 内部逻辑
outputs = self.model(input_ids=None, inputs_embeds=..., attention_mask=..., position_ids=...)

# 注入逻辑 (残差加法):
def hook(module, args):
    hs = args[0]  # hidden_states [B, S, D]
    ds_feat = deepstack_visual_embeds[layer_idx]  # [B, N_patches, D]
    scattered = scatter_to_positions(ds_feat, visual_pos_mask)
    hs = hs + scattered  # ← 残差注入
    return (hs,) + rest
```

### Merger 模块

每个 ViT 层有独立的 `DeepStackMerger` (不共享权重):
```
LayerNorm(vit_dim) → Linear(vit_dim → llm_dim) → GELU → Linear(llm_dim → llm_dim)
```

### 代码文件

| 文件 | 角色 |
|------|------|
| `llava/model/multimodal_encoder/deepstack.py` | `DeepStackMerger` 类 |
| `llava/model/multimodal_encoder/dinov2_encoder.py` | 返回 `(main_features, deepstack_list)` |
| `llava/model/llava_arch.py` | `encode_images` 双返回值; `prepare_inputs_labels_for_multimodal` 传递 deepstack |
| `llava/model/language_model/llava_qwen.py` | `_deepstack_forward()` via hooks |
| `llava/model/language_model/llava_qwen3.py` | 同上 |
| `llava/model/qwen3vl_extractor.py` | Qwen3-VL checkpoint → 纯 LLM 权重提取 |

---

## 训练脚本

| 脚本 | 说明 |
|------|------|
| `train_dinov2_qwen2-1.5b.sh` | GPU 单卡: Qwen2-1.5B + DINOv2 + DeepStack |
| `train_dinov2_qwen3-8b.sh` | GPU 多卡: Qwen3-8B + DINOv2 + DeepStack + ZeRO-3 |
| `train_dinov2_qwen3vl-2b.sh` | GPU 单卡: Qwen3-VL-2B LLM + DINOv2 + DeepStack |
| `train_dinov2_qwen3vl-8b.sh` | GPU 多卡: Qwen3-VL-8B LLM + DINOv2 + DeepStack + ZeRO-3 |
| `train_dinov2_qwen2-1.5b_npu.sh` | NPU: Qwen2-1.5B + train→infer |
| `train_dinov2_qwen3vl-8b_npu.sh` | NPU: Qwen3-VL-8B LLM + train→infer |

---

## 验证矩阵

| 测试 | 状态 | 说明 |
|------|------|------|
| Qwen2.5-1.5B + DINOv2 + DeepStack (训练) | ✅ | Loss 0.75→0.57 |
| Qwen3-VL-2B + DINOv2 + DeepStack (训练) | ✅ | Loss 0.97→0.66 |
| 推理 (加载完整模型) | ✅ | generate 正常输出 |
