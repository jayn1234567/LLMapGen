# RC LLM 脚本作用总览

这份文档介绍最小可运行分支里保留下来的主要入口与核心模块，分成三层来看：
1. `train_required`：正式训练必需入口
2. `eval_optional`：评估、可视化与抽查入口
3. `unimapgen`：训练和评估共用的核心数据/模型模块

## 1. 训练必需脚本

### 1. `train_required/scripts/train_rc_structure_seg.py`

作用：

- 训练 Stage A 的纯视觉道路结构分割模型
- 支持 `binary / multiclass / structure_multiclass / dash_solid` 四种监督模式

主要输入：

- `train.jsonl / meta_train.jsonl`
- `val.jsonl / meta_val.jsonl`
- RC patch 图像

主要输出：

- `latest.pt`
- `best.pt`
- `metrics.jsonl`

### 2. `train_required/scripts/train_qwen3_rc_dinov2_clip_align.py`

作用：

- 训练 Stage 1 粗语义对齐
- 让 `DINOv2` 图像 embedding 和 `Qwen3` 文本 embedding 在语义层面靠近

主要输入：

- Stage A 视觉编码器 checkpoint
- clean semantic align root

主要输出：

- `rc_dinov2_clip_align_modules.pt`
- `args.json`

### 3. `train_required/scripts/train_qwen3_rc_dinov2_caption_llava.py`

作用：

- 训练 Stage 2 细粒度 caption 对齐
- 使用 `Scene + GridStates(8x8)` 作为监督
- 直接把视觉 embedding 替换到 `<vis_patch>` 槽位

主要输入：

- Stage A 视觉编码器 checkpoint
- Stage 1 bridge modules
- caption short root

主要输出：

- `rc_dinov2_caption_modules.pt`
- `args.json`

### 4. `train_required/scripts/train_qwen3_rc_dinov2_centerline_json_sft.py`

作用：

- 训练 Stage 3 中心线 `SFT v1`
- 直接生成原生 JSON
- 同时训练 bridge 和 `Qwen LoRA`

主要输入：

- Stage A 视觉编码器 checkpoint
- Stage 2 bridge modules
- centerline root

主要输出：

- checkpoint 目录
- `args.json`
- `rc_dinov2_centerline_json_modules.pt`

## 2. 评估可选脚本

### 1. `eval_optional/scripts/render_rc_structure_seg_predictions.py`

作用：

- 渲染 Stage A 分割结果
- 输出输入图、GT、Pred、Overlay 的 panel

适合用来做：

- 视觉 sanity check
- 不同监督模式的误差对比

### 2. `eval_optional/scripts/eval_qwen3_rc_dinov2_clip_retrieval.py`

作用：

- 做 Stage 1 的离线 retrieval eval
- 计算图到文 / 文到图的 recall、margin、top1 scene / side 命中率

### 3. `eval_optional/scripts/eval_qwen3_rc_dinov2_caption_structured.py`

作用：

- 做 Stage 2 的结构化离线评估
- 把生成文本 parse 回 `Scene + GridStates`
- 输出 `scene_acc / grid_cell_acc / macro_f1 / exact_match`

### 4. `eval_optional/scripts/render_stage2_structured_eval_viz.py`

作用：

- 把 Stage 2 的 `Input | GT State | Pred State` 渲染成三栏图
- 错格子会用红框标出来

### 5. `eval_optional/scripts/predict_qwen3_rc_dinov2_centerline_json_sft.py`

作用：

- 用 Stage 3 checkpoint 做 JSON 推理
- 解析输出 JSON
- 汇总 `parse_ok_rate` 和线条数量统计

### 6. `eval_optional/scripts/render_eval_predictions_jsonl.py`

作用：

- 把 Stage 3 的 `predictions.jsonl` 渲染成几何对比图
- 当前采用 `GT | Pred` 双栏展示

## 3. 核心模块

### 1. `unimapgen/rc_llm_runtime.py`

作用：
- 统一管理随机种子、`meta_jsonl` 回推、视觉 checkpoint 参数回读、训练参数兼容构建

### 2. `unimapgen/data/rc_structure_seg_dataset.py`

作用：
- Stage A 的数据读取入口
- 负责图像读取、mask 回退逻辑、训练增强和 batch 拼装

### 3. `unimapgen/data/rc_semantic_align_dataset.py`

作用：
- Stage 1 的数据读取入口
- 负责把 scene / visible sides / semantic text 收拢成统一监督目标

### 4. `unimapgen/data/rc_caption_short_dataset.py`

作用：
- Stage 2 的数据读取入口
- 负责 `Scene + GridStates(8x8)` caption 的构造与标签 mask

### 5. `unimapgen/data/rc_centerline_json_sft_dataset.py`

作用：
- Stage 3 的数据读取入口
- 负责原始 JSON 规整、视觉占位 prompt 构造和 assistant-only loss mask

### 6. `unimapgen/models/rc_structure_seg.py`

作用：
- Stage A 结构分割模型
- 封装视觉编码器和轻量解码头

### 7. `unimapgen/models/qwen3_rc_dinov2_clip_align.py`

作用：
- Stage 1 粗对齐模型
- 定义共享视觉桥和 grouped CLIP-style contrastive loss

### 8. `unimapgen/models/qwen3_rc_dinov2_caption_llava.py`

作用：
- Stage 2 细对齐模型
- 负责把 bridge 输出直接替换到 `<vis_patch>` 槽位

### 9. `unimapgen/models/qwen3_rc_dinov2_centerline_json_sft.py`

作用：
- Stage 3 JSON SFT 模型
- 在 Stage 2 bridge 基础上继续训练，并给 Qwen 打 LoRA

### 10. `unimapgen/models/encoders/satellite_encoder.py`

作用：
- 统一封装视觉编码器加载逻辑
- 当前主线主要服务 DINOv2，也兼容少量保留的 fallback 路径

## 4. 推荐阅读顺序

如果是第一次接手这套代码，建议按下面顺序看：

1. 先看 [rc_llm_minimal_runnable_usage.md](./rc_llm_minimal_runnable_usage.md)
2. 再看 `train_required/launchers/`
3. 再看 `train_required/scripts/`
4. 再看 `eval_optional/scripts/`
5. 最后看 `dataset_examples/`

## 5. 哪些脚本最关键

如果只挑最关键的 4 个入口，我建议优先看：

1. `train_required/scripts/train_rc_structure_seg.py`
2. `train_required/scripts/train_qwen3_rc_dinov2_clip_align.py`
3. `train_required/scripts/train_qwen3_rc_dinov2_caption_llava.py`
4. `train_required/scripts/train_qwen3_rc_dinov2_centerline_json_sft.py`

这 4 个脚本基本就串起了整条当前主线。
