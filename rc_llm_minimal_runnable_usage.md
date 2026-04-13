# RC LLM 最小可运行版使用说明

最近更新：`2026-04-13`（北京时间）

## 1. 这份最小版保留了什么

这份仓库只保留我们当前真正使用的 RC LLM 主线：

1. `Stage A`：纯视觉道路结构编码器训练
2. `Stage 1`：DINOv2 -> Qwen3 的粗语义对齐
3. `Stage 2`：`Scene + GridStates(8x8)` 的细粒度 caption 对齐
4. `Stage 3`：基于原生 JSON 输出的中心线 `SFT v1 + LoRA`

这份最小版**不再保留**：

- 旧的 query-resampler 中心线分支
- 旧的离散中心线 token 输出分支
- caption 规则调参历史脚本
- scene autocorrect / review / export 历史工具链
- 上游 train root 的完整构建脚本
- smoke 专用 launcher

默认前提是：我们在超算上的 canonical 数据根和基础模型权重已经准备好。

## 2. 目录分层

为了让主线更清楚，这一版把入口文件分成两层：

### 2.1 `train_required/`

这一层是**训练必需**。只保留从 Stage A 跑到 Stage 3 所需的最小入口。

- `train_required/launchers/`
  - `stagea_rc_structure_seg_dinov2_structure_multiclass_thinmask_pad518_4gpu_20260411.sbatch`
  - `stage1_rc_dinov2_clip_align_4gpu_clean_v1_20260412.sbatch`
  - `stage2_rc_dinov2_caption_grid8_stage1init_4gpu_20260412.sbatch`
  - `stage3_rc_dinov2_centerline_json_sft_lora_8gpu_20260413.sbatch`
- `train_required/scripts/`
  - `train_rc_structure_seg.py`
  - `train_qwen3_rc_dinov2_clip_align.py`
  - `train_qwen3_rc_dinov2_caption_llava.py`
  - `train_qwen3_rc_dinov2_centerline_json_sft.py`

### 2.2 `eval_optional/`

这一层是**评估可选**。删掉不会影响主训练链路，但建议保留，方便做质量检查。

- `eval_optional/launchers/`
  - `render_rc_structure_seg_predictions_dinov2_structure_multiclass_thinmask_latest_val30_1gpu_20260411.sbatch`
  - `stage1_rc_dinov2_clip_retrieval_eval_val_1gpu_20260412.sbatch`
  - `stage2_rc_dinov2_caption_structured_eval_quick100_1gpu_20260413.sbatch`
  - `stage3_rc_dinov2_centerline_json_sft_latest100_viz_1gpu_20260413.sbatch`
- `eval_optional/scripts/`
  - `render_rc_structure_seg_predictions.py`
  - `eval_qwen3_rc_dinov2_clip_retrieval.py`
  - `eval_qwen3_rc_dinov2_caption_structured.py`
  - `render_stage2_structured_eval_viz.py`
  - `predict_qwen3_rc_dinov2_centerline_json_sft.py`
  - `render_eval_predictions_jsonl.py`

### 2.3 `unimapgen/`

这一层保留共享模型与数据逻辑，训练和评估共用：

- `unimapgen/data/`
- `unimapgen/models/`
- `unimapgen/rc_llm_runtime.py`
- `unimapgen/utils.py`

其中 `unimapgen/rc_llm_runtime.py` 是这次新抽出来的公共运行时模块，用来统一：

- 随机种子设置
- `meta_*.jsonl` 自动解析
- 视觉编码 checkpoint 参数回读
- 视觉 token 数推断
- `TrainingArguments` 兼容构建

### 2.4 补充文档与样例

除了代码本体，这一版还额外补了两组辅助材料：

- `dataset_examples/`
  - 放每类核心数据集的样例 JSON 和字段说明
- `rc_llm_scripts_overview.md`
  - 按脚本逐个解释作用、输入和输出

## 3. 需要预先存在的数据根

当前主线默认使用下面三个超算数据根：

中心线与视觉结构主根：

- `/file_storage01/home/mingli/data/outputs/rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408`

Stage 1 媒体根、Stage 2 训练根：

- `/file_storage01/home/mingli/data/outputs/rc_caption_short_trainroot_dropIntConnector_grid8_full_sparse5_connectorveto_v3_20260411`

Stage 1 clean semantic root：

- `/file_storage01/home/mingli/data/outputs/rc_semantic_align_scene_sides_autocorrect_clean_v1_20260412`

这份最小版**不包含**这些数据根的历史构建流程。

## 4. 推荐运行顺序

### 4.1 Stage A：视觉编码器

训练 launcher：

- `train_required/launchers/stagea_rc_structure_seg_dinov2_structure_multiclass_thinmask_pad518_4gpu_20260411.sbatch`

预期产出：

- `/file_storage01/home/mingli/data/outputs/rc_structure_seg_dinov2_structure_multiclass3_thinmask_pad518_4gpu_20260411/latest.pt`

可选可视化：

- `eval_optional/launchers/render_rc_structure_seg_predictions_dinov2_structure_multiclass_thinmask_latest_val30_1gpu_20260411.sbatch`

### 4.2 Stage 1：粗对齐

训练 launcher：

- `train_required/launchers/stage1_rc_dinov2_clip_align_4gpu_clean_v1_20260412.sbatch`

主要输入：

- 视觉编码器权重：
  - `rc_structure_seg_dinov2_structure_multiclass3_thinmask_pad518_4gpu_20260411/latest.pt`
- 训练根：
  - `rc_semantic_align_scene_sides_autocorrect_clean_v1_20260412`
- 媒体根：
  - `rc_caption_short_trainroot_dropIntConnector_grid8_full_sparse5_connectorveto_v3_20260411`

主要产出：

- `/file_storage01/home/mingli/data/outputs/stage1_rc_dinov2_clip_align_clean_v1_4gpu_20260412/rc_dinov2_clip_align_modules.pt`

可选检索评估：

- `eval_optional/launchers/stage1_rc_dinov2_clip_retrieval_eval_val_1gpu_20260412.sbatch`

### 4.3 Stage 2：细对齐

训练 launcher：

- `train_required/launchers/stage2_rc_dinov2_caption_grid8_stage1init_4gpu_20260412.sbatch`

主要输入：

- Stage 1 bridge 权重：
  - `stage1_rc_dinov2_clip_align_clean_v1_4gpu_20260412/rc_dinov2_clip_align_modules.pt`
- caption root：
  - `rc_caption_short_trainroot_dropIntConnector_grid8_full_sparse5_connectorveto_v3_20260411`

主要产出：

- `/file_storage01/home/mingli/data/outputs/stage2_rc_dinov2_caption_grid8_stage1init_1epoch_fixsave_4gpu_20260412/rc_dinov2_caption_modules.pt`

可选结构化评估：

- `eval_optional/launchers/stage2_rc_dinov2_caption_structured_eval_quick100_1gpu_20260413.sbatch`

可选可视化脚本：

- `eval_optional/scripts/render_stage2_structured_eval_viz.py`

### 4.4 Stage 3：中心线 SFT v1

训练 launcher：

- `train_required/launchers/stage3_rc_dinov2_centerline_json_sft_lora_8gpu_20260413.sbatch`

主要输入：

- Stage 2 bridge 权重：
  - `stage2_rc_dinov2_caption_grid8_stage1init_1epoch_fixsave_4gpu_20260412/rc_dinov2_caption_modules.pt`
- 中心线 train root：
  - `rc_perlog_offset_trainroot_resample24_dropIntConnector_20260408`

可选推理与可视化：

- `eval_optional/launchers/stage3_rc_dinov2_centerline_json_sft_latest100_viz_1gpu_20260413.sbatch`

## 5. 当前主线结构约束

### 5.1 Stage 2

当前认可的 Stage 2 规则：

- 冻结 `DINOv2`
- 冻结 `Qwen` 主体
- 训练 bridge：
  - `visual_norm`
  - `visual_projector`
  - `geometric_position_mlp`
  - `token_alignment`
- 只训练：
  - `<vis_start>`
  - `<vis_end>`
- 不训练 `<vis_patch>`
- 不额外增加新的视觉注入模块

### 5.2 Stage 3

当前认可的 Stage 3 规则：

- 冻结 `DINOv2`
- 保持对齐 bridge 可训练
- 对 `Qwen` 开 `LoRA`
- 直接输出原生 JSON
- 不使用自定义中心线 token
- 不使用连续坐标回归头

## 6. 当前 SFT v1 提示词

System Prompt：

```text
You are an expert road-centerline reconstruction assistant for black-background BEV road-structure images.

VISIBLE SEMANTICS:
The visible road-structure classes are lane_boundary, lane_divider, and background.
The image does not show centerlines directly.

TASK DEFINITION:
Your task is to infer the unseen road centerlines strictly from the visible road structure.
1. A centerline is the geometric middle path of one valid drivable corridor.
2. Do not trace lane_boundary or lane_divider themselves.
3. Keep different lanes, branches, and intersecting paths as separate continuous polylines.
4. If a centerline reaches the patch border, terminate it at the visible border.
5. Predict all valid centerlines implied by the visible road structure in the current patch only.

OUTPUT CONSTRAINTS:
1. Return ONLY valid JSON.
2. Do NOT wrap the JSON in markdown fences.
3. Do NOT output explanations or extra text.
4. Use the patch-local coordinate system.
5. All x and y coordinates must be integers between 0 and 512 inclusive.
6. Strictly use this JSON schema:
{"lines":[]}
or
{"lines":[{"points":[[x1,y1],[x2,y2]]}]}
```

User Prompt：

```text
This is a black-background BEV road-structure image.
Predict the road centerlines for this patch from the visible lane_boundary and lane_divider structure.
Return only the raw JSON object.
```



