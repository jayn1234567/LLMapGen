# Python 工具脚本手册

本目录包含数据制作、检查、难度分析、推理、指标、可视化和 checkpoint 管理工具。所有命令应在仓库根目录执行：

```bash
python scripts/tools/<tool>.py --help
```

## 1. 推理、指标与可视化

| 脚本 | 作用 | 关键参数 |
|---|---|---|
| `infer_centerline_checkpoint.py` | Stage A checkpoint 推理引擎；支持 LoRA、全参和 HF 分片权重 | `--checkpoint-dir`, `--test-json`, `--image-folder`, `--device`, `--max-new-tokens`, `--map-task`, `--coord-mode` |
| `infer_centerline_state_update.py` | Stage B 顺序状态更新推理 | `--checkpoint-dir`, `--patch-json`, `--image-folder`, `--vision_tower`, `--include-intersections` 及 trace 参数 |
| `centerline_eval_metrics.py` | 从推理 summary 计算单类别中心线指标 | `--summary-json`, `--meter-per-pixel`, `--buffer-size`, `--match-threshold`, `--category` |
| `summarize_centerline_eval.py` | 计算 lane、intersection、联合指标和类型准确率 | `--summary-json`, `--output-json`, `--map-task`, `--intersection-iou-threshold` |
| `rebuild_infer_eval_from_summary.py` | 已有 `summary.json` 时重建 `eval.json` | `--summary-json`, `--output-json`, `--map-task`, `--intersection-iou-threshold` |
| `visualize_centerline.py` | 绘制 patch 级真值与预测，可同时计算指标和整图可视化 | `--input-dir`, `--image-folder`, `--output-dir`, `--map-task`, `--max-samples` |
| `visualize_state_update_global.py` | 将 Stage B patch 结果拼成全局状态更新图 | `--summary-json`, `--output`, `--background`, `--canvas-width`, `--canvas-height` |
| `split_single_pass_eval_by_difficulty.py` | 将一次推理 summary 按固定难度集拆分并分别评估 | `--summary-json`, `--split-root`, `--output-root`, `--expected-counts`, `--image-folder` |
| `map_visualization.py` | 可视化公共库，供其他工具导入，不作为独立 CLI 入口 | 无 |

Stage A 最小示例：

```bash
python scripts/tools/infer_centerline_checkpoint.py \
  --checkpoint-dir /path/to/checkpoint \
  --test-json /path/to/eval.jsonl \
  --image-folder /path/to/dataset \
  --device npu \
  --map-task lane_intersection \
  --output-dir /path/to/output
```

## 2. 难度分析与结构检查

| 脚本 | 作用 | 关键参数 |
|---|---|---|
| `tag_hard_map_samples.py` | 按几何复杂度标注 easy/medium/hard/very_hard，并抽样可视化 | `--dataset-root`, `--phase`, `--split`, `--output-dir`, `--visualize-per-difficulty`, `--seed` |
| `build_difficulty_eval_splits.py` | 从 JSONL 构建可复用的固定难度评估集 | `--input-jsonl`, `--output-dir`, `--samples-per-difficulty-spec`, `--seed` |
| `analyze_map_structure_points.py` | 统计分叉点、转弯点、弯道和结构位置 | `--dataset-root`, `--split`, `--junction-tol`, `--turn-point-threshold`, `--visualize-top-k` |
| `audit_staged_512_difficulty.py` | 对 staged local512 记录重新分级并可视化 | `--staging-root`, `--variant`, `--split`, `--profile`, `--visualize-per-difficulty` |
| `prepare_difficulty_bucket_fill.py` | 将难度边界附近样本提升到缺口桶，生成 sidecar | `--input-jsonl`, `--target-samples`, `--difficulty-ratios`, `--fill-rules` |
| `inspect_lane_intersection_training_dataset.py` | 正式训练前检查 schema、prompt、坐标、类别和图像 | `--dataset-root`, `--expected-image-size`, `--coord-max`, `--allowed-centerline-type`, `--allowed-intersection-type` |
| `validate_visualize_rc_dataset_v2.py` | 验证 Dataset V2 数量、难度比例、路口比例和图像可解码性 | `--dataset-root`, `--variant`, `--expected-train-samples`, `--difficulty-ratios`, `--expected-intersection-ratio` |
| `verify_dataset_v3_eval_sources.py` | 检查多个 Dataset V3 视图的 eval/test 是否来自同一批原图 | `--dataset-root`, `--output` |
| `analyze_tokenization_mismatch.py` | 检查 prompt/target 经 tokenizer 后的长度和模板偏差 | `--model-path`, `--data-path`, `--image-folder`, `--limit`, `--show-examples` |
| `inspect_caprl_v9_best_assets.py` | 检查 CapRL 模型目录与 v9 数据样本 | `--caprl-obs-path`, `--dataset-obs-path`, `--work-dir`, `--sample-lines` |

难度分级示例：

```bash
python scripts/tools/tag_hard_map_samples.py \
  --dataset-root /path/to/dataset \
  --phase phase_a \
  --split train \
  --output-dir /path/to/difficulty_report \
  --visualize-per-difficulty 50 \
  --seed 42
```

## 3. 数据 schema 转换与派生

| 脚本 | 作用 | 关键参数 |
|---|---|---|
| `convert_legacy_fixed_eval_schema.py` | 将旧固定评估集转换到当前 prompt/schema | `--reference-dir`, `--output-dir`, `--prompt-template-jsonl`, `--image-source-root`, `--materialize-images` |
| `remap_fixed_eval_to_dataset.py` | 按 patch id/位置把固定评估集映射到新数据版本 | `--reference-dir`, `--target-dataset-root`, `--allowed-target-splits`, `--ground-truth-source`, `--verify-pixels` |
| `derive_stage_b_from_phase_a.py` | 从 Phase A 数据派生含 left/top incoming hints 的 Phase B | `--dataset-root`, `--phase-a`, `--phase-b`, `--trace-spacing-px`, `--trace-point-count` |
| `derive_intersection_prompt_dataset.py` | 把当前 patch 的路口真值放入 user prompt，只监督中心线输出 | `--input-root`, `--output-root`, `--copy-mode`, `--resume` |
| `prepare_typeclean_lane_intersection_sft.py` | 清洗 type-clean 512 标签并生成语义类别字段 | `--input-root`, `--output-root`, `--phase`, `--splits`, `--overwrite` |
| `split_single_pass_eval_by_difficulty.py` | 复用已有推理结果构建分难度输出 | 见第 1 节 |

## 4. Dataset V2 / V3 构建工具

### 4.1 通用 OBS 构建

| 脚本 | 作用 | 关键参数 |
|---|---|---|
| `build_rc_dataset_v2_from_obs.py` | 下载七个原始 RC 数据源并生成 Dataset V2 | `--source-obs-root`, `--work-root`, `--views`, `--train-target-samples`, `--difficulty-ratios`, `--intersection-target-ratio` |
| `build_rc_dataset_v2_streaming_from_obs.py` | 逐源下载、暂存、删除，降低本地磁盘占用 | `--source-obs-root`, `--work-root`, `--staging-root`, `--views`, `--train-target-samples`, `--archive-workers` |
| `setup_rc_dataset_v2_windows.ps1` | Windows 上准备 Dataset V2 环境和入口参数 | PowerShell 参数见脚本顶部 |
| `build_local512v3_550k_stageab_windows.ps1` | Windows 一键构建 local512v3 550k Stage A/B | PowerShell 参数见脚本顶部 |

### 4.2 Dataset V2 Windows 视图

| 脚本 | 作用 | 关键参数 |
|---|---|---|
| `build_rc_dataset_v2_context512_windows.py` | 从 local 数据构建 Context512 ROI256 550k/100k | `--work-root`, `--source-obs-root`, `--archive-workers`, `--resume` |
| `build_rc_dataset_v2_local512_windows.py` | 构建真正的 local512 550k/200k 及给定路口 prompt 版本 | `--work-root`, `--train-stride`, `--quick-train-target-samples`, `--resume` |
| `build_rc_dataset_v2_local512v2_windows.py` | 从可复用 staging 构建均衡 local512v2 | `--work-root`, `--train-stride`, `--skip-local256`, `--resume` |
| `build_rc_dataset_v2_rawlane_256_context_windows.py` | 构建 Raw-Lane local256 与 Context512 ROI256 | `--work-root`, `--raw-root`, `--raw-lane-threshold`, `--copy-mode`, `--archive-workers` |
| `build_rc_dataset_v2_three_image_local256_stride256_all_windows.py` | 复用 clean staging 与 RawLane/Pose staging，生成不做难度配额的三图 local256 数据集；训练只保留 256 stride base 网格，保留全部非空样本并将空样本限制为最多 5%，自动打包 tar | `--clean-staging-root`, `--aux-staging-root`, `--work-root`, `--empty-ratio`, `--copy-mode`, `--resume` |
| `build_context512_roi256_three_image_ablation_windows.py` | 从已完成的三图 Context512/ROI256 数据集生成严格 512 网格不重叠 ablation、追加旋转副本，或按 stride256 空间邻居交替旋转并替换原训练行；流式处理并自动打包 | `--mode nonoverlap|rotation|neighbor_rotation`, `--input-root`, `--output-root`, `--angles`, `--neighbor-angles`, `--copy-mode`, `--resume` |

`neighbor_rotation` 是 stride-256 的邻居旋转替换实验：每个训练样本只
保留一行，按 `(x0 // 256 + y0 // 256) % 3` 选择 `0/45/135` 度；水平和
垂直相邻样本的相位不同，三张输入图和 ROI 真值同步旋转，eval/test 不旋转。
它降低相邻窗口的相同方向冗余，但不会改变 512 context 在 stride256 下的
物理重叠。输入必须是已经
完成的三图 Context512/ROI256 数据集。完整说明见
`docs/CONTEXT512_THREE_IMAGE_ABLATIONS.md`。

### 4.3 Dataset V3

| 脚本 | 作用 | 关键参数 |
|---|---|---|
| `build_rc_dataset_v3_balanced_windows.py` | 从 staging 生成多种均衡 Dataset V3 视图 | `--work-root`, `--local512-staging-root`, `--context-staging-root`, `--resume` |
| `build_rc_dataset_v3_local512_550k_stageab_windows.py` | 生成 local512v3 550k Phase A/B | `--staging-root`, `--audit-jsonl`, `--output-root`, `--skip-phase-b` |
| `build_rc_dataset_v3_local512_1000k_filled_windows.py` | 通过边界桶填充构建严格 100 万样本 | `--staging-root`, `--filled-jsonl`, `--target-samples`, `--difficulty-ratios` |
| `build_rc_dataset_v3_rotaug_800k_windows.py` | 使用 45/135 度旋转增强构建 80 万样本 | `--input-root`, `--target-samples`, `--angles`, `--difficulty-ratios` |

大规模构建默认支持 `--resume`。恢复任务时不要删除 staging、下载完成标记和候选缓存。

## 5. 私有 DINO 分割数据与模型验证

| 脚本 | 作用 | 关键参数 |
|---|---|---|
| `download_rc_lane_segmentation_obs.py` | 下载配对的 `images/` 与 `labels_lane/` | `--output-root`, `--limit`, `--only`, `--threads`, `--skip-download` |
| `verify_dinov2_vision_tower.py` | 验证导出的 DINOv2 能被 MLLM wrapper 加载 | `--vision-tower`, `--device`, `--input-size`, `--expected-tokens`, `--expected-hidden-size` |
| `verify_dinov3_vision_tower.py` | 验证 DINOv3 hidden size、层数、register token 与输出 token | `--vision-tower`, `--device`, `--input-size`, `--expected-register-tokens` |
| `resolve_latest_dinov2_vision_tower_obs.py` | 从 OBS 资产根目录找到最新成功的 DINOv2 tower | `--obs-root`, `--report`, `--allow-run-root` |
| `resolve_best_checkpoint.py` | 解析 best/eval-best/普通 checkpoint 中最新成功候选 | `--output-dir`, `--best-name`, `--allow-direct` |

## 6. 原端到端工程桥接与评估

| 脚本 | 作用 | 关键参数 |
|---|---|---|
| `prepare_rc_e2e_inference_dataset.py` | 从 `*_inter.tif` 构建端到端 patch 推理 JSONL | `--input-root`, `--output-root`, `--view-mode`, `--target-size`, `--context-size`, `--stride` |
| `build_rc_e2e_jsonl_from_original_manifest.py` | 根据原工程 crop manifest 生成模型输入 JSONL | `--manifest-json`, `--output-root`, `--prompt-profile`, `--patch-size` |
| `audit_rc_e2e_patch_parity.py` | 审计当前 patch 与原切图器的 ID/数量一致性 | `--raw-e2e-root`, `--original-input-root`, `--current-manifest`, `--strict` |
| `validate_rc_e2e_raster_alignment.py` | 检查 inter/lane TIF 的像素网格对齐 | `--input-root`, `--patch-size`, `--atol`, `--rtol` |
| `evaluate_rc_e2e_patch_metrics.py` | patch 预测对 scene-level GeoJSON 真值的指标 | `--raw-e2e-root`, `--prediction-dir`, `--meter-per-pixel`, `--ignore-lane-types` |
| `evaluate_rc_e2e_wholemap_lane_metrics.py` | 使用原端到端 lane recipe 评估拼接整图 | `--raw-e2e-root`, `--prediction-dir`, `--lane-buffer-size`, `--stitch-distance` |
| `analyze_e2e_high_low_crossing_lanes.py` | 对比最终 `Lane.geojson` 中同时进入 high/low 互斥区域的预测实例 | `--run LABEL=E2E_ROOT`, `--output-dir`, `--min-intersection-length` |

端到端工具依赖原端到端工程的数据结构。它们与通用 Stage A JSONL 评测不是同一入口。

## 7. 参数和输出约定

- 所有 JSON/JSONL 使用 UTF-8。
- `coord_mode=norm1000` 时，评估前会按 `patch_width/patch_height` 转回像素。
- `summary.json` 保存模型原始文本、解析后的几何、真值和元数据。
- `eval.json` 是派生结果，可以从 `summary.json` 重建。
- 大规模工具建议设置 `--progress-every`，并保留生成的 report JSON 便于复现。
