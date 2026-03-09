# UniMapGen Agent Handoff Update（2026-03-08）

## 1. 这份更新文档解决什么问题

这份文档用于承接 [agent_handoff_20260307.md](/mnt/data/project/jn/UniMapGen/docs/agent_handoff_20260307.md) 之后的新进展。

重点记录：

- 2026-03-07 到 2026-03-08 新完成的复现工作
- 当前 partial AV2-aligned OpenSatMap 主线的真实结论
- 已经落地的脚本、配置、数据快照、评估工具
- 后续 agent 继续复现时最应该接着做什么

配套流程图文档：

- [current_model_flowchart_20260308.md](/mnt/data/project/jn/UniMapGen/docs/current_model_flowchart_20260308.md)
- [paper_aligned_model_flowchart_20260308.md](/mnt/data/project/jn/UniMapGen/docs/paper_aligned_model_flowchart_20260308.md)

## 2. 当前总判断

当前仓库已经从“工程 smoke 可跑”推进到“partial 对齐卫星数据上的正式训练闭环已跑通”。

更具体地说：

- partial 数据快照已构建完成，可直接给当前 Qwen 主线训练
- no-state baseline 正式训练已跑完
- no-state baseline 正式评估闭环已跑完
- state 正式训练已跑完
- quick state-scan 评估和 official metrics 已跑通
- line type、official metrics、paper-style patch augmentation 脚手架已经接入

但当前距离“完全复现论文”仍然有明显差距。现在最重要的结论不是“state 已经优于 baseline”，而是：

- 当前工程链路是通的
- baseline 已经学起来了
- 当前 state 配置在 partial 数据上暂时没有带来增益
- 论文式 full augmentation、更 paper-aligned 的 tokenizer / state / staged training 仍需继续补

## 3. 新完成的数据与配置工作

### 3.1 partial 快照数据集

已新增 partial 数据集构建脚本：

- [build_av2_opensatmap_partial_dataset.py](/mnt/data/project/jn/UniMapGen/scripts/build_av2_opensatmap_partial_dataset.py)
- [build_av2_opensatmap_partial_dataset.sh](/mnt/data/project/jn/UniMapGen/scripts/build_av2_opensatmap_partial_dataset.sh)

当前生成出的稳定快照目录：

- [av2_opensatmap_partial_fix](/mnt/data/project/jn/UniMapGen/data_samples/av2_opensatmap_partial_fix)

目录中已包含：

- `train/`
- `val/`
- `annotations.json`
- `splits_meta.json`
- `patch_geometry.json`
- `manifest.json`
- `summary.json`

重要事实：

- 动态源目录仍然是 `/mnt/data/project/jn/satellite_tools/av2_opensatmap_crops_paper896_fix`
- 裁剪脚本仍可能继续往里面追加样本
- 训练必须基于快照目录，而不是直接读取动态源目录

### 3.2 partial 训练配置与脚本

已新增或稳定可用的配置：

- [qwen_dinov2_map_serialization_av2_partial.yaml](/mnt/data/project/jn/UniMapGen/configs/qwen_dinov2_map_serialization_av2_partial.yaml)
- [qwen_dinov2_map_serialization_av2_partial_state.yaml](/mnt/data/project/jn/UniMapGen/configs/qwen_dinov2_map_serialization_av2_partial_state.yaml)
- [qwen_dinov2_map_serialization_av2_partial_quick.yaml](/mnt/data/project/jn/UniMapGen/configs/qwen_dinov2_map_serialization_av2_partial_quick.yaml)
- [qwen_dinov2_map_serialization_av2_partial_state_quick.yaml](/mnt/data/project/jn/UniMapGen/configs/qwen_dinov2_map_serialization_av2_partial_state_quick.yaml)

对应运行脚本：

- [run_qwen_dinov2_map_serialization_av2_partial.sh](/mnt/data/project/jn/UniMapGen/scripts/run_qwen_dinov2_map_serialization_av2_partial.sh)
- [run_qwen_dinov2_map_serialization_av2_partial_state.sh](/mnt/data/project/jn/UniMapGen/scripts/run_qwen_dinov2_map_serialization_av2_partial_state.sh)
- [run_qwen_dinov2_map_serialization_av2_partial_quick.sh](/mnt/data/project/jn/UniMapGen/scripts/run_qwen_dinov2_map_serialization_av2_partial_quick.sh)
- [run_qwen_dinov2_map_serialization_av2_partial_state_quick.sh](/mnt/data/project/jn/UniMapGen/scripts/run_qwen_dinov2_map_serialization_av2_partial_state_quick.sh)

### 3.3 line type 接入

`line_type` 已经接入当前主线，不再只是 GT JSON 里有字段但训练忽略。

关键修改位置：

- [serialization.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/serialization.py)
- [qwen_map_dataset.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_dataset.py)
- [qwen_map_pipeline.py](/mnt/data/project/jn/UniMapGen/unimapgen/qwen_map_pipeline.py)
- [state_geometry.py](/mnt/data/project/jn/UniMapGen/unimapgen/state_geometry.py)

对应配置与脚本：

- [qwen_dinov2_map_serialization_av2_partial_state_line_type.yaml](/mnt/data/project/jn/UniMapGen/configs/qwen_dinov2_map_serialization_av2_partial_state_line_type.yaml)
- [qwen_dinov2_map_serialization_av2_partial_state_line_type_quick.yaml](/mnt/data/project/jn/UniMapGen/configs/qwen_dinov2_map_serialization_av2_partial_state_line_type_quick.yaml)
- [run_qwen_dinov2_map_serialization_av2_partial_state_line_type.sh](/mnt/data/project/jn/UniMapGen/scripts/run_qwen_dinov2_map_serialization_av2_partial_state_line_type.sh)
- [run_qwen_dinov2_map_serialization_av2_partial_state_line_type_quick.sh](/mnt/data/project/jn/UniMapGen/scripts/run_qwen_dinov2_map_serialization_av2_partial_state_line_type_quick.sh)

## 4. 新完成的评估与 paper 对齐工作

### 4.1 official metrics

已新增论文口径的 OpenSatMap 官方评估入口：

- [eval_opensatmap_official.py](/mnt/data/project/jn/UniMapGen/unimapgen/eval_opensatmap_official.py)

目前可以计算：

- `mIoU`
- `APM`
- `APM50`
- `APM75`
- `APC0.9`
- `APC1.5`
- `APC3.0`
- `APC4.5`

并已接到 state 评估脚本：

- [eval_qwen_dinov2_map_serialization_av2_partial_state.sh](/mnt/data/project/jn/UniMapGen/scripts/eval_qwen_dinov2_map_serialization_av2_partial_state.sh)
- [eval_qwen_dinov2_map_serialization_av2_partial_state_quick.sh](/mnt/data/project/jn/UniMapGen/scripts/eval_qwen_dinov2_map_serialization_av2_partial_state_quick.sh)
- [eval_qwen_dinov2_map_serialization_av2_partial_state_line_type.sh](/mnt/data/project/jn/UniMapGen/scripts/eval_qwen_dinov2_map_serialization_av2_partial_state_line_type.sh)
- [eval_qwen_dinov2_map_serialization_av2_partial_state_line_type_quick.sh](/mnt/data/project/jn/UniMapGen/scripts/eval_qwen_dinov2_map_serialization_av2_partial_state_line_type_quick.sh)

quick state-scan 的 official metrics 产物已存在：

- [official_metrics_state_scan.json](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_state_quick/official_metrics_state_scan.json)

### 4.2 paper-style patch augmentation

已新增论文式 augmentation 数据构建脚本：

- [build_av2_opensatmap_paper_augmented_dataset.py](/mnt/data/project/jn/UniMapGen/scripts/build_av2_opensatmap_paper_augmented_dataset.py)
- [build_av2_opensatmap_paper_augmented_dataset.sh](/mnt/data/project/jn/UniMapGen/scripts/build_av2_opensatmap_paper_augmented_dataset.sh)
- [build_av2_opensatmap_paper_augmented_dataset_quick.sh](/mnt/data/project/jn/UniMapGen/scripts/build_av2_opensatmap_paper_augmented_dataset_quick.sh)

已支持的增强类型：

- base
- rotation
- overlap
- overlap + rotation
- inclined
- inclined + rotation

这条链现在可以真正读取原始 OpenSatMap root：

- `/mnt/data/data1/OpenSateMap`

因为该目录已经具备：

- `picuse20trainvaltest/`
- `GPS_info_all.json`
- `annotrainval20.json`

对应 stage1 配置与脚本也已补齐：

- [qwen_dinov2_map_serialization_av2_paper_stage1_aug.yaml](/mnt/data/project/jn/UniMapGen/configs/qwen_dinov2_map_serialization_av2_paper_stage1_aug.yaml)
- [qwen_dinov2_map_serialization_av2_paper_stage1_aug_quick.yaml](/mnt/data/project/jn/UniMapGen/configs/qwen_dinov2_map_serialization_av2_paper_stage1_aug_quick.yaml)
- [run_qwen_dinov2_map_serialization_av2_paper_stage1_aug.sh](/mnt/data/project/jn/UniMapGen/scripts/run_qwen_dinov2_map_serialization_av2_paper_stage1_aug.sh)
- [run_qwen_dinov2_map_serialization_av2_paper_stage1_aug_quick.sh](/mnt/data/project/jn/UniMapGen/scripts/run_qwen_dinov2_map_serialization_av2_paper_stage1_aug_quick.sh)
- [eval_qwen_dinov2_map_serialization_av2_paper_stage1_aug.sh](/mnt/data/project/jn/UniMapGen/scripts/eval_qwen_dinov2_map_serialization_av2_paper_stage1_aug.sh)
- [eval_qwen_dinov2_map_serialization_av2_paper_stage1_aug_quick.sh](/mnt/data/project/jn/UniMapGen/scripts/eval_qwen_dinov2_map_serialization_av2_paper_stage1_aug_quick.sh)
- [qwen_dinov2_map_serialization_av2_paper_stage1_aug_noline_semantic_quick.yaml](/mnt/data/project/jn/UniMapGen/configs/qwen_dinov2_map_serialization_av2_paper_stage1_aug_noline_semantic_quick.yaml)
- [run_qwen_dinov2_map_serialization_av2_paper_stage1_aug_noline_semantic_quick.sh](/mnt/data/project/jn/UniMapGen/scripts/run_qwen_dinov2_map_serialization_av2_paper_stage1_aug_noline_semantic_quick.sh)
- [eval_qwen_dinov2_map_serialization_av2_paper_stage1_aug_noline_semantic_quick.sh](/mnt/data/project/jn/UniMapGen/scripts/eval_qwen_dinov2_map_serialization_av2_paper_stage1_aug_noline_semantic_quick.sh)

本轮还新增了一套受控规模的 quick paper-aug 快照，位置在：

- [av2_opensatmap_paper_aug_partial](/mnt/data/project/jn/UniMapGen/data_samples/av2_opensatmap_paper_aug_partial)

当前这套 quick 快照的 `summary.json` 显示：

- `max_samples = 320`
- `train_base = 288`
- `train_rotation = 864`
- `train_overlap = 1527`
- `train_overlap_rotation = 4581`
- `train_inclined = 576`
- `train_inclined_rotation = 1728`
- `val_base = 32`
- `num_train_tokens = 9564`
- `num_val_tokens = 32`

这说明：

- paper-style augmentation 快照已经真正落盘，不再只是脚手架
- quick 阶段也能覆盖 rotation / overlap / inclined 全组合
- 后续可以直接基于这套快照跑 non-line_type paper-stage1 quick

路线文档：

- [paper_patch_augmentation_route_20260307.md](/mnt/data/project/jn/UniMapGen/docs/paper_patch_augmentation_route_20260307.md)

## 5. 新完成的训练工程修复

### 5.1 懒加载与初始化日志

为了解决“看起来像卡死，但其实还在初始化”的问题，已做：

- `transformers` 懒加载
- dataset 初始化日志
- tokenizer / model 加载日志
- shell 脚本开启 `PYTHONUNBUFFERED=1`

关键文件：

- [qwen_map_pipeline.py](/mnt/data/project/jn/UniMapGen/unimapgen/qwen_map_pipeline.py)
- [train_qwen_map.py](/mnt/data/project/jn/UniMapGen/unimapgen/train_qwen_map.py)
- [qwen_map_dataset.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_dataset.py)
- [activate_unimapgen_gpu_env.sh](/mnt/data/project/jn/UniMapGen/scripts/activate_unimapgen_gpu_env.sh)

### 5.2 state_lines 磁盘缓存

`build_state_lines` 已做持久化缓存，避免每次训练和评估都重新构建一遍。

实现位置：

- [qwen_map_dataset.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_dataset.py)

缓存目录形式：

- `data_samples/av2_opensatmap_partial_fix/.cache/state_lines_<split>_<signature>.json`

### 5.3 patch geometry 进程内缓存

`patch_geometry.json` 的解析已改成进程内缓存，避免同一轮训练里 train/val dataset 重复解析同一份几何文件。

实现位置：

- [state_geometry.py](/mnt/data/project/jn/UniMapGen/unimapgen/state_geometry.py)

### 5.4 epoch 间长等待问题

之前训练看起来在每个 epoch 之间“卡很久”，现在已定位并部分修复：

- 之前验证阶段无日志，容易误判为卡住
- 每个 epoch 末尾会写 `latest.pt` 和 `best.pt`，单次写盘量可达数 GB

已做的修复：

- 在 [train_qwen_map.py](/mnt/data/project/jn/UniMapGen/unimapgen/train_qwen_map.py) 中加入 validation / checkpoint 阶段日志
- 记录 `train_sec / val_sec / checkpoint_sec`
- 使用原子保存
- 主配置默认关闭 `latest.pt`，只保留 `best.pt`

### 5.5 激活脚本卡在 env 输出后的问题

原因已确认不是 `conda activate` 本身，而是后续的 `torch / CUDA` 探测。

已在 [activate_unimapgen_gpu_env.sh](/mnt/data/project/jn/UniMapGen/scripts/activate_unimapgen_gpu_env.sh) 中做：

- 先只打印 `python`
- torch/CUDA 探测加 `timeout`
- 支持 `UNIMAPGEN_SKIP_TORCH_PROBE=1`

### 5.6 `UNIMAPGEN_SKIP_TORCH_PROBE=1` 的脚本 bug 修复

本轮定位到一个真实问题：

- [activate_unimapgen_gpu_env.sh](/mnt/data/project/jn/UniMapGen/scripts/activate_unimapgen_gpu_env.sh) 会被训练 / 评估脚本用 `source` 调用
- 之前当设置 `UNIMAPGEN_SKIP_TORCH_PROBE=1` 时，脚本里使用的是 `exit 0`
- 这会直接结束外层训练脚本，而不是仅跳过 probe

现已修复为：

- 在被 `source` 时使用 `return 0`
- 仅在非 `source` 场景下 fallback 到 `exit 0`

结论：

- 之后可以安全使用 `UNIMAPGEN_SKIP_TORCH_PROBE=1`
- 这对当前单卡复现实验很重要，因为它能避免启动时被 torch/CUDA 探测拖住

## 6. 已完成的实验结果

### 6.1 partial baseline 正式训练

输出目录：

- [qwen_dinov2_map_serialization_av2_partial](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial)

最终结果见 [metrics.jsonl](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial/metrics.jsonl) 最后一行：

- `epoch=3`
- `train_loss=2.1675`
- `val_loss=2.1597`
- `val_token_acc=0.5647`

结论：

- no-state baseline 已经明显学起来
- 这条结果是后续判断 state 是否有效的正式参照线

### 6.1.1 partial baseline 正式评估闭环

`baseline` 正式版评估闭环已经成功跑完，命令是：

```bash
cd /mnt/data/project/jn/UniMapGen
bash scripts/eval_qwen_dinov2_map_serialization_av2_partial.sh
```

这次已确认三步全部跑通：

1. `eval_qwen_map`
2. `predict_qwen_map`
3. `eval_opensatmap_official`

产物位置：

- [predictions_val.json](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial/predictions_val.json)
- [official_metrics_val.json](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial/official_metrics_val.json)

本次正式评估结果：

- `val_loss = 2.159573052059952`
- `val_token_acc = 0.5648379052369077`

official metrics 结果：

- `mIoU = 0.018504829428330694`
- `APM = 8.44712218201654e-08`
- `APM50 = 7.598818839599574e-07`
- `APM75 = 0.0`
- `APC0.9 = 5.3661113473247285e-05`
- `APC1.5 = 0.0003184975751339328`
- `APC3.0 = 0.02406539412972729`
- `APC4.5 = 0.09413544401625651`

与 `state` 正式评估对比的结论：

- `baseline` 明显优于当前 `state`
- 当前退步已经不只是 token-level 指标，official metrics 也支持同样结论
- 当前最需要验证的是 `state + line_type` 是否能追回这部分差距

### 6.2 partial state quick

输出目录：

- [qwen_dinov2_map_serialization_av2_partial_state_quick](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_state_quick)

训练结果：

- `epoch=1`
- `train_loss=7.0310`
- `val_loss=6.1902`
- `val_token_acc=0.1026`

quick state-scan 已成功生成：

- [predictions_state_scan.json](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_state_quick/predictions_state_scan.json)
- [official_metrics_state_scan.json](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_state_quick/official_metrics_state_scan.json)

结论：

- 工程链路已打通
- quick 结果仍偏早期，不能据此判断 state 最终有效性

### 6.2.1 partial state + line_type quick

输出目录：

- [qwen_dinov2_map_serialization_av2_partial_state_line_type_quick](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_state_line_type_quick)

训练结果：

- `epoch=1`
- `train_loss=6.889676451683044`
- `val_loss=5.985535219311714`
- `val_token_acc=0.08521369790284046`

quick 评估命令：

```bash
cd /mnt/data/project/jn/UniMapGen
bash scripts/eval_qwen_dinov2_map_serialization_av2_partial_state_line_type_quick.sh
```

产物位置：

- [predictions_state_scan.json](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_state_line_type_quick/predictions_state_scan.json)
- [official_metrics_state_scan.json](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_state_line_type_quick/official_metrics_state_scan.json)

本次 quick 评估结果：

- `val_loss = 5.985535219311714`
- `val_token_acc = 0.08521369790284046`

official metrics 结果：

- `mIoU = 0.0`
- `APM = 0.0`
- `APM50 = 0.0`
- `APM75 = 0.0`
- `APC0.9 = 0.0`
- `APC1.5 = 0.0`
- `APC3.0 = 0.0`
- `APC4.5 = 0.0016111891420680985`

额外观察：

- 32 个样本都有非空预测，不是“完全没输出”
- 平均每个样本预测 `17` 条线，GT 平均约 `10.16` 条
- 当前预测的 `line_type` 明显塌缩，主要集中在 `solid`

当前判断：

- `state + line_type quick` 比 `state quick` 还差
- 当前更像是 `line_type` 接入后增加了学习难度，或者 token / decode 设计还没对齐
- 在直接投入 full 训练前，应该先做一轮轻量诊断，确认不是序列化或标签设计本身导致模型学偏

### 6.2.2 partial baseline + line_type quick

输出目录：

- [qwen_dinov2_map_serialization_av2_partial_line_type_quick](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_line_type_quick)

训练结果：

- `epoch=1`
- `train_loss=6.666059780865908`
- `val_loss=5.868018224835396`
- `val_token_acc=0.11016724183700558`

official metrics：

- `mIoU = 0.0`
- `APC3.0 = 0.0`
- `APC4.5 = 0.0`

预测形态：

- 32 个样本都有非空预测
- 平均每个样本预测 `17` 条线
- 预测类别全部塌缩到 `lane_line`
- 预测 line type 全部塌缩到 `solid`
- 绝大多数预测点都是 `[[0,0],[0,0]]`

结论：

- 单独引入 `line_type` 就已经足以让模型从 baseline 退化到全 0 official metrics
- 这说明问题不只在 `state`

### 6.2.3 partial baseline + line_type prompt fix quick

输出目录：

- [qwen_dinov2_map_serialization_av2_partial_line_type_promptfix_quick](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_line_type_promptfix_quick)

训练结果：

- `epoch=1`
- `train_loss=6.727317398414016`
- `val_loss=5.920347228646278`
- `val_token_acc=0.12211308733740377`

official metrics：

- `mIoU = 0.0`
- `APC3.0 = 0.0`
- `APC4.5 = 0.0`

诊断结果：

- 已给 prompt 显式补充 `line_type_instruction`
- token acc 有轻微回升，但预测形态与 `line_type quick` 基本完全一致
- 仍然是 `17` 条 `lane_line/solid` 的两点零坐标模板

结论：

- prompt 不是主因
- 当前 line_type 崩坏主要不是提示词缺失导致

### 6.2.4 partial baseline + line_type optional quick

改动：

- 将语法从“有 line_type 词表时必须输出 line_type”改成“line_type 可选”

输出目录：

- [qwen_dinov2_map_serialization_av2_partial_line_type_optional_quick](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_line_type_optional_quick)

训练结果：

- `epoch=1`
- `train_loss=6.698285462334752`
- `val_loss=5.899843782186508`
- `val_token_acc=0.13246615343774887`

official metrics：

- `mIoU = 0.0`
- `APC3.0 = 0.0`
- `APC4.5 = 0.0`

诊断结果：

- 预测 line type 变成空字符串，不再固定为 `solid`
- 但预测类别仍全部塌缩到 `lane_line`
- 平均预测线数从 `17` 变成 `19`
- 几何仍几乎全部是 `[[0,0],[0,0]]`

结论：

- 仅放宽 grammar 不足以解决问题
- 训练目标本身还存在 `None -> others` 的监督污染

### 6.2.5 partial baseline + line_type skip-none quick

改动：

- 修正 `serialize_opensatmap_lines()`，不再把空 `line_type` 强行回填成 `others`
- 确认 `curb` 训练样本现在不再携带 `line_type`

输出目录：

- [qwen_dinov2_map_serialization_av2_partial_line_type_skipnone_quick](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_line_type_skipnone_quick)

训练结果：

- `epoch=1`
- `train_loss=6.813820984214544`
- `val_loss=6.096927493810654`
- `val_token_acc=0.09010746762193442`

official metrics：

- `mIoU = 0.0`
- `APC3.0 = 0.0`
- `APC4.5 = 0.0`

预测形态：

- 32 个样本都有非空预测
- 平均每个样本预测 `19` 条线
- 预测类别仍全部塌缩到 `lane_line`
- 预测 line type 为空
- 绝大多数预测点仍是 `[[0,0],[0,0]]`

结论：

- 即使去掉 `None -> others` 的错误监督，当前 line_type 主线依然崩坏
- 现在已经有足够证据说明：当前仓库里的 `line_type` 方案不值得直接上 full
- 后续应把 `line_type` 视为阻塞分支，而不是主线增益项

### 6.2.6 partial baseline quick 对照实验

输出目录：

- [qwen_dinov2_map_serialization_av2_partial_quick](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_quick)

训练结果：

- `epoch=1`
- `train_loss=6.740380272269249`
- `val_loss=6.034810438752174`
- `val_token_acc=0.08454386984311447`

official metrics：

- `mIoU = 0.00022642213076544`
- `APC0.9 = 1.8568033273915625e-05`
- `APC1.5 = 0.0024479922622094854`
- `APC3.0 = 0.0031965617926224613`
- `APC4.5 = 0.008419828433261556`

结论：

- 当前 quick baseline 本身学得很差
- quick 线只能用于比较改动方向，不能和正式 baseline 直接横向比较

### 6.2.7 partial baseline quick + semantic init

改动：

- 为新增 map token 增加语义初始化
- 初始化位置：
  [qwen_map_tokenizer.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_tokenizer.py)
  [qwen_map_generator.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/qwen_map_generator.py)
  [qwen_map_pipeline.py](/mnt/data/project/jn/UniMapGen/unimapgen/qwen_map_pipeline.py)

输出目录：

- [qwen_dinov2_map_serialization_av2_partial_quick_semantic_init](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_quick_semantic_init)

训练结果：

- `epoch=1`
- `train_loss=2.633886966854334`
- `val_loss=2.809843599796295`
- `val_token_acc=0.5183033120278907`

official metrics：

- `mIoU = 0.0017733210859359053`
- `APC0.9 = 0.0007928085112716191`
- `APC1.5 = 0.0033112484286612147`
- `APC3.0 = 0.008753471123776812`
- `APC4.5 = 0.008753471123776812`

与 `partial baseline quick` 对比：

- `val_token_acc` 从 `0.0845` 提升到 `0.5183`
- `mIoU` 从 `0.000226` 提升到 `0.001773`
- `APC3.0` 从 `0.00320` 提升到 `0.00875`

结论：

- `semantic init` 是当前已验证有效的修正项
- 它不能单独把指标拉到正式 baseline 水平，但显著改善了 quick 线的学习稳定性
- 后续所有 non-line_type 新实验都应默认带上 `semantic_init_new_map_tokens=true`

### 6.2.8 partial state quick + semantic init

新增配置与脚本：

- [qwen_dinov2_map_serialization_av2_partial_state_quick_semantic_init.yaml](/mnt/data/project/jn/UniMapGen/configs/qwen_dinov2_map_serialization_av2_partial_state_quick_semantic_init.yaml)
- [run_qwen_dinov2_map_serialization_av2_partial_state_quick_semantic_init.sh](/mnt/data/project/jn/UniMapGen/scripts/run_qwen_dinov2_map_serialization_av2_partial_state_quick_semantic_init.sh)
- [eval_qwen_dinov2_map_serialization_av2_partial_state_quick_semantic_init.sh](/mnt/data/project/jn/UniMapGen/scripts/eval_qwen_dinov2_map_serialization_av2_partial_state_quick_semantic_init.sh)

输出目录：

- [qwen_dinov2_map_serialization_av2_partial_state_quick_semantic_init](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_state_quick_semantic_init)

训练结果：

- `epoch=1`
- `train_loss=2.620399570092559`
- `val_loss=2.9453748166561127`
- `val_token_acc=0.4947704822777455`

official metrics：

- `mIoU = 0.04129335493338308`
- `APC0.9 = 0.00033140364307826406`
- `APC1.5 = 0.0031275397808513392`
- `APC3.0 = 0.07056529478506202`
- `APC4.5 = 0.1382744216808563`

与 `partial state quick` 对比：

- `val_token_acc` 从 `0.1026` 提升到 `0.4948`
- `mIoU` 从接近 `0` 提升到 `0.0413`
- `APC3.0` 从接近 `0` 提升到 `0.0706`
- `APC4.5` 从接近 `0` 提升到 `0.1383`

当前判断：

- `state` 之前在 quick 线上的大退步，至少很大一部分来自新增 map token 的初始化过弱
- `semantic init` 对 `state` 的收益比对 no-state baseline 还更明显
- 这说明后续若继续做 `state` 或 paper-stage1，`semantic init` 不应再作为可选项，而应作为默认配方

### 6.2.9 paper stage1 aug no-line-type quick + semantic init

输出目录：

- [qwen_dinov2_map_serialization_av2_paper_stage1_aug_noline_semantic_quick](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_paper_stage1_aug_noline_semantic_quick)

训练结果：

- `epoch=1`
- `train_loss=3.111438473686576`
- `val_loss=2.747724875807762`
- `val_token_acc=0.5300099206349206`

official metrics：

- `mIoU = 0.009029343111684065`
- `APC0.9 = 0.0`
- `APC1.5 = 0.0005271100897255455`
- `APC3.0 = 0.0069315638892226365`
- `APC4.5 = 0.011046972698910789`

与 `partial baseline quick + semantic init` 对比：

- `val_token_acc` 从 `0.5183` 提升到 `0.5300`
- `mIoU` 从 `0.00177` 提升到 `0.00903`
- `APC4.5` 从 `0.00875` 提升到 `0.01105`
- `APC3.0` 从 `0.00875` 下降到 `0.00693`

与 `partial state quick + semantic init` 对比：

- `val_token_acc` 略高于 `state quick + semantic init`
- 但 `mIoU / APC3.0 / APC4.5` 明显落后于 `state quick + semantic init`

当前判断：

- non-line_type 的 paper-stage1 augmentation 没有崩，说明这条主线可继续
- 但在当前 quick 规模下，它还没有超过 `state + semantic init`
- 现阶段最值得优先推进的组合，不再是“纯 baseline”或“paper aug 单独试”，而是把 `semantic init` 作为基础配方后，继续做更 paper-aligned 的 `state` 与 staged 训练

### 6.3 partial state 正式训练

输出目录：

- [qwen_dinov2_map_serialization_av2_partial_state](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_state)

最终结果见 [metrics.jsonl](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_state/metrics.jsonl) 最后一行：

- `epoch=3`
- `train_loss=2.3218`
- `val_loss=2.3772`
- `val_token_acc=0.5496`

与 baseline 对比的结论：

- 当前 state 版本在 partial 数据上没有超过 baseline
- 当前更像是“state 机制可运行，但 prefix 设计和训练配方还没收敛”

### 6.4 partial state 正式评估闭环

`state` 正式版评估闭环已经成功跑完，命令是：

```bash
cd /mnt/data/project/jn/UniMapGen
bash scripts/eval_qwen_dinov2_map_serialization_av2_partial_state.sh
```

这次已确认三步全部跑通：

1. `eval_qwen_map`
2. `predict_qwen_state_scan`
3. `eval_opensatmap_official`

产物位置：

- [predictions_state_scan.json](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_state/predictions_state_scan.json)
- [official_metrics_state_scan.json](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_state/official_metrics_state_scan.json)

本次正式评估结果：

- `val_loss = 2.377478450091917`
- `val_token_acc = 0.5496346852167826`

official metrics 结果：

- `mIoU = 0.0019352629171583522`
- `APM = 0.0`
- `APM50 = 0.0`
- `APM75 = 0.0`
- `APC0.9 = 0.0`
- `APC1.5 = 0.0`
- `APC3.0 = 0.002795370168158987`
- `APC4.5 = 0.009403540462620033`

结论：

- 当前 `state` 链路在工程上已经完整闭环
- 但模型质量仍然很弱，官方指标基本接近 0
- 这进一步支持前面的判断：当前 `state` 版不是“没跑通”，而是“跑通了但没有学好”

### 6.5 本轮环境与评估结论

这轮排查里有一个重要经验需要留给后续 agent：

1. 如果卡在 `Building tokenizer/collator/model...`，不能立刻判定是代码问题。  
   之前已经确认过，这一步可能卡在：
   - `transformers` 导入
   - `torch` 初始化
   - GPU 可见时的 CUDA 初始化链

2. 但在当前这一轮最终成功跑通评估时，环境日志显示：
   - `torch = 2.9.1+cu130`
   - `cuda_built = 13.0`
   - `cuda_available = True`
   - `device_count = 1`

3. 也就是说，之前那类“完全卡住”的问题并不稳定复现。  
   后续 agent 应优先先看：
   - `activate_unimapgen_gpu_env.sh` 输出
   - 是否已经进入 `[Init] Loading Qwen tokenizer...`
   - 是否能完成 `eval_qwen_map`

4. 当前代码侧的评估静默问题已经修过：  
   [eval_qwen_map.py](/mnt/data/project/jn/UniMapGen/unimapgen/eval_qwen_map.py) 和 [predict_qwen_state_scan.py](/mnt/data/project/jn/UniMapGen/unimapgen/predict_qwen_state_scan.py) 现在已经有更明确的初始化日志和进度输出。

### 6.6 partial state full + semantic init 已完成正式训练与正式评估

运行命令：

```bash
env UNIMAPGEN_SKIP_TORCH_PROBE=1 bash /mnt/data/project/jn/UniMapGen/scripts/run_qwen_dinov2_map_serialization_av2_partial_state_semantic_init.sh
env UNIMAPGEN_SKIP_TORCH_PROBE=1 bash /mnt/data/project/jn/UniMapGen/scripts/eval_qwen_dinov2_map_serialization_av2_partial_state_semantic_init.sh
```

输出目录：

- [qwen_dinov2_map_serialization_av2_partial_state_semantic_init](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_state_semantic_init)
- [metrics.jsonl](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_state_semantic_init/metrics.jsonl)
- [predictions_state_scan.json](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_state_semantic_init/predictions_state_scan.json)
- [official_metrics_state_scan.json](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_state_semantic_init/official_metrics_state_scan.json)

本次正式训练结果：

- `epoch1 train_loss = 1.9148524686776471`
- `epoch1 val_loss = 1.9673710302333927`
- `epoch1 val_token_acc = 0.5857067856674104`
- `epoch2 train_loss = 1.4458130647464351`
- `epoch2 val_loss = 1.9521576569448063`
- `epoch2 val_token_acc = 0.5887911799448746`
- `epoch3 train_loss = 1.3884459782700513`
- `epoch3 val_loss = 1.962328106016662`
- `epoch3 val_token_acc = 0.5890755567222296`

official metrics 结果：

- `mIoU = 0.054402087322102576`
- `APM = 1.1726828409283486e-06`
- `APM50 = 7.777794694823889e-06`
- `APM75 = 0.0`
- `APC0.9 = 0.0015197412558194143`
- `APC1.5 = 0.012053863651314804`
- `APC3.0 = 0.09418226114251094`
- `APC4.5 = 0.211287851650644`

和已有 full 结果对比：

- 相比 baseline full，`mIoU` 从 `0.01850` 提升到 `0.05440`
- 相比 baseline full，`APC3.0` 从 `0.02407` 提升到 `0.09418`
- 相比 baseline full，`APC4.5` 从 `0.09414` 提升到 `0.21129`
- 相比旧 state full，`mIoU` 从 `0.00194` 提升到 `0.05440`
- 相比旧 state full，`APC3.0` 从 `0.00280` 提升到 `0.09418`
- 相比旧 state full，`APC4.5` 从 `0.00940` 提升到 `0.21129`

结论：

- `semantic init` 不是小修补，而是当前 `state` 主线能否成立的关键条件
- 在当时已完成的 full 口径对比里，`state + semantic init` 已经从“明显退步”翻转成阶段性最强结果
- 当前最合理的下一步不再是回头重复 old baseline/state，而是把 full paper-style augmentation 和 staged training 接到这条主线上
- 本轮为了避免训练 I/O 被裁剪任务拖慢，曾临时暂停卫星裁剪；训练与评估完成后已恢复该裁剪任务继续运行

### 6.7 baseline full + semantic init 已完成正式评估

说明：

- 这条线原始 full 训练在 `epoch1` 后进入和之前相同的尾部 I/O 挂起
- 为了把实验补完整，后续使用 [best.pt](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_semantic_init/best.pt) 做 warm restart，再继续 2 个 epoch
- 由于 `outputs/` 目录写大 `predictions_val.json` 仍会被底层存储拖住，最终 official eval 改为写到 `/tmp` 完成，再把关键结果回写到输出目录和本文档

相关文件：

- [qwen_dinov2_map_serialization_av2_partial_semantic_init.yaml](/mnt/data/project/jn/UniMapGen/configs/qwen_dinov2_map_serialization_av2_partial_semantic_init.yaml)
- [run_qwen_dinov2_map_serialization_av2_partial_semantic_init.sh](/mnt/data/project/jn/UniMapGen/scripts/run_qwen_dinov2_map_serialization_av2_partial_semantic_init.sh)
- [qwen_dinov2_map_serialization_av2_partial_semantic_init_resume.yaml](/mnt/data/project/jn/UniMapGen/configs/qwen_dinov2_map_serialization_av2_partial_semantic_init_resume.yaml)
- [run_qwen_dinov2_map_serialization_av2_partial_semantic_init_resume.sh](/mnt/data/project/jn/UniMapGen/scripts/run_qwen_dinov2_map_serialization_av2_partial_semantic_init_resume.sh)
- [eval_qwen_dinov2_map_serialization_av2_partial_semantic_init_resume.sh](/mnt/data/project/jn/UniMapGen/scripts/eval_qwen_dinov2_map_serialization_av2_partial_semantic_init_resume.sh)
- [metrics.jsonl](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_semantic_init/metrics.jsonl)
- [metrics.jsonl](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_semantic_init_resume/metrics.jsonl)
- [eval_summary_val.json](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_semantic_init_resume/eval_summary_val.json)
- [official_metrics_val.json](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_av2_partial_semantic_init_resume/official_metrics_val.json)

本次可确认的训练/验证结果：

- 原始 full `epoch1 train_loss = 1.7644217022874737`
- 原始 full `epoch1 val_loss = 1.330607225645834`
- 原始 full `epoch1 val_token_acc = 0.6730542065887912`
- warm restart `epoch1 train_loss = 1.1752430417919686`
- warm restart `epoch1 val_loss = 1.0332993233381813`
- warm restart `epoch1 val_token_acc = 0.7382639891499322`
- 最终 best checkpoint 的单独正式 loss eval：`val_loss = 1.0053184592901772`
- 最终 best checkpoint 的单独正式 loss eval：`val_token_acc = 0.7455702848142801`

official metrics 结果：

- `mIoU = 0.07374541881296288`
- `APM = 3.577758112775223e-05`
- `APM50 = 0.000178820339013289`
- `APM75 = 1.0254926665181132e-05`
- `APC0.9 = 0.013050152714274838`
- `APC1.5 = 0.049020012320650064`
- `APC3.0 = 0.2109197192721454`
- `APC4.5 = 0.3528470790958374`

和已有 full 结果对比：

- 相比 baseline full，`mIoU` 从 `0.01850` 提升到 `0.07375`
- 相比 baseline full，`APC3.0` 从 `0.02407` 提升到 `0.21092`
- 相比 baseline full，`APC4.5` 从 `0.09414` 提升到 `0.35285`
- 相比 `state + semantic init` full，`mIoU` 从 `0.05440` 提升到 `0.07375`
- 相比 `state + semantic init` full，`APC3.0` 从 `0.09418` 提升到 `0.21092`
- 相比 `state + semantic init` full，`APC4.5` 从 `0.21129` 提升到 `0.35285`

结论：

- `semantic init` 单独作用在 baseline 上也能带来非常大的收益
- 当前最强 full 结果不再是 `state + semantic init`，而是 `baseline + semantic init`
- 这说明目前还不能说 `state` 在修正初始化后带来了额外收益；至少在当前 partial full 口径下，`state` 反而不如不加 `state`
- 当前最合理的论文对齐主线应更新为：先把 non-state + semantic init 作为 strongest baseline，再把 paper-style augmentation / staged training 接到这条线上

## 7. 当前最重要的事实结论

1. partial 数据主线已经跑到了正式训练阶段，不再只是 smoke。
2. 当前 baseline 是有效参照，但它已经不再是最强结果；`baseline + semantic init` full 已经明确超过 baseline。
3. 旧 `state` 正式评估确实退步，且在 current partial full 口径下即使加上 `semantic init` 也仍不如 `baseline + semantic init`。
4. `line_type` 相关 quick 对比已经做了 4 轮：原版、prompt fix、optional grammar、skip-none supervision，结果全部 official metrics 为 0。
5. 当前 `line_type` 分支的共同退化模式非常稳定：输出塌成 `lane_line` 的两点零坐标模板。
6. `semantic init` 已被 quick 和 full 对照验证为有效修正项，应视为后续主线默认设置。
7. 这说明当前最不该做的是继续投入 `state + line_type full`。
8. line type 和 official metrics 已接入，paper augmentation 快照与 non-line_type quick 配置现在都已经可用。
9. 在当前 full 对比里，最强结果已经变成 `baseline + semantic init`；`state + semantic init` 虽然显著优于旧 state，但不是当前最优。
10. 当前离论文完全复现仍差：
   - full paper-scale augmentation
   - 更 paper-aligned tokenizer / high-res bins
   - 更 paper-aligned non-line_type `cut_points` 实验
   - staged training
   - PV 正式接入当前主线

## 8. 后续 agent 继续复现时的推荐顺序

### 第一优先级

1. 暂停 `state + line_type full`
2. 以 `baseline + semantic init` 作为当前默认 strongest baseline 继续推进复现
3. 后续 `baseline/state/paper-stage1` 新实验默认启用 `semantic init`
4. 在此基础上继续 full paper-style augmentation 数据构建
5. 再做更 paper-aligned 的 staged / augmentation 组合，而不是单独重复 quick baseline
6. 继续直接对比 `baseline`、`baseline + semantic init`、`old state`、`state + semantic init` 与后续 non-line_type 新实验

### 第二优先级

1. 优先做 non-line_type 的下一步 paper 对齐实验
2. 推荐先把 full paper augmentation 数据集裁剪完整
3. 然后在其上补“去掉 `line_type` 且带 `semantic init`”的 full / staged 实验
4. 之后再重新判断 `state` 是否值得接到 paper augmentation 主线上
5. 或者补更论文式的 non-line_type `cut_points` state 分支
6. 必要时再补 `baseline` / `state` 的更细粒度错误样例分析

### 第三优先级

1. 如果 `paper_stage1_aug_quick` 有正向结果，再构建 full paper augmentation 数据集
2. 再决定是否进入更完整的 staged training

### 第四优先级

1. 若后续仍想保留 `line_type`，应重做设计而不是继续堆训练
2. 可考虑把 `line_type` 限制为 lane_line 专属分支，或单独 staged 训练，不要再做全类别强制串行 token
3. 补 high-res coordinate bins
4. 再规划 PV 接入

## 9. 后续 agent 不要重复踩的坑

1. 不要把动态裁剪目录直接当训练集。
2. 不要把 quick 结果当成正式结论。
3. 不要在单卡上同时并行跑多个训练。
4. 不要看到 epoch 进度条结束就以为训练卡住，先看是否在验证或写 checkpoint。
5. 不要把 `Activated env` 后的等待误判成 `conda activate` 卡住，优先怀疑 torch/CUDA probe。

## 10. 结果记录约定

后续新的正式训练或正式评估结果，建议继续直接追加到本文件，而不是只停留在终端输出。

每次至少补这几项：

1. 运行命令
2. 输出目录
3. 关键训练指标：`epoch / train_loss / val_loss / val_token_acc`
4. 关键评估指标：`mIoU / APM / APM50 / APM75 / APC0.9 / APC1.5 / APC3.0 / APC4.5`
5. 一句话结论：比谁更好，或者还缺什么

这样后续 agent 可以只读这份文档就快速接上，不需要再手动翻终端历史。

## 11. 后续复现工作清单

下面这份清单按当前优先级排序，后续 agent 应直接从上往下执行，不要跳回已经证明收益较低的旧分支。

### P0 当前主线

1. 持续监控 full paper augmentation 裁剪任务是否完成  
   当前裁剪目录：[/mnt/data/project/jn/satellite_tools/av2_opensatmap_crops_paper896_fix](/mnt/data/project/jn/satellite_tools/av2_opensatmap_crops_paper896_fix)
2. 用完整 paper-style augmentation 数据构建新的 full 训练快照
3. 在 full augmentation 数据上先跑 non-state + semantic init 主线
4. 对该主线做完整 `eval_qwen_map` + `predict_qwen_map` + `eval_opensatmap_official`
5. 与当前 strongest baseline `baseline + semantic init full` 做正式对比

### P1 论文对齐

1. 在 non-state + semantic init strongest baseline 上尝试 staged training
2. 补更 paper-aligned 的 non-line_type `cut_points` 实验
3. 评估是否需要把 `coord_num_bins` / tokenizer 设计改到更 paper-aligned
4. 继续保持 `semantic init` 为默认设置，不再回退到原始 token 初始化

### P2 state 分支再判断

1. 不要直接继续 `state + line_type full`
2. 只有在 non-state 主线 augmentation / staged 结果稳定后，再决定是否把 `state` 接回主线
3. 如果重启 `state`，优先做 non-line_type state，而不是 line_type state
4. 所有新的 state 实验都必须和 `baseline + semantic init full` 直接对比

### P3 暂缓项

1. 暂缓 `line_type`，除非先重做任务定义
2. 暂缓多实验并行训练，当前磁盘 I/O 会把训练和预测拖入 `D` 状态
3. 暂缓把 quick 结果当结论，后续必须优先 full 口径

### 每轮实验结束必须补的内容

1. 运行命令
2. 输出目录
3. `val_loss` 和 `val_token_acc`
4. `mIoU / APC0.9 / APC1.5 / APC3.0 / APC4.5`
5. 和 `baseline + semantic init full` 的对比结论
