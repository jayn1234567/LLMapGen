# UniMapGen 仓库架构审计（2026-03-07）

## 1. 结论

`/mnt/data/project/jn/UniMapGen` 已经不是零散代码集合，而是一个已经搭起项目架构的仓库。

更准确地说，当前状态是：

- 已经搭起了完整的工程骨架
- 已经有一条真实可运行的主线
- 已经有论文对齐方向的脚手架
- 但还没有完成论文完整版架构的全部闭环

因此，判断应为：

- 从工程角度：架构已经搭起来了
- 从论文完整复现角度：还没有完全搭完

## 2. 当前仓库的层次结构

当前仓库可以拆成 6 层：

1. 数据层
2. 序列化层
3. 编码器与模型层
4. 训练/评估/推理入口层
5. 配置与脚本层
6. 文档与实验产物层

### 2.1 数据层

主要目录：
- [unimapgen/data](/mnt/data/project/jn/UniMapGen/unimapgen/data)

当前已经具备的数据模块：
- `NuScenesSatelliteMapDataset`
- `NuScenesSDMapDataset`
- `OpenSatMapDataset`
- `OpenSatMapQwenDataset`
- 统一 builder 入口

关键文件：
- [builders.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/builders.py)
- [dataset.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/dataset.py)
- [nuscenes_sdmap_dataset.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/nuscenes_sdmap_dataset.py)
- [opensatmap_dataset.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/opensatmap_dataset.py)
- [qwen_map_dataset.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_dataset.py)
- [__init__.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/__init__.py)

判断：
- 数据层已经成型
- 不是靠单文件硬编码
- 已支持多数据源切换

### 2.2 序列化层

当前已经独立实现地图 token 化和反序列化：
- polyline -> discrete token
- token -> polyline
- state prefix 构造
- grammar constraint

关键文件：
- [serialization.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/serialization.py)
- [qwen_map_tokenizer.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_tokenizer.py)
- [qwen_map_dataset.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_dataset.py)
- [state_geometry.py](/mnt/data/project/jn/UniMapGen/unimapgen/state_geometry.py)

判断：
- 序列化层已经完整搭起
- 而且已经和训练/推理主线联通

### 2.3 编码器与模型层

主要目录：
- [unimapgen/models](/mnt/data/project/jn/UniMapGen/unimapgen/models)

当前可以分成三类模型：

1. 基础序列模型线
- [unimapgen_v1.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/unimapgen_v1.py)

2. 论文对齐脚手架线
- [unimapgen_paper.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/unimapgen_paper.py)

3. 当前最稳定的 Qwen 主线
- [qwen_map_generator.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/qwen_map_generator.py)

编码器层：
- [satellite_encoder.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/encoders/satellite_encoder.py)
- [pv_encoder.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/encoders/pv_encoder.py)
- [vision_adapter.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/adapters/vision_adapter.py)
- [map_llm.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/llm/map_llm.py)
- [dino_lane_seg.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/dino_lane_seg.py)

统一模型工厂：
- [models/__init__.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/__init__.py)

判断：
- 模型层已经有统一组织
- 不同路线之间已经形成可区分的结构
- 但论文完整模型还没有全部收口到同一条最终主线

### 2.4 训练/评估/推理入口层

当前至少有两套入口：

1. 通用 `v1/paper` 训练入口
- [train.py](/mnt/data/project/jn/UniMapGen/unimapgen/train.py)
- [eval.py](/mnt/data/project/jn/UniMapGen/unimapgen/eval.py)
- [predict.py](/mnt/data/project/jn/UniMapGen/unimapgen/predict.py)

2. Qwen map serialization 专用入口
- [train_qwen_map.py](/mnt/data/project/jn/UniMapGen/unimapgen/train_qwen_map.py)
- [eval_qwen_map.py](/mnt/data/project/jn/UniMapGen/unimapgen/eval_qwen_map.py)
- [predict_qwen_map.py](/mnt/data/project/jn/UniMapGen/unimapgen/predict_qwen_map.py)
- [predict_qwen_state_scan.py](/mnt/data/project/jn/UniMapGen/unimapgen/predict_qwen_state_scan.py)

另外还有 lane segmentation 验证分支：
- [train_lane_seg.py](/mnt/data/project/jn/UniMapGen/unimapgen/train_lane_seg.py)
- [eval_lane_seg.py](/mnt/data/project/jn/UniMapGen/unimapgen/eval_lane_seg.py)
- [predict_lane_seg.py](/mnt/data/project/jn/UniMapGen/unimapgen/predict_lane_seg.py)

判断：
- 入口层已经成型
- 不只是 notebook 风格原型
- 已经支持独立训练、评估、预测和状态扫描推理

### 2.5 配置与脚本层

主要目录：
- [configs](/mnt/data/project/jn/UniMapGen/configs)
- [scripts](/mnt/data/project/jn/UniMapGen/scripts)

说明：
- 当前已经有多套 config
- 已经有 smoke 运行脚本
- 已经有单独 train 脚本
- 已经有 paper scaffold 的 staged config

判断：
- 配置层已经具备实验管理雏形
- 但还没有完全统一成最终单一训练体系

### 2.6 文档与实验产物层

主要目录：
- [docs](/mnt/data/project/jn/UniMapGen/docs)
- [outputs](/mnt/data/project/jn/UniMapGen/outputs)

说明：
- 文档已经不只是 README，而是含路线、对比、分支说明、阶段总结
- `outputs` 中已经存在多条分支真实运行产物

判断：
- 这说明仓库不是“只写了架构没运行”
- 至少主线实验已经真实执行过

## 3. 当前仓库架构图

可以把当前仓库理解成下面这张结构图：

```text
ckpts/
  DINOv2 / Qwen / CLIP local checkpoints

unimapgen/data/
  builders.py
  dataset.py / nuscenes_sdmap_dataset.py / opensatmap_dataset.py
  serialization.py
  qwen_map_dataset.py
  qwen_map_tokenizer.py

unimapgen/models/
  encoders/
    satellite_encoder.py
    pv_encoder.py
  adapters/
    vision_adapter.py
  llm/
    map_llm.py
  unimapgen_v1.py
  unimapgen_paper.py
  qwen_map_generator.py
  dino_lane_seg.py

unimapgen/
  train.py / eval.py / predict.py
  train_qwen_map.py / eval_qwen_map.py / predict_qwen_map.py
  predict_qwen_state_scan.py
  train_lane_seg.py / eval_lane_seg.py / predict_lane_seg.py
  qwen_map_pipeline.py
  state_geometry.py

configs/
  v1 / paper / qwen / lane-seg smoke configs

scripts/
  run_*.sh
  train_*.sh

docs/
  branch docs / roadmap / stage docs / task assignment

outputs/
  smoke checkpoints / metrics / predictions
```

## 4. 哪些模块是当前“核心主线”

这里的核心主线，指的是：
- 代码已经联通
- 训练/推理已经实际跑过
- 不只是概念性脚手架

### 4.1 当前最核心的主线

当前仓库最成熟的一条主线是：

`OpenSatMapQwenDataset -> MapSequenceTokenizer -> QwenMapTokenizer -> QwenSatelliteMapGenerator -> train_qwen_map / predict_qwen_map / predict_qwen_state_scan`

关键文件：
- [qwen_map_dataset.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_dataset.py)
- [serialization.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/serialization.py)
- [qwen_map_tokenizer.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_tokenizer.py)
- [qwen_map_generator.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/qwen_map_generator.py)
- [qwen_map_pipeline.py](/mnt/data/project/jn/UniMapGen/unimapgen/qwen_map_pipeline.py)
- [train_qwen_map.py](/mnt/data/project/jn/UniMapGen/unimapgen/train_qwen_map.py)
- [predict_qwen_state_scan.py](/mnt/data/project/jn/UniMapGen/unimapgen/predict_qwen_state_scan.py)

这条线已经具备：
- 卫星图 DINOv2 编码
- map serialization
- prompt tokenization
- state update
- constrained decoding
- 训练、评估、预测闭环

### 4.2 次核心主线：基础通用训练线

这条线主要是 `v1/paper` 路线：
- [train.py](/mnt/data/project/jn/UniMapGen/unimapgen/train.py)
- [builders.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/builders.py)
- [models/__init__.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/__init__.py)
- [unimapgen_v1.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/unimapgen_v1.py)

这条线说明仓库已经有统一训练器和模型工厂，不只是 Qwen 专线。

### 4.3 验证性辅线：卫星图 lane segmentation

这条线的作用不是论文主线，而是验证：
- DINOv2 backbone 可加载
- 卫星图数据能训练
- 基础视觉分支能运行

关键文件：
- [lane_seg_dataset.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/lane_seg_dataset.py)
- [dino_lane_seg.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/dino_lane_seg.py)
- [train_lane_seg.py](/mnt/data/project/jn/UniMapGen/unimapgen/train_lane_seg.py)

## 5. 哪些模块是“脚手架 / 实验线 / 半成品”

### 5.1 `UniMapGenPaper` 是论文脚手架，不是最终完成版

文件：
- [unimapgen_paper.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/unimapgen_paper.py)

它已经体现了论文方向：
- satellite encoder
- PV encoder
- text prompt prefix
- Qwen-like decoder memory fusion

但它仍然更接近“paper-aligned scaffold”，原因是：
- 当前最稳定、最完整的 map serialization + state 训练闭环不在这条线上
- 它还没有成为项目唯一主线
- 它更多是在统一 BEV/PV/Text 方向上的架构预留

所以它是：
- 有意义的架构脚手架
- 不是无用占位
- 但也不是已完成的论文最终实现

### 5.2 `configs/unimapgen_paper_scaffold.yaml` 是脚手架配置

文件：
- [unimapgen_paper_scaffold.yaml](/mnt/data/project/jn/UniMapGen/configs/unimapgen_paper_scaffold.yaml)

从命名和内容都能看出来，这是一套 paper scaffold config，而不是最终正式训练 config。

### 5.3 多条路线并存，说明仓库还在收敛阶段

当前同时存在：
- `unimapgen_v1*.yaml`
- `unimapgen_paper*.yaml`
- `qwen_dinov2_map_serialization*.yaml`
- `dinov2_lane_seg*.yaml`

这说明：
- 架构探索已经比较充分
- 但最终统一到哪条主线，还没有完全收敛到单一入口

## 6. 证据：当前仓库不是“空架构”

以下现象表明仓库已经至少完成了项目级搭建：

1. 已有真实输出目录：
- [outputs/qwen_dinov2_map_serialization_smoke](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_smoke)
- [outputs/qwen_dinov2_map_serialization_state_smoke](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_state_smoke)
- [outputs/dinov2_lane_seg_unaligned_smoke](/mnt/data/project/jn/UniMapGen/outputs/dinov2_lane_seg_unaligned_smoke)

2. 已有 train-only 和 end-to-end 运行脚本：
- [train_qwen_dinov2_map_serialization_smoke.sh](/mnt/data/project/jn/UniMapGen/scripts/train_qwen_dinov2_map_serialization_smoke.sh)
- [train_qwen_dinov2_map_serialization_state_smoke.sh](/mnt/data/project/jn/UniMapGen/scripts/train_qwen_dinov2_map_serialization_state_smoke.sh)
- [run_qwen_dinov2_map_serialization_state_smoke.sh](/mnt/data/project/jn/UniMapGen/scripts/run_qwen_dinov2_map_serialization_state_smoke.sh)

3. 已有阶段文档和任务文档：
- [reproduction_status_20260306.md](/mnt/data/project/jn/UniMapGen/docs/reproduction_status_20260306.md)
- [qwen_dinov2_map_serialization_branch.md](/mnt/data/project/jn/UniMapGen/docs/qwen_dinov2_map_serialization_branch.md)
- [full_reproduction_task_assignment.md](/mnt/data/project/jn/UniMapGen/docs/full_reproduction_task_assignment.md)

## 7. 当前仓库的真实状态判断

### 7.1 已经完成的架构能力

当前已经明确具备：
- 多数据源数据层
- 统一 builder 入口
- 多模型实现层
- Qwen 主线可运行
- state update 可运行
- 单独的 lane segmentation 验证线
- 配置和脚本层
- 文档与实验产物层

### 7.2 仍然没有完全完成的部分

当前还没有完全收口的主要是：
- 正式对齐卫星数据的大规模接入
- 论文规模 patch 增强流程
- PV 分支正式并入当前最稳定主线
- 统一到最终单一训练主线
- 论文最终评估和消融闭环

## 8. 结论性判断

如果问题是：

“这个仓库有没有把 UniMapGen 的工程架构搭起来？”

答案是：
- 有，已经搭起来了

如果问题是：

“这个仓库是不是已经把论文完整架构全部实现完了？”

答案是：
- 还没有

更准确的判断是：

- 当前仓库已经完成了 `工程骨架 + 可运行主线 + paper scaffold`
- 还没有完成 `论文完整版最终统一主线`

## 9. 建议你怎么看这个仓库

建议不要把当前仓库理解成“还没开始搭架构”，而应该理解成：

- 第一阶段：基础 v1 架构探索
- 第二阶段：Qwen + DINOv2 map serialization 主线打通
- 第三阶段：paper scaffold 与多模态方向预埋
- 当前正在从“多路线并存”收敛到“论文完整版统一主线”

这也是为什么你现在最该做的，不是重写架构，而是：

1. 确定最终主线以哪条为主
2. 把正式对齐数据接进来
3. 把 PV 和正式评估补到当前主线上
