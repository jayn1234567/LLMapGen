# UniMapGen Agent Handoff（2026-03-07）

## 1. 文档目的

这份文档用于给后续 agent 直接接手 `/mnt/data/project/jn/UniMapGen` 的复现工作。

目标不是重复解释论文背景，而是说明：
- 已经做了什么
- 当前哪条主线最成熟
- 代码和实验跑到了哪一步
- 还缺什么
- 后续应该按什么顺序继续

## 2. 当前项目结论

当前仓库已经具备：
- 完整工程骨架
- 一条可运行的 Qwen 主线
- 一套可运行的 state update 机制
- 一条验证用的 DINOv2 lane segmentation 分支
- 论文方向的 `UniMapGenPaper` 脚手架

当前仓库还不具备：
- 论文完整版统一主线
- 正式对齐卫星数据上的完整训练闭环
- 正式 PV 多模态主线闭环
- 论文规模实验与最终指标复现

最重要的判断：
- 当前最成熟、最可继续开发的主线是 `Qwen + DINOv2 + map serialization + state update`
- 不是 `UniMapGenPaper` 这一层

参考文档：
- [current_repo_architecture_audit_20260307.md](/mnt/data/project/jn/UniMapGen/docs/current_repo_architecture_audit_20260307.md)
- [reproduction_status_20260306.md](/mnt/data/project/jn/UniMapGen/docs/reproduction_status_20260306.md)
- [qwen_dinov2_map_serialization_branch.md](/mnt/data/project/jn/UniMapGen/docs/qwen_dinov2_map_serialization_branch.md)
- [full_reproduction_task_assignment.md](/mnt/data/project/jn/UniMapGen/docs/full_reproduction_task_assignment.md)

## 3. 已完成工作

### 3.1 卫星图编码验证分支

已完成：
- 接入本地 `DINOv2 ViT-L/14`
- 冻结 backbone
- 训练一个 lane segmentation 分支，用于验证卫星图特征可正常提取

关键文件：
- [satellite_encoder.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/encoders/satellite_encoder.py)
- [dino_lane_seg.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/dino_lane_seg.py)
- [lane_seg_dataset.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/lane_seg_dataset.py)
- [train_lane_seg.py](/mnt/data/project/jn/UniMapGen/unimapgen/train_lane_seg.py)

关键配置：
- [dinov2_lane_seg_unaligned_smoke.yaml](/mnt/data/project/jn/UniMapGen/configs/dinov2_lane_seg_unaligned_smoke.yaml)

已有输出：
- [outputs/dinov2_lane_seg_unaligned_smoke](/mnt/data/project/jn/UniMapGen/outputs/dinov2_lane_seg_unaligned_smoke)

### 3.2 Qwen + DINOv2 地图序列化主线

已完成：
- DINOv2 卫星图 token 提取
- linear projection 到 Qwen hidden
- Qwen tokenizer 扩展 map token
- map serialization / detokenizer
- prompt tokenization
- 当前 patch map token 监督训练
- constrained decoding

关键文件：
- [serialization.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/serialization.py)
- [qwen_map_tokenizer.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_tokenizer.py)
- [qwen_map_dataset.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_dataset.py)
- [qwen_map_generator.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/qwen_map_generator.py)
- [qwen_map_pipeline.py](/mnt/data/project/jn/UniMapGen/unimapgen/qwen_map_pipeline.py)
- [train_qwen_map.py](/mnt/data/project/jn/UniMapGen/unimapgen/train_qwen_map.py)
- [eval_qwen_map.py](/mnt/data/project/jn/UniMapGen/unimapgen/eval_qwen_map.py)
- [predict_qwen_map.py](/mnt/data/project/jn/UniMapGen/unimapgen/predict_qwen_map.py)

关键配置：
- [qwen_dinov2_map_serialization_smoke.yaml](/mnt/data/project/jn/UniMapGen/configs/qwen_dinov2_map_serialization_smoke.yaml)

已有输出：
- [outputs/qwen_dinov2_map_serialization_smoke](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_smoke)

### 3.3 State Update 主线

已完成：
- 训练侧 previous state prefix
- 推理侧 patch scan
- 局部邻接 state，而不是直接使用全部全局历史
- `cut` 状态建模
- endpoint primitive 到 short trace primitive 的增强
- geometry-aware global merge
- geometry-aware state projection

关键文件：
- [state_geometry.py](/mnt/data/project/jn/UniMapGen/unimapgen/state_geometry.py)
- [qwen_map_dataset.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_dataset.py)
- [predict_qwen_state_scan.py](/mnt/data/project/jn/UniMapGen/unimapgen/predict_qwen_state_scan.py)

关键配置：
- [qwen_dinov2_map_serialization_state_smoke.yaml](/mnt/data/project/jn/UniMapGen/configs/qwen_dinov2_map_serialization_state_smoke.yaml)

已有输出：
- [outputs/qwen_dinov2_map_serialization_state_smoke](/mnt/data/project/jn/UniMapGen/outputs/qwen_dinov2_map_serialization_state_smoke)

### 3.4 文档与路线整理

已完成：
- 阶段性复现总结
- 分支说明
- 架构审计
- 团队任务分工文档

关键文档：
- [reproduction_status_20260306.md](/mnt/data/project/jn/UniMapGen/docs/reproduction_status_20260306.md)
- [qwen_dinov2_map_serialization_branch.md](/mnt/data/project/jn/UniMapGen/docs/qwen_dinov2_map_serialization_branch.md)
- [current_repo_architecture_audit_20260307.md](/mnt/data/project/jn/UniMapGen/docs/current_repo_architecture_audit_20260307.md)
- [full_reproduction_task_assignment.md](/mnt/data/project/jn/UniMapGen/docs/full_reproduction_task_assignment.md)

## 4. 当前最重要的主线

后续 agent 默认应该继续沿这条线推进：

`OpenSatMapQwenDataset -> MapSequenceTokenizer -> QwenMapTokenizer -> QwenSatelliteMapGenerator -> train_qwen_map -> predict_qwen_state_scan`

关键链路：
- [qwen_map_dataset.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_dataset.py)
- [serialization.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/serialization.py)
- [qwen_map_tokenizer.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_tokenizer.py)
- [qwen_map_generator.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/qwen_map_generator.py)
- [predict_qwen_state_scan.py](/mnt/data/project/jn/UniMapGen/unimapgen/predict_qwen_state_scan.py)

原因：
- 这条线已经真实跑通过
- 训练、评估、推理和 state-scan 都有产物
- 已经具备后续接正式数据和补 PV 的基础

## 5. 当前已验证过的运行状态

### 5.1 环境

已建立独立环境：
- `/mnt/data/project/jn/UniMapGen/.envs/unimapgen-gpu`

激活脚本：
- [activate_unimapgen_gpu_env.sh](/mnt/data/project/jn/UniMapGen/scripts/activate_unimapgen_gpu_env.sh)

注意：
- 当前 agent 会话未必总能看到 GPU
- 用户本机终端已经验证 `torch.cuda.is_available() == True`
- 所以后续真正训练应优先由用户终端执行

环境说明：
- [environment_gpu_setup.md](/mnt/data/project/jn/UniMapGen/docs/environment_gpu_setup.md)

### 5.2 已跑通命令

Qwen 主线已成功跑通过：
- `python -m unimapgen.train_qwen_map --config configs/qwen_dinov2_map_serialization_smoke.yaml`
- `python -m unimapgen.predict_qwen_map --config ...`
- `python -m unimapgen.eval_qwen_map --config ...`

State 版已成功跑通过：
- `bash scripts/run_qwen_dinov2_map_serialization_state_smoke.sh`
- `python -m unimapgen.predict_qwen_state_scan --config ...`

Lane seg 已成功跑通过：
- `python -m unimapgen.train_lane_seg --config configs/dinov2_lane_seg_unaligned_smoke.yaml`

## 6. 当前已知设计结论

### 6.1 关于架构状态

- 仓库架构已经搭起来了
- 当前不是“缺架构”，而是“缺最终收敛和正式数据闭环”

### 6.2 关于 `UniMapGenPaper`

- [unimapgen_paper.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/unimapgen_paper.py) 是有意义的 paper-aligned scaffold
- 但不是当前最成熟、最应该继续推进的主线
- 当前真正闭环的是 `qwen_map_generator.py` 这一支

### 6.3 关于 State Update

已经明确成立：
- 当前 patch 不应直接吃全局历史
- 应从 previous global state 中抽取当前 patch 邻接的局部状态
- 当前实现已经切到这种局部邻接 state 逻辑

### 6.4 关于 state prefix 形式

当前默认推荐保留：
- `cut_traces`

理由：
- 比单点 `cut_points` 信息更强
- 仍然属于局部边界状态
- 比直接全量历史更接近论文的局部连续性目标

## 7. 当前未完成工作

### 7.1 必须继续做的

1. 正式对齐卫星数据接入
- 当前仍主要依赖 small-set / unaligned data 做验证
- 后续必须切到你提供的对齐卫星数据

2. 论文规模 patch 构建与增强
- 正式 patch grid
- overlap
- rotate
- tilt 等增强流程

3. PV 分支接入当前主线
- 当前仓库有 PV encoder 和通用训练入口支持
- 但没有把 PV 正式并到当前最成熟的 Qwen serialization 主线里

4. 正式评估与消融
- 当前主要验证 pipeline
- 还没进入论文实验收口阶段

### 7.2 建议继续做的

1. tokenizer 细节再对齐论文
2. prompt 体系再收敛
3. geometry merge 继续严格化
4. 统一通用训练线和 Qwen 训练线的职责边界

## 8. 后续 agent 的推荐执行顺序

### 第一优先级

1. 接入正式对齐卫星数据
2. 让 `qwen_map_dataset.py` 和 `predict_qwen_state_scan.py` 都切到正式 geometry
3. 确认 state update 在正式数据上继续成立

### 第二优先级

1. 把 PV 分支接到当前 Qwen 主线
2. 形成 `satellite + state + PV` 的最小闭环
3. 保持现有 serialization 和 constrained decoding 不被破坏

### 第三优先级

1. 放大训练配置
2. 做正式评估和消融
3. 再决定是否进一步调整 tokenizer / prompt / merge 规则

## 9. 后续 agent 不建议做的事

1. 不要推翻当前 Qwen 主线重写架构
- 当前主线已经能跑
- 应该在其上继续接正式数据和多模态

2. 不要过早追论文指标
- 当前首要目标仍然是正式数据闭环与多模态闭环

3. 不要把 `UniMapGenPaper` 误判成当前唯一主线
- 它是脚手架，不是当前最稳的生产主线

4. 不要把 state update 再退回“直接喂全局历史”
- 当前已经明确改成局部邻接 state

## 10. AV2 + OpenSatMap 对齐裁剪进展

### 10.1 本轮新增结论

已完成一条可运行的 AV2 对齐卫星图构建链：
- 用 AV2 官方 API 把每帧 `city_xy` 转成 WGS84
- 从 OpenSatMap 裁论文尺寸 `896x896` 卫星 patch
- 同步提取 OpenSatMap 线真值
- 额外支持把 AV2 地图真值叠加到同一张卫星图上做人工核验

这一轮最重要的结论：
- AV2 官方坐标到 GPS 的链路是可用的
- 之前 AV2 真值叠到卫星图上的“系统性偏移”主要不是 AV2 GT 本身问题
- 根因是 OpenSatMap 的 `GPS_info_all.json` 提供的是网格中心点，不是整张 `4096x4096` 图的真实边界
- 修正边界推断后，AV2 GT 叠加结果已经明显合理

### 10.2 当前可用脚本

新增/更新脚本：
- [crop_opensatmap_for_av2.py](/mnt/data/project/jn/satellite_tools/crop_opensatmap_for_av2.py)
- [visualize_av2_satellite_gt_overlay.py](/mnt/data/project/jn/satellite_tools/visualize_av2_satellite_gt_overlay.py)
- [visualize_av2_vs_opensatmap_gt.py](/mnt/data/project/jn/satellite_tools/visualize_av2_vs_opensatmap_gt.py)

各脚本用途：
- `crop_opensatmap_for_av2.py`
  - 输入 AV2 `map-change-dataset`
  - 输出每帧对应的 OpenSatMap 卫星图与 OpenSatMap GT
- `visualize_av2_satellite_gt_overlay.py`
  - 检查 OpenSatMap GT 与裁出的卫星图是否对齐
- `visualize_av2_vs_opensatmap_gt.py`
  - 在同一张卫星图上对比 OpenSatMap GT 与 AV2 GT
  - 当前 AV2 叠加使用的是 `lane boundary`

### 10.3 已定位并修复的问题

偏移问题来源：
- 旧实现直接使用 `min/max(centerGPS)` 作为 tile 边界
- 但 OpenSatMap 每张图实际上对应 `8x8` 的中心点网格
- 正确做法应当先根据相邻中心点间距，向四周各扩半个 grid step，恢复真实 tile 边界

修复方式：
- 在裁剪脚本和可视化脚本里都加入了 `infer_axis_bounds_from_centers(...)`
- 对经纬度范围做半步扩边，再做 GPS -> 像素映射

一个明确样本：
- 旧版样本中心：`(978, 697)`
- 修正后中心：`(1112, 866)`
- 差异约：`(+134 px, +169 px)`

结论：
- 旧目录 [av2_opensatmap_crops_paper896](/mnt/data/project/jn/satellite_tools/av2_opensatmap_crops_paper896) 对 OpenSatMap GT 叠加仍可看
- 但如果要检查或使用 AV2 GT 对齐，不应继续使用旧目录
- 后续应统一使用修复后重新裁出的目录

### 10.4 当前推荐使用的数据与可视化结果

推荐使用：
- 修复后裁剪目录 [av2_opensatmap_crops_paper896_fix](/mnt/data/project/jn/satellite_tools/av2_opensatmap_crops_paper896_fix)
- 10 张人工核验结果 [av2_vs_opensatmap_vis10_fix](/mnt/data/project/jn/satellite_tools/av2_vs_opensatmap_vis10_fix)

其中重点查看：
- [compare](/mnt/data/project/jn/satellite_tools/av2_vs_opensatmap_vis10_fix/compare)

人工检查结论：
- 修复后 AV2 GT 的几何位置基本回到合理区域
- 仍存在少量不完全重合，这是预期现象
- 原因主要来自：
  - AV2 GT 当前画的是 `lane boundary`
  - OpenSatMap GT 更接近卫星图上可见的道路线/边界线
  - 两套地图可能存在时间差与标注口径差异

### 10.5 当前建议的数据使用方式

当前更合理的监督分工是：
- 卫星图分支 / DINOv2 分支：优先用 OpenSatMap GT
- 最终结构化地图输出分支：优先用 AV2 GT

原因：
- OpenSatMap GT 与卫星图同源，更适合卫星图像监督
- AV2 GT 更接近自动驾驶地图任务定义，更适合最终地图生成目标

### 10.6 后续 agent 的直接建议

后续继续时，默认按下面顺序做：

1. 基于 [av2_opensatmap_crops_paper896_fix](/mnt/data/project/jn/satellite_tools/av2_opensatmap_crops_paper896_fix) 继续，不再沿用旧版 `av2_opensatmap_crops_paper896`
2. 将需要训练的 sample 时间戳从“每个 pose”切换到“按 lidar 或目标传感器时间戳采样”，避免一个 log 产生过密 patch
3. 把论文里的 `overlapped + inclined crop + rotation` 作为下一阶段数据增强加入
4. 若要做更严格对比，可继续把 AV2 `drivable area` 边界也加入可视化脚本

## 11. 后续 agent 开始前建议先读的文件

按顺序建议阅读：

1. [current_repo_architecture_audit_20260307.md](/mnt/data/project/jn/UniMapGen/docs/current_repo_architecture_audit_20260307.md)
2. [reproduction_status_20260306.md](/mnt/data/project/jn/UniMapGen/docs/reproduction_status_20260306.md)
3. [qwen_dinov2_map_serialization_branch.md](/mnt/data/project/jn/UniMapGen/docs/qwen_dinov2_map_serialization_branch.md)
4. [qwen_map_dataset.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_dataset.py)
5. [qwen_map_generator.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/qwen_map_generator.py)
6. [predict_qwen_state_scan.py](/mnt/data/project/jn/UniMapGen/unimapgen/predict_qwen_state_scan.py)
7. [full_reproduction_task_assignment.md](/mnt/data/project/jn/UniMapGen/docs/full_reproduction_task_assignment.md)

## 12. 一句话交接结论

后续 agent 应该基于当前已经跑通的 `Qwen + DINOv2 + serialization + local-adjacent state update` 主线，优先完成：

- 正式对齐卫星数据接入
- PV 分支接入
- 正式训练与评估闭环

而不是重新搭架构。

## 13. 2026-03-08 更新入口

这份文档之后的新进展，见：

- [agent_handoff_update_20260308.md](/mnt/data/project/jn/UniMapGen/docs/agent_handoff_update_20260308.md)

该更新文档已补充：

- partial 快照数据集构建
- baseline / state 正式训练结果
- line type 接入
- official metrics 接入
- paper-style patch augmentation 构建器
- 初始化、缓存、epoch 间等待问题的工程修复
