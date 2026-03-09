# UniMapGen 后续复现路线（2026-03-07）

## 1. 这份路线解决什么问题

这份文档用于在已有交接文档基础上，继续推进 `/mnt/data/project/jn/UniMapGen` 的未完成复现工作。

当前新增前提是：

- 你已经准备了 AV2 + OpenSatMap 对齐裁剪数据的一部分
- 全量裁剪还没完成
- 现在希望先用已经裁好的这部分数据，把正式训练主线继续往前推

因此，这一轮的目标不是追论文最终指标，而是先完成：

1. 把“修复后的部分对齐数据”接进当前最成熟主线
2. 跑通一条正式的小规模训练闭环
3. 再把 state update 和后续扩容路线接上

## 2. 当前我对项目状态的理解

结合交接文档、现有代码和当前数据目录，当前判断如下。

### 2.1 最应该继续的主线

当前最成熟的不是 `UniMapGenPaper`，而是：

`Qwen + DINOv2 + map serialization + local-adjacent state update`

对应链路：

- `unimapgen/data/qwen_map_dataset.py`
- `unimapgen/data/serialization.py`
- `unimapgen/models/qwen_map_generator.py`
- `unimapgen/train_qwen_map.py`
- `unimapgen/predict_qwen_state_scan.py`

### 2.2 当前已经具备的条件

- Qwen 主线 smoke 已跑通
- state-scan 推理已跑通
- DINOv2 lane segmentation 验证分支已跑通
- 本地 Qwen2.5-1.5B 和 DINOv2-L 权重已就位
- 修复后的对齐裁剪目录已经存在：`/mnt/data/project/jn/satellite_tools/av2_opensatmap_crops_paper896_fix`

### 2.3 当前部分对齐数据的真实状态

当前可直接看到：

- 卫星图目录：`/mnt/data/project/jn/satellite_tools/av2_opensatmap_crops_paper896_fix/satellite`
- 单样本真值目录：`/mnt/data/project/jn/satellite_tools/av2_opensatmap_crops_paper896_fix/ground_truth`

我本轮检查到的现状：

- `satellite/` 下约有 `1327` 张图
- `ground_truth/` 下约有 `1326` 个 JSON
- 单样本 JSON 已包含 `image_width / image_height / lines / source_image / crop_box / gps_center`
- 但目录根下还没有训练入口直接需要的聚合文件：
  - `annotations.json`
  - `splits_meta.json`
  - `patch_geometry.json`
  - `manifest.json` 或 `summary.json`

这里要特别注意一件事：

- `av2_opensatmap_crops_paper896_fix` 目前是动态目录，不是静态数据集快照
- 因为裁剪脚本还在持续运行，里面的卫星图和标注数量会继续变化
- 所以任何数量统计都只能理解为“某一时刻的快照”，不能当成最终规模
- 训练不应直接依赖这个动态目录，而应先导出一个固定快照数据根目录

这意味着：

- 这批数据已经足够支持“先训一版”
- 但还不能直接喂给当前 `qwen_map_dataset.py`
- 第一优先级必须先做“数据整理成 UniMapGen 可读格式的稳定快照”

## 3. 当前最合理的推进原则

后续继续时，建议严格遵守下面三条。

1. 不重写架构  
当前主线已经真实跑通，应继续在 `Qwen + DINOv2 + serialization + state update` 上接正式数据。

2. 先做“小规模正式闭环”，再追“完整论文规模”  
你现在已经有部分对齐数据，最划算的是先把这部分变成一条稳定训练线，而不是等全量裁剪完成才开始。

3. 先卫星主线收敛，再补 PV  
当前最缺的不是多模态花样，而是正式对齐数据上的训练与 state 几何闭环。

## 4. 推荐的后续复现顺序

### 阶段 A：把部分对齐数据整理成当前主线可读格式

这是第一优先级，也是当前唯一不能跳过的阶段。

目标：

- 从 `av2_opensatmap_crops_paper896_fix` 生成一个“UniMapGen 可直接训练”的稳定快照数据根目录

建议整理后的最小结构：

```text
<aligned_partial_root>/
├── train/
├── val/
├── annotations.json
├── splits_meta.json
└── patch_geometry.json
```

其中：

- `train/` 和 `val/` 中放 `*_satellite.png`
- `annotations.json` 是聚合后的字典，key 为图片名，value 至少含：
  - `lines`
  - `image_width`
  - `image_height`
- `splits_meta.json` 至少要有：
  - `train_tokens`
  - `val_tokens`
- `patch_geometry.json` 至少要能让 `load_patch_geometry_map(...)` 正常读取，建议每个 patch 记录：
  - `sample_token`
  - `gps_center: [lon, lat]`
  - `image_width`
  - `image_height`
  - `crop_region.x_min/y_min/x_max/y_max`
  - `crop_region.original_image`

这一阶段要顺手处理的两个问题：

1. `1327` 张图和 `1326` 个标注文件不一致  
需要显式审计缺失样本，决定是剔除、保留为空标注，还是重新导出。

2. patch 扫描顺序不能再只靠文件名  
既然这批数据已经有 `gps_center` 和 `crop_box`，就应该正式生成 `patch_geometry.json`，让 state-scan 使用真实几何顺序。

这一阶段还要明确一个操作纪律：

- 源裁剪目录继续增长时，不要让训练直接读取它
- 每次准备启动一轮训练前，先重新构建一次快照目录
- 一旦某轮训练开始，就固定使用那一版快照，不要在训练中途混入新增样本

阶段验收标准：

- `python -m unimapgen.check_data --config <新配置>` 可以通过
- `build_qwen_map_dataset(...)` 能成功加载 train / val
- `use_state_update=false` 与 `use_state_update=true` 都能读到这批数据

### 阶段 B：先做“无 state”的正式小规模基线

这一步的目的不是最终效果，而是先隔离“数据是否可训练”。

建议做法：

- 基于 `configs/qwen_dinov2_map_serialization_smoke.yaml` 新建一份“aligned partial baseline”配置
- 先不开 `state_update`
- 先保持当前成熟超参，不急着加复杂增强

建议首轮配置原则：

- `image_size` 先保持 `224`
- `freeze_satellite: true`
- `freeze_llm: false`
- `batch_size` 先从显存能承受的最小稳定值开始
- `epochs` 明显高于 smoke，但先不要一口气拉满论文规模
- 类别先继续使用：
  - `curb`
  - `lane_line`
  - `virtual_line`

这一阶段的目标是回答三个问题：

1. 部分对齐数据能否稳定收敛到比 smoke 更合理的 token accuracy
2. 模型是否开始生成更长、更完整的 polyline
3. OpenSatMap GT 作为卫星监督时，序列化输出是否明显优于未对齐样例集

建议命令形式：

```bash
cd /mnt/data/project/jn/UniMapGen
python -m unimapgen.check_data --config configs/qwen_dinov2_map_serialization_av2_partial.yaml
python -m unimapgen.train_qwen_map --config configs/qwen_dinov2_map_serialization_av2_partial.yaml
python -m unimapgen.eval_qwen_map --config configs/qwen_dinov2_map_serialization_av2_partial.yaml --checkpoint outputs/qwen_dinov2_map_serialization_av2_partial/best.pt
python -m unimapgen.predict_qwen_map --config configs/qwen_dinov2_map_serialization_av2_partial.yaml --checkpoint outputs/qwen_dinov2_map_serialization_av2_partial/best.pt
```

阶段验收标准：

- 训练和验证闭环跑通
- 预测结果不再主要是极短碎线
- 可以明确判断“这批部分对齐数据是有效监督，而不是仅能跑通”

### 阶段 C：把 state update 切到正式几何

只有在阶段 B 确认数据本身可训练后，再进入这一阶段。

建议做法：

- 基于 `configs/qwen_dinov2_map_serialization_state_smoke.yaml` 新建 “aligned partial state” 配置
- 把 `patch_geometry_json` 指向新生成的几何文件
- 默认继续保留：
  - `state_update_mode: patch_scan`
  - `state_prefix_mode: cut_traces`

这一步的关键不是单 patch loss，而是验证三件事：

1. `build_patch_scan_order(...)` 是否真的按几何顺序扫描 patch
2. `build_state_lines_from_global(...)` 是否能从真实相邻 patch 提取局部状态
3. `predict_qwen_state_scan.py` 在这批部分正式数据上，是否比“无 state”更稳定地维持跨 patch 连续性

建议命令形式：

```bash
cd /mnt/data/project/jn/UniMapGen
python -m unimapgen.train_qwen_map --config configs/qwen_dinov2_map_serialization_av2_partial_state.yaml
python -m unimapgen.predict_qwen_state_scan \
  --config configs/qwen_dinov2_map_serialization_av2_partial_state.yaml \
  --checkpoint outputs/qwen_dinov2_map_serialization_av2_partial_state/best.pt \
  --split val \
  --output outputs/qwen_dinov2_map_serialization_av2_partial_state/predictions_state_scan.json
```

阶段验收标准：

- state-scan 不再依赖 `splits_meta` 近似顺序，而是吃真实 geometry
- 输出里 `num_projected_lines / num_state_lines / num_global_lines` 有意义增长
- 可视化检查时，跨 patch 断裂比无 state 更少

### 阶段 D：继续补齐剩余裁剪，并加入论文式 patch 扩增

当阶段 B 和 C 都站稳后，再进入数据规模扩张。

这一阶段建议做的事：

1. 继续完成全量 AV2 + OpenSatMap 修复版裁剪
2. 把采样单位从“过密 pose”收敛到“按 lidar 或目标时间戳采样”
3. 在对齐裁剪之后，再增加：
  - overlap
  - inclined crop
  - rotation
4. 保持基础 patch 和增强 patch 的元信息可追溯，不要混成不可回滚的数据池

注意：

- 当前 `crop_opensatmap_for_av2.py` 已完成“对齐中心裁剪”
- 论文中的 `overlapped + inclined crop + rotation` 仍应视为下一层数据构建任务
- 这层增强不应先于阶段 B/C

### 阶段 E：最后再接 PV 分支

PV 仍然是后续必须做的，但不应该在当前立刻抢第一优先级。

推荐前置条件：

- 卫星对齐数据训练已稳定
- state update 在正式 geometry 上已经成立
- 已经拿到一个比 smoke 更可信的卫星-only 基线

然后再做：

1. 把 PV encoder 接到当前 Qwen serialization 主线
2. 形成 `satellite + state + PV` 的最小闭环
3. 再做多模态消融，而不是一开始就全开

## 5. 接下来最值得立刻执行的 5 件事

如果只看“你现在就该做什么”，建议顺序如下。

1. 写一个数据整理脚本  
把 `av2_opensatmap_crops_paper896_fix/{satellite,ground_truth}` 组装成 `train/val + annotations.json + splits_meta.json + patch_geometry.json`。

2. 跑一次 `check_data`  
先确认路径、类别、空标注、越界点和 split 是否正常。

3. 新建一份“aligned partial baseline”配置  
先用无 state 版本训练一轮。

4. 在这份部分数据上拿到第一版正式基线  
至少完成 `train + eval + predict`。

5. 再新建“aligned partial state”配置  
用真实 `patch_geometry.json` 跑 `predict_qwen_state_scan`，验证局部邻接 state 是否继续成立。

## 6. 当前不建议做的事

1. 不要先把 PV 分支作为第一优先级  
这会掩盖当前真正的问题，到时候你很难分清是数据问题、几何问题还是多模态问题。

2. 不要在数据还没整理成标准格式前直接改训练主线  
当前训练主线接口已经足够清楚，优先适配数据，不要反过来为临时目录结构改核心代码。

3. 不要跳过“无 state 基线”直接做 state  
否则一旦效果不好，很难判断是 state 机制问题，还是基础数据监督本身没立住。

4. 不要过早追论文规模和最终指标  
当前最重要的是先把“正式部分数据上的可重复闭环”建立起来。

## 7. 一句话执行结论

下一步最正确的动作不是重构模型，而是先把 `av2_opensatmap_crops_paper896_fix` 这批已裁好的修复数据整理成当前 Qwen 主线可直接读取的标准数据集格式，然后按“无 state 基线 -> state 几何闭环 -> 扩大全量裁剪 -> 最后接 PV”的顺序继续复现。
