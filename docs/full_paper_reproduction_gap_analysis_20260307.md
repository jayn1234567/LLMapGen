# UniMapGen 完全复现差距分析（2026-03-07）

## 1. 结论先行

当前仓库已经完成的是：

- 一条可训练的 `Qwen + DINOv2 + map serialization + state update` 工程主线
- 正式对齐卫星 patch 的最小可用数据接入
- state update 的几何投影与扫描推理闭环

但如果目标是“完全复现论文”，当前距离论文完整版仍有明显缺口。

最重要的判断是：

1. 当前代码已经完成“工程复现主线”
2. 但还没有完成“论文方法、论文数据、论文训练流程、论文评估协议”的完整对齐
3. 真正的完全复现，不是只把当前 quick/full 训练继续放大，而是要补齐论文中缺失的关键模块

## 2. 对照论文后，当前还缺什么

下面按论文的 4 个核心维度拆分。

### 2.1 数据与 patch 构建层

论文明确使用：

- OpenSatMap level-20
- 原图 `4096x4096`
- 分辨率 `0.15m/pixel`
- patch 尺寸 `896x896`
- overlapped crop
- inclined crop
- rotation augmentation
- 最终把约 `2K` patch 扩到近 `700K`

当前代码状态：

- 已经开始用修复后的 `896x896` 对齐裁剪 patch
- 但训练配置仍把图像 resize 到 `224`
- 还没有完成论文式 `overlap + inclined crop + rotation` 的正式 patch 扩增
- 还没有形成论文规模的 patch 数量
- 当前训练样本仍然只是“部分对齐快照”，不是论文规模正式训练集

因此，完全复现所需新增项：

1. 正式 patch augmentation 生成器  
需要一个专门的数据准备脚本，按论文式 stride 和角度生成 `896x896` patch。

2. patch 级元数据升级  
不仅要有 `gps_center/crop_box`，还要有：
  - patch center index
  - crop stride 方案
  - rotation angle
  - patch 来源类型（原始/overlap/inclined/rotation）

3. 训练分辨率升级  
当前 `224` 只是工程可跑版本。论文复现至少要逐步升到 `896` 或接近论文输入尺度。

### 2.2 地图表示与 tokenizer 层

论文强调：

- 等距采样，间隔 `6m`
- 按首点到原点距离重排
- previous-state map 序列化
- `start/end/cut` 语义
- 坐标和词都按“整词 token”表示
- 线类别之外，还使用 line type 作为训练属性

当前代码状态：

- 已实现 `6m` 等距采样
- 已实现按首点到原点距离重排
- 已实现 `start/end/cut`
- 已实现 previous-state map 序列化
- 但当前主线只稳定使用：
  - `category`
  - `start_type`
  - `end_type`
- 还没有把论文中的 `line type` 正式纳入主线 token 监督
- 当前主线采用的是 `<x_i>/<y_i>` 风格离散 token，不是论文表述里更接近“整数字 token”的格式
- 当前常用配置仍是 `coord_num_bins=224`，而不是论文对齐方向里提到的高分辨率量化

因此，完全复现所需新增项：

1. line type token 化  
至少把论文使用的线型属性补进主线，而不是只保留三类线类别。

2. 高分辨率坐标量化  
需要把当前工程默认 `224 bins` 升级到论文对齐的高分辨率版本，并检查 seq length 与显存。

3. tokenizer 论文对齐版  
当前工程 tokenizer 可以作为 baseline 保留，但需要额外实现一版更贴近论文的“整词/整数字”向量 tokenizer。

### 2.3 模型架构层

论文核心结构是：

- BEV Encoder: DINOv2-L/14
- PV Encoder: `3DConv + Qwen2-VL-2B image encoder`
- LLM: Qwen2.5-1.5B
- flexible multi-modal input
- 任意组合：BEV / PV / Text / Previous Map
- 训练中随机 mask 模态

当前代码状态：

- 卫星主线确实已经是 `DINOv2 + Qwen2.5`
- 但当前最稳定主线实际上还是 satellite-only
- 当前 PVEncoder 还是轻量工程替代，不是论文里的 `3DConv + Qwen2-VL-ViT`
- 当前最稳定的 Qwen 主线还没有正式接入 PV
- 当前 Text prompt 也没有论文里完整的数据构建和输入形态
- 当前 `UniMapGenPaper` 还是 scaffold，不是已经收口的论文主线

因此，完全复现所需新增项：

1. 把 PV 正式并入当前 Qwen 主线  
不是停留在 `paper scaffold`，而是要让 `train_qwen_map / predict_qwen_state_scan` 真正支持：
  - satellite-only
  - PV-only
  - satellite+PV
  - satellite+PV+text+state

2. PV encoder 论文对齐  
当前 `unimapgen/models/encoders/pv_encoder.py` 需要从轻量替代版升级到论文结构，至少接口和输入组织要完全对齐。

3. PV 位姿 prompt  
论文明确把 PV frame 的 `[x,y]` 与方向角 `theta` 纳入输入。当前主线还没有这部分 prefix 设计。

4. 真正的 multi-modal masking  
虽然仓库其他分支已经有模态 mask 思路，但当前 Qwen 主线还没有做成论文式多模态统一训练。

### 2.4 状态更新与全局构图层

论文明确强调：

- `left-to-right, top-to-bottom` patch 更新顺序
- inference 时使用相邻 patch 的 `cut points`
- training 时 start/end 类型来自当前 patch GT
- 目标是直接得到 globally continuous map，而不是后处理拼接

当前代码状态：

- 已经有真实 geometry 驱动的 patch scan
- 已经有 global merge
- 已经有 local-adjacent state
- 已经实现 `cut_traces` 这种工程增强
- 但当前和论文仍有两点差异：
  - 论文强调 `cut points`，当前默认主推 `cut_traces`
  - 当前 merge/projection 仍是工程几何规则，不是已经通过论文消融验证的最终版

因此，完全复现所需新增项：

1. paper-aligned `cut_points` 分支  
当前 `cut_traces` 可以保留做工程对照，但必须补一个严格按论文 `cut points` 的正式对齐实验。

2. state training / inference 完全统一  
虽然现在已经很接近，但仍需把训练侧 state prefix 抽取逻辑和推理侧投影逻辑做成同一套正式定义。

3. state update 正式消融  
至少需要形成以下对照：
  - no state
  - cut_points
  - cut_traces
  - local adjacent vs broader history

## 3. 论文训练流程，当前还差什么

根据当前本地论文笔记和已有整理，论文不是单阶段直接训到底，而是有明显的阶段化训练思想：

1. Stage 1：地图生成预训练/稳定化
2. Stage 2：视觉-语言-地图能力对齐
3. Stage 3：state update 训练

当前代码状态：

- 已支持 `init_checkpoint`
- 已能跑 state quick/full
- 但还没有把训练流程正式拆成 paper-aligned staged pipeline

因此，完全复现所需新增项：

1. staged config 体系  
需要正式区分：
  - BEV-only pretrain
  - multi-modal alignment
  - state update finetune

2. checkpoint 衔接脚本  
不仅是手动传路径，而是明确写出 stage1 -> stage2 -> stage3 的运行脚本和产物目录。

3. 每阶段的数据与损失边界  
当前所有逻辑基本都塞在同一个 Qwen 主线里，完全复现需要把阶段职责拆清楚。

## 4. 论文评估协议，当前还差什么

论文使用的不是只有 `loss` 和 `token_acc`，还包括：

- mIoU
- Mask AP
- Chamfer AP
- 连续性/全局构图质量
- 多模态对比
- 甚至 lane topology 指标

当前代码状态：

- `eval_qwen_map.py` 还是 loss/token_acc 级别
- quick/state-scan 结果目前主要靠人工读 JSON
- 还没有实现论文式 instance-level 和 semantic-level evaluation

因此，完全复现所需新增项：

1. OpenSatMap 论文指标实现  
至少把：
  - mIoU
  - Mask AP
  - Chamfer AP
补进主线评估脚本。

2. state continuity 指标  
需要专门评估跨 patch 连续性，而不是只看 token accuracy。

3. multi-modal ablation 表格生成  
需要最终形成论文式表格，而不是只靠 quick 观察。

## 5. 当前代码里哪些可以继续沿用

不是所有东西都要推翻。下面这些可以继续作为完全复现的基础：

1. `OpenSatMapQwenDataset` 主线框架  
数据结构和 state prefix 组织已经足够接近论文方向。

2. `state_geometry.py`  
虽然还不够论文最终版，但已经是很好的几何骨架。

3. `train_qwen_map / eval_qwen_map / predict_qwen_state_scan`  
这些入口可以继续保留，只需扩容为 staged + multi-modal + paper metrics。

4. 对齐裁剪数据构建链  
AV2 + OpenSatMap 修复版裁剪链可以直接作为正式数据准备基础继续扩展。

## 6. 当前代码里哪些必须重构或新增

如果目标是完全复现，下列项不能只靠“调参”解决。

### 必须新增

1. 论文式 patch augmentation 生成器
2. 论文式 PV 数据对齐与采样管线
3. 论文式 text prompt 数据构建
4. paper-aligned tokenizer 分支
5. paper metrics 实现
6. staged training 脚本

### 必须重构

1. 当前 Qwen 主线的输入拼接接口  
需要升级到真正支持 BEV/PV/Text/State 任意组合。

2. PV encoder  
需要从轻量替代版重构到论文结构。

3. 评估体系  
需要从 token-level evaluation 升级到论文级 map construction evaluation。

## 7. 如果目标是“完全复现”，建议的正式推进顺序

### 第一阶段：把 BEV-only 论文版做完整

目标：

- 不先碰 PV
- 先完成 `OpenSatMap20 + 896 patch + 700k augmentation + paper metrics`
- 形成 BEV-only 论文对齐基线

必须完成：

1. 896 patch augmentation
2. line type token
3. 高分辨率 tokenizer
4. 正式评估指标

### 第二阶段：把 staged training 补齐

目标：

- 从当前单阶段训练改为 stage1/2/3

必须完成：

1. stage1 config
2. stage2 config
3. stage3 config
4. checkpoint 串联脚本

### 第三阶段：把 PV 分支接到当前最稳定主线

目标：

- 不在 scaffold 中停留
- 而是让 Qwen 主线真正吃到 PV

必须完成：

1. PV encoder 论文对齐
2. PV frame 位姿 prompt
3. 最多 10 张 front-view 帧采样
4. satellite+PV 联合训练

### 第四阶段：把 text prompt 任务补齐

目标：

- 复现 targeted map generation

必须完成：

1. target map paired prompt 数据
2. trace point prompt
3. text-only / BEV+text / BEV+PV+text 对照

### 第五阶段：state update 论文收口

目标：

- 在完整多模态和正式数据上做 state update

必须完成：

1. `cut_points` 正式版
2. no-state vs state 消融
3. continuity 指标
4. full-map global construction 可视化

## 8. 当前最值得立刻补的 6 件事

如果你现在要求“完全复现”而不是“继续 quick/full 工程实验”，我建议下面 6 件事按顺序启动：

1. 实现论文式 `896 + overlap + inclined + rotation` patch augmentation
2. 在 Qwen 主线中补 `line type` 和高分辨率 tokenizer
3. 给 Qwen 主线补正式 OpenSatMap 指标
4. 先做完整 BEV-only 论文版基线
5. 把 PV encoder 和 PV prompt 真正并入 Qwen 主线
6. 再把 state update 迁移到论文完整多模态链路

## 9. 一句话结论

当前仓库已经完成“工程可运行复现”，但距离“论文完全复现”还差四大块：

- 论文规模 patch 数据增强
- 论文对齐 tokenizer/属性表示
- 论文完整多模态架构（尤其 PV）
- 论文评估协议与 staged training

如果目标真的是完全复现，那么下一步最正确的方向不是继续只跑当前 quick/full config，而是把上述四大块按顺序补齐。
