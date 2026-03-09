# UniMapGen 更贴论文版本流程图（2026-03-08）

这份图不是“当前已经完全实现的代码图”，而是“按论文理解整理出的目标流程图”，并把当前仓库已经对上的部分和仍未完成的缺口同时标出来。

适合回答两个问题：

- 论文里的完整方法链条是什么
- 我们当前代码已经走到哪一步，还缺哪一步

## 1. 论文目标总流程

```mermaid
flowchart TD
    A[原始 OpenSatMap Level-20<br/>4096x4096 0.15m/pixel] --> B[Patch 构建]
    B --> B1[896x896 base patch]
    B --> B2[overlap crop]
    B --> B3[inclined crop]
    B --> B4[rotation augmentation]

    B1 --> C[Patch 级 GT]
    B2 --> C
    B3 --> C
    B4 --> C

    C --> D[Map Serialization]
    D --> D1[6m 等距采样]
    D --> D2[按首点到原点距离排序]
    D --> D3[start end cut 端点语义]
    D --> D4[line category]
    D --> D5[line type]
    D --> D6[高分辨率坐标量化]

    E[输入模态] --> E1[BEV satellite]
    E --> E2[PV frames]
    E --> E3[Text prompt]
    E --> E4[Previous map state]

    E1 --> F[多模态 UniMapGen]
    E2 --> F
    E3 --> F
    E4 --> F

    F --> G[Stage 1<br/>地图生成预训练]
    G --> H[Stage 2<br/>视觉 语言 地图对齐]
    H --> I[Stage 3<br/>State Update 全局增量构图]

    I --> J[自回归生成 map tokens]
    J --> K[decode 成 polyline map]
    K --> L[official metrics<br/>mIoU Mask AP Chamfer AP]
```

## 2. 论文架构视角

```mermaid
flowchart LR
    A1[BEV Encoder<br/>DINOv2-L/14] --> Z[Qwen2.5-1.5B]
    A2[PV Encoder<br/>3DConv + Qwen2-VL image encoder] --> Z
    A3[Text Prompt Encoder<br/>Qwen tokenizer path] --> Z
    A4[Previous Map State<br/>serialized previous G_n-1] --> Z

    Z --> B[自回归生成当前 patch map tokens]
    B --> C[Map Decoder]
    C --> D[当前 patch 局部地图]
    D --> E[global merge]
    E --> F[得到连续全局地图 G_n]
```

## 3. 论文中的 state update 逻辑

```mermaid
flowchart TD
    A[按 left-to-right top-to-bottom 扫描 patch] --> B[当前 patch P_n]
    C[上一时刻或已生成的全局地图 G_n-1] --> D[抽取与 P_n 相邻的 previous state]
    D --> D1[论文更强调 cut_points]

    B --> E[当前观测]
    D1 --> F[UniMapGen]
    E --> F

    F --> G[预测当前 patch map]
    G --> H[与历史全局图 merge]
    H --> I[更新得到 G_n]
    I --> J[供下一个 patch 使用]
```

## 4. 当前仓库对论文的对应关系

```mermaid
flowchart TD
    A[论文模块] --> B[当前对应实现]

    B --> B1[BEV Encoder<br/>已对上<br/>DINOv2 + Qwen 主线]
    B --> B2[Map Serialization 骨架<br/>已基本对上]
    B --> B3[State 几何 scan/merge/projection<br/>已工程打通]
    B --> B4[official metrics<br/>已接入]
    B --> B5[paper-style patch augmentation builder<br/>脚手架已完成]

    C[仍未完全对齐] --> C1[PV Encoder 论文版]
    C --> C2[Text prompt 全量论文路径]
    C --> C3[高分辨率 10k 级坐标量化]
    C --> C4[严格论文式 cut_points state]
    C --> C5[stage1 到 stage3 正式衔接]
    C --> C6[full paper-scale augmentation 数据集]
    C --> C7[最终完整论文表格]
```

## 5. 当前代码与论文差距图

```mermaid
flowchart LR
    A[当前已完成] --> A1[partial AV2-aligned OpenSatMap 快照]
    A --> A2[baseline 正式训练与评估]
    A --> A3[state 正式训练与评估]
    A --> A4[line_type 接入]
    A --> A5[official metrics]
    A --> A6[augmentation builder]

    B[论文要求] --> B1[full OpenSatMap patch expansion]
    B --> B2[近 700k patch 量级]
    B --> B3[stage1 stage2 stage3]
    B --> B4[BEV PV Text State 任意组合]
    B --> B5[论文式连续全局构图性能]

    A6 --> C[当前状态: 工程主线可运行]
    B5 --> D[目标状态: 完整论文复现]
```

## 6. 推荐的论文对齐推进顺序

```mermaid
flowchart TD
    A[Step 1<br/>full paper augmentation] --> B[Step 2<br/>stage1 BEV-only 稳定训练]
    B --> C[Step 3<br/>补高分辨率 tokenizer 与 cut_points]
    C --> D[Step 4<br/>stage2 对齐训练]
    D --> E[Step 5<br/>stage3 state update 正式训练]
    E --> F[Step 6<br/>PV Text 接入当前主线]
    F --> G[Step 7<br/>完整论文指标表]
```

## 7. 当前最现实的理解

```mermaid
flowchart TD
    A[当前 baseline] --> A1[已经学起来]
    B[当前 state] --> B1[工程闭环通了]
    B --> B2[但性能暂时弱于 baseline]

    C[因此下一步不是盲目继续加训] --> C1[先补 full augmentation]
    C --> C2[再补 paper-aligned tokenizer/state]
    C --> C3[再进入 staged training]
```

## 8. 和当前代码文档怎么配合看

建议配合下面三份文档一起看：

- [current_model_flowchart_20260308.md](/mnt/data/project/jn/UniMapGen/docs/current_model_flowchart_20260308.md)
- [agent_handoff_update_20260308.md](/mnt/data/project/jn/UniMapGen/docs/agent_handoff_update_20260308.md)
- [full_paper_reproduction_gap_analysis_20260307.md](/mnt/data/project/jn/UniMapGen/docs/full_paper_reproduction_gap_analysis_20260307.md)

如果只想快速理解：

1. 先看这份文档，理解论文目标链。
2. 再看 `current_model_flowchart_20260308.md`，理解当前实际跑通链。
3. 最后看 handoff 文档，确定下一步该做什么。
