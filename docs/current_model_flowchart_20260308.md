# UniMapGen 当前模型流程图（2026-03-08）

这份文档把当前已经实际跑通的复现主线画成流程图，方便快速理解：

- 数据是怎么进入模型的
- `baseline` 和 `state` 两条训练/评估链怎么闭环
- 当前已经完成了什么
- 离“完全复现论文”还差什么

## 1. 当前主线总览

```mermaid
flowchart TD
    A[动态裁剪源目录<br/>satellite_tools/av2_opensatmap_crops_paper896_fix] --> B[partial 快照构建<br/>build_av2_opensatmap_partial_dataset.py]
    B --> C[data_samples/av2_opensatmap_partial_fix<br/>train val annotations splits geometry]

    C --> D[QwenDataset<br/>qwen_map_dataset.py]
    D --> E[GT 序列化<br/>serialization.py]
    D --> F[state 几何投影<br/>state_geometry.py]

    E --> G[QwenMapTokenizer]
    F --> G

    C --> H[卫星图 patch]
    H --> I[DINOv2 Satellite Encoder]

    G --> J[QwenSatelliteMapGenerator]
    I --> J

    J --> K[train_qwen_map.py]
    K --> L[checkpoint<br/>best.pt]

    L --> M[eval_qwen_map.py]
    L --> N[predict_qwen_state_scan.py]
    N --> O[eval_opensatmap_official.py]
```

## 2. 数据与模型细化

```mermaid
flowchart LR
    A1[卫星图 PNG<br/>896 crop] --> A2[resize 到训练分辨率<br/>当前主线常用 224]
    A2 --> A3[DINOv2 提取视觉 token]

    B1[GT JSON<br/>category line_type points] --> B2[6m 重采样]
    B2 --> B3[折线排序与端点语义<br/>start end cut]
    B3 --> B4[坐标离散 token 化]
    B4 --> B5[Map token 序列]

    C1[patch_geometry.json] --> C2[按 scan 顺序组织 patch]
    C2 --> C3[global merge]
    C3 --> C4[投影回当前 patch 的 previous state]

    B5 --> D[QwenMapTokenizer]
    C4 --> D
    A3 --> E[QwenSatelliteMapGenerator]
    D --> E

    E --> F[预测 map token 序列]
    F --> G[decode 回 polyline]
```

## 3. baseline 与 state 两条实验链

```mermaid
flowchart TB
    A[data_samples/av2_opensatmap_partial_fix]

    A --> B1[baseline 配置<br/>qwen_dinov2_map_serialization_av2_partial.yaml]
    A --> B2[state 配置<br/>qwen_dinov2_map_serialization_av2_partial_state.yaml]
    A --> B3[state + line_type 配置<br/>..._state_line_type.yaml]

    B1 --> C1[run_qwen_dinov2_map_serialization_av2_partial.sh]
    B2 --> C2[run_qwen_dinov2_map_serialization_av2_partial_state.sh]
    B3 --> C3[run_qwen_dinov2_map_serialization_av2_partial_state_line_type.sh]

    C1 --> D1[baseline best.pt]
    C2 --> D2[state best.pt]
    C3 --> D3[state + line_type best.pt]

    D1 --> E1[eval_qwen_map]
    D1 --> F1[state_scan 推理]
    F1 --> G1[official metrics]

    D2 --> E2[eval_qwen_map]
    D2 --> F2[state_scan 推理]
    F2 --> G2[official metrics]

    D3 --> E3[eval_qwen_map]
    D3 --> F3[state_scan 推理]
    F3 --> G3[official metrics]
```

## 4. 当前复现状态图

```mermaid
flowchart TD
    A[已完成的复现工作]

    A --> B1[partial 快照数据集]
    A --> B2[baseline 正式训练]
    A --> B3[baseline 正式评估闭环]
    A --> B4[state 正式训练]
    A --> B5[state 正式评估闭环]
    A --> B6[line_type 接入]
    A --> B7[official metrics 接入]
    A --> B8[paper-style augmentation 脚手架]
    A --> B9[缓存与日志等工程优化]

    C[当前结论]
    C --> C1[工程链路已经跑通]
    C --> C2[baseline 已经学起来]
    C --> C3[state 当前版本弱于 baseline]
    C --> C4[official metrics 也支持同样结论]

    D[未完成的论文对齐工作]
    D --> D1[full paper-scale augmentation]
    D --> D2[更 paper-aligned 的 cut_points state]
    D --> D3[更高分辨率坐标量化 tokenizer]
    D --> D4[stage1 到 stage3 正式训练链]
    D --> D5[PV 分支接入当前 Qwen 主线]
    D --> D6[完整数据上的最终论文指标表]
```

## 5. 当前最重要的实验结论

```mermaid
flowchart LR
    A[baseline 正式版] --> A1[val_loss 约 2.160]
    A --> A2[val_token_acc 约 0.565]

    B[state 正式版] --> B1[val_loss 约 2.377]
    B --> B2[val_token_acc 约 0.550]
    B --> B3[mIoU 约 0.00194]
    B --> B4[APM 0]

    A1 --> C[当前 baseline 优于 state]
    B1 --> C
    A2 --> C
    B2 --> C
    B3 --> C
    B4 --> C
```

## 6. 建议如何用这张图

如果你现在要继续推进复现，建议按下面顺序读图：

1. 先看“当前主线总览”，确认训练和评估闭环已经打通。
2. 再看“baseline 与 state 两条实验链”，明确当前对比是怎么做的。
3. 最后看“当前复现状态图”，判断下一步应该补哪一块论文缺口。

相关交接文档：

- [agent_handoff_20260307.md](/mnt/data/project/jn/UniMapGen/docs/agent_handoff_20260307.md)
- [agent_handoff_update_20260308.md](/mnt/data/project/jn/UniMapGen/docs/agent_handoff_update_20260308.md)
- [paper_aligned_model_flowchart_20260308.md](/mnt/data/project/jn/UniMapGen/docs/paper_aligned_model_flowchart_20260308.md)
- [serialization_and_state_update_flow_20260308.md](/mnt/data/project/jn/UniMapGen/docs/serialization_and_state_update_flow_20260308.md)
- [finetune_dataset_and_prompt_flow_20260308.md](/mnt/data/project/jn/UniMapGen/docs/finetune_dataset_and_prompt_flow_20260308.md)
- [model_architecture_intro_20260308.md](/mnt/data/project/jn/UniMapGen/docs/model_architecture_intro_20260308.md)
