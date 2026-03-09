# 当前微调数据集与 Prompt 流程图（2026-03-08）

这份图只回答一个问题：

`当前训练时，模型到底看到了什么数据、什么 prompt、什么 target。`

对应代码和配置：

- [qwen_map_dataset.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_dataset.py)
- [qwen_map_tokenizer.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_tokenizer.py)
- [qwen_dinov2_map_serialization_av2_partial.yaml](/mnt/data/project/jn/UniMapGen/configs/qwen_dinov2_map_serialization_av2_partial.yaml)
- [qwen_dinov2_map_serialization_av2_partial_state.yaml](/mnt/data/project/jn/UniMapGen/configs/qwen_dinov2_map_serialization_av2_partial_state.yaml)

## 1. 当前单样本总流程

```mermaid
flowchart TD
    A[train 或 val 中的一张 patch 图像] --> B[读取 image]
    A1[annotations.json 中对应 token 的 GT lines] --> C[序列化成 map_token_ids]
    A2[patch_geometry.json 和历史 patch] --> D[构造 state_token_ids]
    A3[prompt_template] --> E[生成 prompt_text]

    B --> F[单样本 sample]
    C --> F
    D --> F
    E --> F

    F --> G[QwenMapCollator]
    G --> H[prompt_input_ids]
    G --> I[state_input_ids]
    G --> J[map_input_ids]
    G --> K[gt_map_token_ids]
    G --> L[image tensor]
```

## 2. 当前 sample 的字段长什么样

```mermaid
flowchart LR
    A[sample] --> A1[image<br/>224x224 RGB tensor]
    A --> A2[prompt_text<br/>纯文本指令]
    A --> A3[state_token_ids<br/>previous map prefix]
    A --> A4[map_token_ids<br/>当前 patch GT token]
    A --> A5[lines<br/>结构化 polyline GT]
```

当前 dataset 返回的就是这五类核心信息。

## 3. baseline 数据样本怎么构成

```mermaid
flowchart TD
    A[patch PNG] --> B[resize 到 224]
    C[GT lines] --> D[serialize_opensatmap_lines]
    D --> E[map_token_ids]
    F[baseline prompt_template] --> G[prompt_text]
    H[no state] --> I[state_token_ids = <state>]

    B --> J[baseline sample]
    E --> J
    G --> J
    I --> J
```

baseline 的核心特点：

- 有卫星图
- 有文字任务指令
- 没有真实 previous state
- `state_token_ids` 只有一个 `<state>`

## 4. state 数据样本怎么构成

```mermaid
flowchart TD
    A[patch PNG] --> B[resize 到 224]
    C[GT lines] --> D[serialize_opensatmap_lines]
    D --> E[map_token_ids]

    F[历史 patch 的 GT 几何链] --> G[patch_scan + global merge]
    G --> H[投影到当前 patch]
    H --> I[state_lines]
    I --> J[过滤为 cut_traces 或 cut_points]
    J --> K[state_token_ids]

    L[state prompt_template] --> M[prompt_text]

    B --> N[state sample]
    E --> N
    K --> N
    M --> N
```

state 的核心特点：

- 当前 patch 的 target 仍然是当前 GT
- 但输入里多了 previous map state prefix
- 这个 prefix 不是拍脑袋拼的，而是由历史 patch GT 通过几何投影得到

## 5. baseline prompt 现在是什么

```mermaid
flowchart LR
    A[prompt_template] --> B[填 categories]
    B --> C[填 max_lines]
    C --> D[填 max_points_per_line]
    D --> E[最终 baseline prompt_text]
```

当前 baseline 配置里的 prompt 语义是：

```text
You are given a satellite image embedding.
Generate the serialized vector map using only reserved map tokens.
Represent curb, lane line, virtual line.
Output at most 48 polylines and at most 24 points per polyline.
```

## 6. state prompt 现在是什么

```mermaid
flowchart TD
    A[state prompt_template] --> B[填 state_instruction]
    B --> C[填 state_prefix_mode]
    C --> D[填 state_prefix_count]
    D --> E[填 categories]
    E --> F[填 max_lines 和 max_points_per_line]
    F --> G[最终 state prompt_text]
```

其中 `state_instruction` 是动态变化的：

```mermaid
flowchart LR
    A{当前 patch 是否有 previous state} -->|有| B[告诉模型: 前面提供了 previous map state<br/>cut 端点表示跨 patch 连通关系<br/>不要重复生成已有段]
    A -->|没有| C[告诉模型: 当前 patch 没有 previous state<br/>从新地图开始生成]
```

当前正式 state 配置还会额外显式告诉模型：

- `state_prefix_mode = cut_traces`
- `state_prefix_count = 当前 prefix primitive 数量`

## 7. line_type 版 prompt 的额外内容

```mermaid
flowchart LR
    A[如果配置启用 line_types] --> B[追加 line_type_instruction]
    B --> C[要求每条 polyline 额外输出 line type]
```

追加的语义大致是：

```text
For each polyline, also output its line type using one of:
solid, thick solid, dashed, short dashed, others.
```

## 8. prompt 和 map token 如何进入 Qwen tokenizer

```mermaid
flowchart TD
    A[prompt_text] --> B[Qwen tokenizer.encode]
    C[state_token_ids] --> D[自定义 map token -> Qwen vocab id]
    E[map_token_ids] --> F[自定义 map token -> Qwen vocab id]

    B --> G[prompt_input_ids]
    D --> H[state_input_ids]
    F --> I[map_input_ids]
```

这里要注意：

- `prompt_text` 走的是普通 Qwen 文本 tokenizer
- `state_token_ids` 和 `map_token_ids` 先是自定义 map token，再映射到 Qwen 扩展词表

## 9. collator 最终打包成什么

```mermaid
flowchart TD
    A[batch samples] --> B[stack images]
    A --> C[pad prompt_input_ids]
    A --> D[pad state_input_ids]
    A --> E[pad map_input_ids]
    A --> F[保留 gt_map_token_ids]
    A --> G[保留 gt_state_token_ids]

    B --> H[训练 batch]
    C --> H
    D --> H
    E --> H
    F --> H
    G --> H
```

最终 batch 里最关键的字段是：

- `image`
- `prompt_input_ids`
- `prompt_attention_mask`
- `state_input_ids`
- `state_attention_mask`
- `map_input_ids`
- `map_attention_mask`
- `gt_map_token_ids`

## 10. 当前微调任务可以怎么理解

```mermaid
flowchart LR
    A[卫星图特征] --> Z[QwenSatelliteMapGenerator]
    B[文本任务指令] --> Z
    C[previous state token prefix] --> Z
    Z --> D[生成当前 patch map token 序列]
```

也就是说，当前并不是纯图像到地图，也不是纯语言到地图，而是：

`视觉条件 + 文本指令 + 可选 previous state -> 当前 patch 序列化地图`

## 11. 当前这套 prompt / 数据集的现实特点

```mermaid
flowchart TD
    A[优点] --> A1[工程简单]
    A --> A2[可稳定训练]
    A --> A3[便于 baseline 和 state 成对比较]

    B[限制] --> B1[prompt 仍然比较短]
    B --> B2[不是完整论文式多模态 prompt]
    B --> B3[当前主线仍以 satellite-only 为主]
    B --> B4[state prefix 目前默认是 cut_traces 而非更论文式 cut_points]
```

## 12. 一句话总结

当前微调样本本质上就是：

`一张卫星图 + 一段任务指令 + 一个 previous map state 前缀 + 当前 patch 的地图 token 监督`

相关文档：

- [serialization_and_state_update_flow_20260308.md](/mnt/data/project/jn/UniMapGen/docs/serialization_and_state_update_flow_20260308.md)
- [current_model_flowchart_20260308.md](/mnt/data/project/jn/UniMapGen/docs/current_model_flowchart_20260308.md)
- [paper_aligned_model_flowchart_20260308.md](/mnt/data/project/jn/UniMapGen/docs/paper_aligned_model_flowchart_20260308.md)
