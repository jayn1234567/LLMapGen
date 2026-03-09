# 序列化与 State Update 具体流程图（2026-03-08）

这份文档专门解释当前仓库里两条最关键的实现链：

- GT 是怎么被序列化成 map token 的
- previous state 是怎么从历史 patch 构出来，再喂给当前 patch 的

对应代码主要在：

- [serialization.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/serialization.py)
- [qwen_map_dataset.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_dataset.py)
- [state_geometry.py](/mnt/data/project/jn/UniMapGen/unimapgen/state_geometry.py)

## 1. GT 序列化总流程

```mermaid
flowchart TD
    A[原始 GT JSON 中的 lines] --> B[category 规范化]
    B --> C[line_type 规范化]
    C --> D[原始点坐标按 src_w src_h 缩放到训练 image_size]
    D --> E[按 interval_meter 重采样]
    E --> F[判断首尾是否贴边]
    F --> G[赋 start_type 和 end_type]
    G --> H[按首点到原点距离排序]
    H --> I[截断到 max_lines]
    I --> J[得到 serialized lines]
    J --> K[encode_lines]
    K --> L[map token 序列<br/>bos ... eos]
```

## 2. 单条 polyline 怎么变成 token

```mermaid
flowchart LR
    A[一条线] --> B[<line>]
    B --> C[<cat_xxx>]
    C --> D[可选 <lt_xxx>]
    D --> E[<s_start> 或 <s_cut>]
    E --> F[<e_end> 或 <e_cut>]
    F --> G[<pts>]
    G --> H[点1 -> <x_i> <y_j>]
    H --> I[点2 -> <x_i> <y_j>]
    I --> J[...]
    J --> K[<eol>]
```

当前实现里的顺序就是：

1. `<line>`
2. `category token`
3. 可选 `line_type token`
4. `start_type token`
5. `end_type token`
6. `<pts>`
7. 一串 `<x_*> <y_*>`
8. `<eol>`

所有线外层再包：

1. `<bos>`
2. 多条 line 序列
3. `<eos>`

## 3. 当前序列化里每一步具体做什么

```mermaid
flowchart TD
    A[raw_lines] --> B{category 是否在训练类别里}
    B -- 否 --> Bx[丢弃]
    B -- 是 --> C[normalize_opensatmap_category]

    C --> D[normalize_line_type]
    D --> E[points 从原图像素坐标缩放到 image_size]
    E --> F[_resample_polyline]
    F --> G{重采样后点数是否至少 2}
    G -- 否 --> Gx[丢弃]
    G -- 是 --> H[_is_border_point 检查首尾]

    H --> I[start_type = start 或 cut]
    H --> J[end_type = end 或 cut]
    I --> K[写入 line dict]
    J --> K

    K --> L[按首点距离排序]
    L --> M[限制 max_lines]
    M --> N[MapSequenceTokenizer.encode_lines]
    N --> O[坐标量化到 x bins / y bins]
    O --> P[最终 token ids]
```

## 4. 数据集初始化时怎么准备 state

```mermaid
flowchart TD
    A[OpenSatMapQwenDataset.__init__] --> B[读取 annotations.json]
    B --> C[生成 items]
    C --> D[对每个 item 先做一遍 serialize_opensatmap_lines]
    D --> E[得到 lines_by_token]

    E --> F[读取 patch_geometry.json]
    F --> G[得到 geom_map]

    G --> H[_load_or_build_state_lines_by_token]
    H --> I{是否已有 state_lines cache}
    I -- 是 --> J[直接加载 cache]
    I -- 否 --> K[_build_state_lines_by_token]
```

## 5. patch_scan 模式下的 state update 构建链

```mermaid
flowchart TD
    A[按 split 顺序遍历 token] --> B[当前 patch token]
    B --> C[查 geom_rec]
    C --> D[从当前已有 global_lines 中构建 state_lines]
    D --> E[把当前 patch 自己的 GT lines 投到全局坐标]
    E --> F[merge_global_lines]
    F --> G[更新 global_lines]
    G --> H[进入下一个 patch]
```

这里要注意：

- `state_lines` 来自“之前已经扫过 patch 的历史全局图”
- `global_lines` 的更新使用的是当前 patch 的 GT，而不是模型预测
- 所以训练侧的 previous state 是 teacher-forced 的几何历史

## 6. 从 global_lines 到当前 patch 的 state_lines

```mermaid
flowchart TD
    A[global_lines] --> B[select_adjacent_global_lines]
    B --> C[筛出与当前 patch 邻近的候选线]

    C --> D[project_global_lines_to_patch]
    D --> E[把候选线投影回当前 patch 像素坐标]
    E --> F[判断投影线的 start cut / end cut]

    C --> G[project_global_cut_traces_to_patch]
    G --> H[额外抽取靠近边界的 cut traces]

    F --> I[projected lines]
    H --> J[endpoint primitives]

    I --> K[去重]
    J --> K
    K --> L[限制 max_lines]
    L --> M[state_lines]
```

## 7. select_adjacent_global_lines 具体筛选逻辑

```mermaid
flowchart LR
    A[每条 global line] --> B{是否与当前 crop 来自同一原始大图}
    B -- 是 --> C[检查 points_source_px 是否落在 crop 窗口附近]
    B -- 否 --> D[检查 points_global_xy 到 patch center 的距离]
    C --> E[保留或丢弃]
    D --> E
```

当前实现用了两种邻近性：

- source image 像素窗口邻近
- 全局米制坐标的中心距离邻近

## 8. 当前 patch GT 怎么并回 global_lines

```mermaid
flowchart TD
    A[当前 patch 的 serialized lines] --> B[patch_lines_to_global]
    B --> C[像素坐标 -> patch local meters]
    C --> D[patch local -> global XY]
    D --> E[附带 source_image 和 points_source_px]
    E --> F[merge_global_lines]
    F --> G{是否能和已有 global line 端点连接}
    G -- 能 --> H[_attach_or_merge_line]
    G -- 不能 --> I[直接追加新 global line]
    H --> J[得到更新后的 global_lines]
    I --> J
```

## 9. 当前 state prefix 怎么从 state_lines 变成 token

```mermaid
flowchart TD
    A[state_lines] --> B[filter_state_prefix_lines]
    B --> C{prefix_mode}

    C --> C1[all]
    C --> C2[cut_only]
    C --> C3[cut_traces]
    C --> C4[cut_points]

    C1 --> D[保留全部 state lines]
    C2 --> E[只保留带 cut 的线]
    C3 --> F[取 cut 端附近前/后若干点]
    C4 --> G[只取 cut 端单点]

    D --> H[encode_lines]
    E --> H
    F --> H
    G --> H

    H --> I[去掉 bos 和 eos]
    I --> J[末尾追加 <state>]
    J --> K[state_token_ids]
```

当前你常见的 `state` 配置默认更接近：

- `state_update_mode = patch_scan`
- `state_prefix_mode = cut_traces`

而更贴论文的方向是：

- `state_prefix_mode = cut_points`

## 10. 训练时单个样本最终长什么样

```mermaid
flowchart LR
    A[image] --> Z[模型输入]
    B[prompt_text] --> Z
    C[state_token_ids] --> Z
    D[map_token_ids] --> Y[训练目标]
```

其中：

- `image` 是 resize 后的卫星图
- `prompt_text` 是文字模板
- `state_token_ids` 是 previous map prefix
- `map_token_ids` 是当前 patch GT 的目标 token 序列

## 11. 推理时和训练时最关键的差别

```mermaid
flowchart TD
    A[训练时] --> A1[state 来自 GT 累积出来的 global_lines]
    B[推理时] --> B1[state 来自模型前面 patch 的预测结果累积]

    A1 --> C[几何链相同]
    B1 --> C
    C --> D[但推理误差会沿扫描顺序传播]
```

这也是当前现象里一个关键点：

- 训练 token-level 指标不算太差
- 但 state-scan official metrics 很弱

因为一旦进入推理闭环，错误会在 `patch_scan -> global merge -> next patch state` 中持续传播。

## 12. 一句话总结

可以把当前实现理解成：

1. 先把当前 patch GT 变成有 `category / line_type / start-end-cut / points` 的 token 序列。
2. 再把历史 patch GT 通过几何投影构成当前 patch 的 previous state。
3. 训练时让模型根据 `卫星图 + previous state + 文本 prompt` 生成当前 patch 的 map token。
4. 推理时把模型预测重新并回全局图，继续供后续 patch 使用。

相关文档：

- [current_model_flowchart_20260308.md](/mnt/data/project/jn/UniMapGen/docs/current_model_flowchart_20260308.md)
- [paper_aligned_model_flowchart_20260308.md](/mnt/data/project/jn/UniMapGen/docs/paper_aligned_model_flowchart_20260308.md)
- [agent_handoff_update_20260308.md](/mnt/data/project/jn/UniMapGen/docs/agent_handoff_update_20260308.md)
