# Stage A / Stage B 微调数据集字段说明

本文档说明 `unimapgen-v1.0` 中两阶段微调所使用的数据集字段，包括：

- Stage A patch-only 数据集
- Stage B state-mixture 数据集
- LLaMAFactory 注册文件 `configs/*/dataset_info.json`
- 数据导出脚本在 `outputs/<dataset_root>/` 下生成的主文件和辅助文件

## 1. 数据目录结构

### Stage A

Stage A 导出后，典型目录如下：

```text
outputs/paper16_patch_only_100img_system/
├── train.jsonl
├── val.jsonl
├── meta_train.jsonl
├── meta_val.jsonl
├── dataset_info.json
└── images/
    ├── train/<family_id>/p00.png ...
    └── val/<family_id>/p00.png ...
```

### Stage B

Stage B 导出后，典型目录如下：

```text
outputs/paper16_sft_100img_system_paper_serialized_neighborfix_mixture/
├── train.jsonl
├── val.jsonl
├── meta_train.jsonl
├── meta_val.jsonl
├── dataset_info.json
└── images/
    ├── train/<family_id>/p00.png ...
    └── val/<family_id>/p00.png ...
```

## 2. 训练主文件：`train.jsonl` / `val.jsonl`

这两个文件是真正喂给 LLaMAFactory 的主训练样本。每一行是一个 JSON 对象。

### 通用顶层字段

| 字段 | 类型 | Stage A | Stage B | 含义 |
|---|---|---:|---:|---|
| `id` | `string` | yes | yes | 样本唯一 ID，一般是 `<family_id>_p<patch_id>` |
| `messages` | `list[object]` | yes | yes | ShareGPT 格式多轮消息，至少包含 `user` 和 `assistant`，通常还包含 `system` |
| `images` | `list[string]` | yes | yes | 与当前样本绑定的图像相对路径，当前实现固定为单张 patch 图像 |

### `id`

格式示例：

```text
ATX_-1_0_sat__paper16_r0_c0_p00
```

含义：

- `ATX_-1_0_sat`：原始 4096 图像 ID
- `paper16_r0_c0`：该 family 在 paper16 网格中的起始窗口
- `p00`：当前 family 内的 patch 编号，范围通常是 `p00` 到 `p15`

### `images`

示例：

```json
"images": ["images/train/ATX_-1_0_sat__paper16_r0_c0/p00.png"]
```

说明：

- 路径是相对于数据集根目录的相对路径
- 当前实现中只放一张图，因此 `images` 列表长度为 1
- LLaMAFactory 会根据这个路径去加载 patch 图像

## 3. `messages` 字段说明

`messages` 是 ShareGPT 风格消息列表。每个元素至少有：

| 字段 | 类型 | 含义 |
|---|---|---|
| `role` | `string` | 角色，常见值：`system` / `user` / `assistant` |
| `content` | `string` | 该轮文本内容 |

### 3.1 Stage A 的 `messages`

Stage A 的任务是：

- 只看当前 patch 图像
- 直接预测当前 patch 的完整路网
- 不提供 previous state

典型结构：

```json
{
  "id": "...",
  "messages": [
    {"role": "system", "content": "...可选系统提示词..."},
    {"role": "user", "content": "<image>
Please construct the complete road map in the current satellite patch."},
    {"role": "assistant", "content": "{"lines":[...]}"}
  ],
  "images": ["images/train/.../p00.png"]
}
```

各字段语义：

| 位置 | 字段 | 含义 |
|---|---|---|
| `messages[0]` | `system` | 可选系统提示词，要求模型只输出 JSON，不输出解释 |
| `messages[1]` | `user` | 固定 patch-only prompt，不含 previous state |
| `messages[2]` | `assistant` | 目标标注，JSON 字符串，内容为当前 patch 的完整路网 |

### 3.2 Stage B 的 `messages`

Stage B 的任务是：

- 看当前 patch 图像
- 同时读取 previous state（来自已处理邻居 patch 的 cut traces）
- 预测当前 patch 的 ownership region map，并学习跨 patch continuity

典型结构：

```json
{
  "id": "...",
  "messages": [
    {"role": "system", "content": "...可选系统提示词..."},
    {
      "role": "user",
      "content": "<image>
Please construct the road map in the current patch.
...
Previous state:
{"lines":[...]}"
    },
    {"role": "assistant", "content": "{"lines":[...]}"}
  ],
  "images": ["images/train/.../p00.png"]
}
```

各字段语义：

| 位置 | 字段 | 含义 |
|---|---|---|
| `messages[0]` | `system` | 可选系统提示词，要求利用 previous state 保持跨 patch 连续性 |
| `messages[1]` | `user` | 含图像占位符 `<image>`，并把 `state_json` 文本插入 prompt |
| `messages[2]` | `assistant` | 目标标注，JSON 字符串，内容为当前 patch 目标路网 |

## 4. `assistant.content` 的 JSON 结构

无论 Stage A 还是 Stage B，`assistant.content` 都是一个 JSON 字符串，顶层结构统一为：

```json
{
  "lines": [
    {
      "category": "road",
      "start_type": "start",
      "end_type": "cut",
      "points": [[x1, y1], [x2, y2], ...]
    }
  ]
}
```

### `lines` 中每条 polyline 的字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `category` | `string` | 输出类别，当前导出脚本默认是 `road` |
| `start_type` | `string` | 起点类型，通常为 `start` 或 `cut` |
| `end_type` | `string` | 终点类型，通常为 `end` 或 `cut` |
| `points` | `list[list[int,int]]` | 折线顶点，patch-local 坐标系，单位是像素 |

### `start_type` / `end_type`

| 值 | 含义 |
|---|---|
| `start` | 该端点是自然起点，不是被 patch 边界截断 |
| `end` | 该端点是自然终点，不是被 patch 边界截断 |
| `cut` | 该端点位于 patch 边界附近，表示线段在这里被截断，需要跨 patch continuation |

### `points`

说明：

- 坐标系是 `patch_local_896`
- 左上角是 `(0, 0)`
- 右下角接近 `(895, 895)`
- 已经过裁剪、去重和整数化处理
- 至少包含 2 个点

## 5. Stage B 中 `user` prompt 里的 `state_json`

Stage B 的 previous state 是通过文本形式嵌入到 `user.content` 中的，结构如下：

```json
{
  "lines": [
    {
      "source_patch": 3,
      "category": "road",
      "start_type": "cut",
      "end_type": "cut",
      "points": [[x1, y1], [x2, y2], ...]
    }
  ]
}
```

### `state_json.lines[*]` 字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `source_patch` | `int` | 这条 state trace 来自哪个已处理邻居 patch |
| `category` | `string` | 类别，当前默认 `road` |
| `start_type` | `string` | 起点类型，通常是 `cut` |
| `end_type` | `string` | 终点类型，通常是 `cut` |
| `points` | `list[list[int,int]]` | 从邻居 patch handoff 过来的 trace 点，坐标已转换到当前 patch 的 local 坐标系 |

### `source_patch`

- 对当前 patch 来说，通常只会来自左邻或上邻
- 它是 family 内部 patch 的编号，不是全局图像编号
- 用于辅助分析 state trace 是从哪个邻居传入的

## 6. Stage A / Stage B 的目标差异

### Stage A 目标

Stage A 的 `assistant.content` 监督的是：

- 当前 patch 内完整可见路网
- 对应 exporter 元数据里 `target_mode = full_patch_map`

换句话说，Stage A 是纯 patch-level reconstruction。

### Stage B 目标

Stage B 的 `assistant.content` 监督的是：

- 当前 patch 的 ownership region map
- 同时结合 neighborfix / state mixture 学习 continuity
- 对应 exporter 的数据集信息里 `target_mode = ownership_region_map`

换句话说，Stage B 不是简单重复 Stage A，而是把任务切换为“带状态的局部 ownership 预测”。

## 7. 辅助元数据文件：`meta_train.jsonl` / `meta_val.jsonl`

这两个文件不直接参与训练，但非常适合调试、可视化、统计和排错。

### 7.1 Stage A `meta_*.jsonl`

每行是一个样本的辅助描述，常见字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | `string` | 样本 ID |
| `split` | `string` | `train` / `val` |
| `family_id` | `string` | family ID |
| `source_image` | `string` | 原始 4096 图像文件名 |
| `patch_id` | `int` | 当前 patch 编号 |
| `row` / `col` | `int` | 当前 patch 在 4x4 family 网格中的行列 |
| `scan_index` | `int` | 当前扫描顺序，当前实现等于 `patch_id` |
| `image` | `string` | 对应 patch 图像相对路径 |
| `crop_box` | `object` | patch 的全局裁剪框 |
| `num_target_lines` | `int` | 目标折线数量 |
| `target_mode` | `string` | 固定为 `full_patch_map` |
| `coord_system` | `string` | 固定为 `patch_local_896` |
| `serialization_mode` | `string` | 当前为 `paper_structured` |
| `line_direction_mode` | `string` | 当前为 `canonical_cut_then_origin` |
| `line_sort_mode` | `string` | 当前为 `first_point_distance_to_patch_origin` |
| `resample_mode` | `string` | 当前为 `equal_distance` |
| `resample_step_px` | `float` | 重采样步长 |
| `has_system_prompt` | `bool` | 是否写入了 system prompt |
| `target_lines` | `list` | 与 `assistant.content` 对应的解析后目标折线 |

### 7.2 Stage B `meta_*.jsonl`

Stage B 的 `meta` 字段更多，除了 Stage A 的公共信息，还增加了 state 相关字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `ownership_rect_global` | `list[float]` | 当前 patch ownership 区域在全局图上的矩形框 |
| `state_source_patch_ids` | `list[int]` | 实际写入 prompt 的 state lines 来自哪些邻居 patch |
| `raw_state_source_patch_ids` | `list[int]` | mixture 处理前的原始 state lines 来自哪些邻居 patch |
| `state_mode_applied` | `string` | 当前样本实际采用的 state 模式 |
| `num_raw_state_lines` | `int` | mixture 前原始 state line 数量 |
| `num_state_lines` | `int` | mixture 后真正写入 prompt 的 state line 数量 |
| `num_target_lines` | `int` | 当前样本目标折线数量 |
| `state_mixture_mode` | `string` | 当前导出策略，一般是 `mixed` |
| `state_fake_ratio` | `float` | fake_state 采样比例配置 |
| `state_lines` | `list` | 实际喂给模型的 previous state |
| `raw_state_lines` | `list` | 还未做 mixture 退化前的原始 state |
| `target_lines` | `list` | 当前样本监督目标 |

### `state_mode_applied` 常见值

| 值 | 含义 |
|---|---|
| `empty` | 当前 patch 没有可用 previous state |
| `no_state` | 有原始 state，但训练时故意不给模型 |
| `weak_state` | 给模型一个弱化版 state |
| `fake_state` | 给模型一个经过扰动/偏移的 state |
| `full_state` | 给模型完整 state |

说明：当前 `unimapgen-v1.0` 主线只保留标准两阶段路线，但 exporter 代码里仍保留了 `fake_state` 分支能力，因此元数据里仍可能出现该模式字段。

## 8. 导出目录下的 `dataset_info.json`

注意这里有两类同名文件。

### 8.1 `outputs/<dataset_root>/dataset_info.json`

这是导出脚本生成的数据集说明文件，用于记录导出参数和摘要，不是 LLaMAFactory 的注册文件。

#### Stage A 版本常见字段

| 字段 | 含义 |
|---|---|
| `dataset_name` | 导出数据集名称 |
| `source_ann_json` | 原始标注 JSON 路径 |
| `source_family_manifest` | family manifest 路径 |
| `target_mode` | `full_patch_map` |
| `coord_system` | `patch_local_896` |
| `serialization_mode` | `paper_structured` |
| `line_direction_mode` | `canonical_cut_then_origin` |
| `line_sort_mode` | `first_point_distance_to_patch_origin` |
| `resample_mode` | `equal_distance` |
| `resample_step_px` | 重采样步长 |
| `use_system_prompt` | 是否使用系统提示词 |
| `system_prompt` | 系统提示词文本 |
| `summary` | 每个 split 的 family / sample 数统计 |

#### Stage B 版本额外字段

在 Stage A 基础上增加：

| 字段 | 含义 |
|---|---|
| `state_mode` | 当前 state 组织方式 |
| `target_mode` | `ownership_region_map` |
| `state_mixture_mode` | `mixed` 或 `full` |
| `state_no_state_ratio` | no_state 采样比例 |
| `state_weak_ratio` | weak_state 采样比例 |
| `state_fake_ratio` | fake_state 采样比例 |
| `state_full_ratio` | full_state 采样比例 |
| `state_weak_trace_points` | weak_state 使用的 trace 点数 |
| `state_line_dropout` | weak_state 线段 dropout |
| `state_point_jitter_px` | weak_state 点抖动 |
| `state_truncate_prob` | weak_state 截断概率 |
| `state_fake_trace_points` | fake_state 使用的 trace 点数 |
| `state_fake_line_dropout` | fake_state 线段 dropout |
| `state_fake_point_jitter_px` | fake_state 点抖动 |
| `state_fake_truncate_prob` | fake_state 截断概率 |
| `state_fake_shift_min_px` / `state_fake_shift_max_px` | fake_state 偏移范围 |

### 8.2 `configs/*/dataset_info.json`

这是 LLaMAFactory 的数据集注册文件，用于告诉训练框架：

- 数据文件在哪
- 使用什么格式解析
- `messages` / `images` 字段怎么映射

典型结构：

```json
{
  "unimapgen_paper16_patch_only_100img_train": {
    "file_name": "../../outputs/paper16_patch_only_100img_system/train.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

字段说明：

| 字段 | 含义 |
|---|---|
| 顶层 key | 训练时使用的数据集名字，例如 `unimapgen_paper16_patch_only_100img_train` |
| `file_name` | 主训练文件路径，即 `train.jsonl` |
| `formatting` | 这里固定为 `sharegpt` |
| `columns.messages` | 主消息列名称 |
| `columns.images` | 图像列名称 |
| `tags.role_tag` | 消息里角色字段名 |
| `tags.content_tag` | 消息里内容字段名 |
| `tags.user_tag` | user 角色值 |
| `tags.assistant_tag` | assistant 角色值 |
| `tags.system_tag` | system 角色值 |

## 9. 最常见的使用误区

### 误区 1：把 `configs/*/dataset_info.json` 当作导出数据说明文件

不是。`configs/*/dataset_info.json` 只是 LLaMAFactory 注册表。

### 误区 2：以为 Stage B 的 `assistant` 目标等于完整 patch 地图

不完全是。Stage B 的 supervision 目标是 ownership region map，并依赖 previous state 学 continuity。

### 误区 3：把 `state_lines` 当作额外图像输入

不是。Stage B 当前实现里，`state_lines` 是 JSON 文本，直接被拼进 `user.content` 里。

### 误区 4：认为 `meta_*.jsonl` 是训练必需文件

不是。训练只依赖 `train.jsonl` / `val.jsonl` 和 `images/`。`meta_*.jsonl` 主要用于分析和调试。
