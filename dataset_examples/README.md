# 数据集样例说明

这份目录用于说明最小 RC LLM 分支里几类核心数据集到底长什么样。

注意两点：

1. 真实训练时使用的是 `jsonl` 文件，每一行都是一个样本。
2. 这里为了便于阅读，示例统一写成 `json`，其中同时展示：
   - 一条 `dataset_jsonl` 的样本行
   - 一条 `dataset_meta_jsonl` 的样本行

## 文件列表

### 1. `rc_centerline_base_example.json`

共享底座格式示例。  
很多派生数据集最终都会回到这套 train root / meta root 结构上取字段。

重点字段：

- `id`：样本唯一标识
- `images[0]`：主图相对路径
- `messages`：原始多轮数据格式
- `target_lines`：中心线真值
- `structure_json`：可见道路结构线
- `seg_binary / seg_structure_multiclass`：Stage A 用到的掩码路径

### 2. `rc_structure_seg_example.json`

Stage A 结构分割训练直接读取的最小字段集合。

重点字段：

- `dataset_jsonl_example.images[0]`
- `dataset_meta_jsonl_example.seg_binary`
- `dataset_meta_jsonl_example.seg_structure_multiclass`

### 3. `rc_semantic_align_example.json`

Stage 1 粗对齐数据示例。

重点字段：

- `semantic_text`：图文对齐用的自然语言文本
- `semantic_scene_label`
- `semantic_visible_sides`
- `semantic_group_key / semantic_group_id`

如果这些字段缺失，代码也能从 `structure_json` 和 `target_lines` 回推一部分信息，但正式数据建议尽量提前写好。

### 4. `rc_caption_short_example.json`

Stage 2 细对齐数据示例。

重点字段：

- `caption_short`
- `caption_label`
- `caption_schema_version=scene_grid_states_v1`
- `seg_structure_multiclass`

Stage 2 当前固定输出：

- `Scene=<scene_label>`
- `GridStates=[state_1,...,state_64]`

### 5. `rc_centerline_json_sft_example.json`

Stage 3 中心线 JSON SFT 的数据示例。

重点字段：

- assistant 里直接放最终 JSON
- meta 里保留 `target_lines`，方便推理脚本做 GT 对比

### 6. `auxiliary_structure_lines_example.json`

可见道路结构线示例，通常对应 `structure_json`。

### 7. `auxiliary_centerline_lines_example.json`

中心线真值示例，通常对应 `centerline_json`。

## 怎么看这些示例

建议按下面顺序看：

1. 先看 `rc_centerline_base_example.json`
2. 再看 `rc_structure_seg_example.json`
3. 再看 `rc_semantic_align_example.json`
4. 再看 `rc_caption_short_example.json`
5. 最后看 `rc_centerline_json_sft_example.json`

这样会更容易看出：

- 同一张 RC 图是怎么被 Stage A / Stage 1 / Stage 2 / Stage 3 逐步消费的
- 哪些字段是共享的
- 哪些字段是某个阶段单独新增的
