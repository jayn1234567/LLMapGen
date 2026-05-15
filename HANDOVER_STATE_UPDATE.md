# State Update 复现交接文档

## 1. 当前目标

当前工作目标不是完整逐项复现论文全部内容，而是先在现有项目框架上复现论文的核心流程：

- 继续使用当前模型主干：
  - `DINOv3` 作为视觉编码器
  - `Qwen3-VL` 抽取出的 LLM 部分作为语言模型
  - `2-layer MLP projector` 作为对齐层
- 第一阶段仅做 `BEV-only`
- 继续使用 `Douglas` 采样，不改成论文的等距采样
- 输出目标从单纯 `centerline` 扩展为：
  - `centerline`
  - `intersection`
- 引入 `cut|inside` 端点语义
- 在推理阶段加入 `state update` 工作流

## 2. 当前共识

### 2.1 模型坐标系

- 模型训练和推理都只输出 **patch 局部坐标**
- 模型不预测全局坐标
- 全局拼接只在外部推理脚本中处理

### 2.2 state update 的位置

`state update` 不放进模型主干，不改 `forward`

它应该作为 **推理时的外部编排脚本** 存在，职责是：

1. 按 patch 顺序遍历整图
2. 从前面 patch 结果中抽取 `incoming traces`
3. 把这些 traces 注入当前 patch 的 user prompt
4. 调用当前模型做单 patch 推理
5. 合并当前 patch 结果，供后续 patch 使用

### 2.3 训练策略

训练不是递归式 state update 训练，而是：

- 仍然做单 patch supervised fine-tuning
- prompt 中可以带 `incoming traces`
- 输出中带 `centerline / intersection / cut|inside`

checkpoint 选择策略分三类入口：

- 普通训练脚本默认不做 eval，也不维护 best-loss 目录。
- `train_full_dinov3_qwen3vl-8b_deepstack_train_best_npu.sh` 用训练过程中的 `loss` 维护 `output/best/`，默认从 `BEST_TRAIN_LOSS_START_STEP=3000` 之后开始比较。
- `train_full_dinov3_qwen3vl-8b_deepstack_eval_best_npu.sh` 使用单独验证集，每 `EVAL_STEPS` 做一次 eval，并按最低 `eval_loss` 维护 `output/eval_best/`。

### 2.4 patch 顺序规则

推理阶段固定使用 row-major 顺序：

```text
(0,0) -> (0,1) -> (0,2) -> ...
(1,0) -> (1,1) -> (1,2) -> ...
```

也就是：

- 从上到下逐行处理
- 每一行从左到右处理

当前 patch `(row, col)` 只允许使用已经处理过的邻居：

- 左邻：`(row, col - 1)`
- 上邻：`(row - 1, col)`

训练时样本可以 shuffle，不要求相邻 patch 连续进入 batch。训练约束只在数据生成阶段体现：每条样本的 `incoming_traces` 必须按同样的左邻/上邻规则从 GT 邻接 patch 构造。

推荐分两阶段：

- **Phase A**
  - 用新 schema 训练
  - `Incoming traces JSON: []`
  - 目标是先学会新格式和 `cut|inside`

- **Phase B**
  - 使用 GT 邻接 patch 构造的 `incoming traces`
  - 每条 trace 从相邻的一条 `centerline` 上取共享边界附近的有序点：优先 3 个点，不满足 3 个则保留 2 个点；如果只剩 1 个边界锚点也保留，避免把“有 cut 延续但方向不足”误编码成“没有 incoming”
  - 只有相邻线端点的 `start_type/end_type` 是 `cut` 时才生成 trace；原始线自然端点即使落在 patch 边界也保持 `inside`，不能作为 continuity hint
  - 用点序列方向替代论文里的显式 direction
  - 目标是让模型学会利用 continuity hints

## 3. 当前 schema 设计

### 3.1 centerline

`centerline` 是开放折线，需要端点类型：

```json
{
  "category": "centerline",
  "start_type": "cut",
  "end_type": "inside",
  "points": [[113,0],[101,21],[54,112]]
}
```

语义：

- `cut`: 端点被 patch 边界裁切，后续可能需要和邻接 patch 连起来
- `inside`: 端点在 patch 内自然结束

### 3.2 intersection

`intersection` 是闭合线，不带 `start_type/end_type`，但需要带 `is_cut`

```json
{
  "category": "intersection",
  "is_cut": true,
  "points": [[92,92],[164,92],[164,164],[92,164],[92,92]]
}
```

语义：

- 用闭合 polyline 圈出路口区域
- `is_cut=true` 表示该路口 polygon 被当前 patch 边界截断，推理时可向右/下相邻 patch 传递边界提示
- `is_cut=false` 表示该路口完整落在当前 patch 内

### 3.3 顶层输出

当前建议统一输出为：

```json
{
  "lines": [
    {
      "category": "centerline",
      "start_type": "cut",
      "end_type": "cut",
      "points": [[0,126],[92,126]]
    },
    {
      "category": "intersection",
      "is_cut": true,
      "points": [[92,92],[164,92],[164,164],[92,164],[92,92]]
    }
  ]
}
```

## 4. 已完成工作

### 4.1 新主文档

已新增：

- [REPRODUCTION_PLAN.md](/media/q/data2/jjh/project/unimapgen_mllm/REPRODUCTION_PLAN.md)

用途：

- 统一记录当前复现边界
- 明确训练 schema
- 明确 state update 的设计位置
- 明确哪些文件该改、哪些先别动

### 4.2 README 入口整理

已更新：

- [README.md](/media/q/data2/jjh/project/unimapgen_mllm/README.md)

作用：

- 作为当前文档入口页

### 4.3 数据转换脚本

已新增：

- [scripts/data/build_sft_dataset.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/data/build_sft_dataset.py)

这个脚本是当前第一版数据处理入口。

支持两种模式：

#### 模式 1：legacy-centerline

用途：

- 把当前旧版 `centerline` 样本转换成新 schema
- 自动填 `Incoming traces JSON: []`
- 自动根据边界位置推断 `cut|inside`

示例：

```bash
python scripts/data/build_sft_dataset.py \
    legacy-centerline \
    --input data/train.jsonl \
    --output data/train_state_phase_a.jsonl
```

#### 模式 2：state-update-meta

用途：

- 把已经带 `incoming_traces`、`target_lines`、patch 元数据的样本转成训练用 SFT 数据

示例：

```bash
python scripts/data/build_sft_dataset.py \
    state-update-meta \
    --input meta_train.jsonl \
    --output train_state_phase_b.jsonl
```

转换后的每条样本会写入以下规则元数据：

- `scan_order`: `row_major_top_to_bottom_left_to_right`
- `available_neighbors`: `["left", "top"]`
- `train_shuffle_allowed`: `true`
- `trace_source_train`: `gt_left_top_neighbors`
- `trace_source_infer`: `predicted_left_top_neighbors`

## 5. 已验证结果

### 5.1 legacy 路径验证通过

输入：

- [data/1.jsonl](/media/q/data2/jjh/project/unimapgen_mllm/data/1.jsonl)

结果：

- 成功转换 4 条样本
- 输出已变成 `{"lines": ...}` 新 schema
- 对边界点自动补出了 `cut|inside`

### 5.2 state-update 路径验证通过

输入：

- [configs/数据样本.json](/media/q/data2/jjh/project/unimapgen_mllm/configs/数据样本.json)

结果：

- 成功转换 8 条样本
- prompt 中保留了 `incoming_traces`
- 输出中保留了 `target_lines`
- patch 元数据被写入 `meta`

## 6. 本轮新增完成工作

### 6.1 prompt 模板

已在 [llava/conversation.py](/media/q/data2/jjh/project/unimapgen_mllm/llava/conversation.py) 新增：

- `conv_qwen_2_state_update_centerline`
- `conv_qwen_3_state_update_centerline`

### 6.2 推理 parser

已扩展 [scripts/infer_centerline_checkpoint.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/infer_centerline_checkpoint.py)：

- 新增 `parse_map_json`
- 兼容旧格式 `[{points, category: CenterLine}]`
- 兼容新格式 `{"lines": [...]}`
- 支持 `centerline`
- 支持 `intersection`

### 6.3 state update 推理脚本

已新增：

- [scripts/infer_centerline_state_update.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/infer_centerline_state_update.py)

当前第一版能力：

- 按 `(row, col)` 排序
- 只从左邻和上邻的 **已预测结果** 抽取 centerline traces
- 可通过 `--include-intersections` 从左邻和上邻的 **已预测 intersection** 抽取路口边界提示
- 将 traces / intersections 注入当前 user prompt
- 调用现有模型推理单 patch
- 支持 `--dry-run-prompts` 不加载模型验证流程
- 输出每个 patch 的局部结果和合并后的全局坐标结果

### 6.4 可视化脚本

已新增：

- [scripts/visualize_state_update_global.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/visualize_state_update_global.py)

作用：

- 读取 `scripts/infer_centerline_state_update.py` 输出的 summary json
- 直接绘制 `merged_global.lines`
- 支持绘制 patch 网格
- `centerline` 用红色，`intersection` 用蓝色

已扩展 [scripts/visualize_centerline.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/visualize_centerline.py)：

- 支持旧 list 格式
- 支持新 `{"lines": [...]}` 格式
- 支持绘制 `intersection`

## 7. 当前项目代码改动清单

### 数据处理

- [scripts/data/build_sft_dataset.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/data/build_sft_dataset.py)

作用：

- `legacy-centerline`: 旧中心线数据转新 schema，Phase A 使用
- `state-update-meta`: 带 `incoming_traces/target_lines` 的元数据转 SFT 样本，Phase B 使用
- 输出样本中记录 row-major 和 left/top 邻接规则

### prompt 模板

- [llava/conversation.py](/media/q/data2/jjh/project/unimapgen_mllm/llava/conversation.py)

新增：

- `conv_qwen_2_state_update_centerline`
- `conv_qwen_3_state_update_centerline`

### 推理 parser

- [scripts/infer_centerline_checkpoint.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/infer_centerline_checkpoint.py)

新增：

- `parse_map_json`

作用：

- 兼容旧 `CenterLine` list
- 兼容新 `{"lines": [...]}`
- 支持 `centerline` 和 `intersection`
- 保留 `intersection.is_cut`；旧输出没有 `is_cut` 时默认按 `false` 处理

### 评估指标

- [scripts/centerline_eval_metrics.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/centerline_eval_metrics.py)

作用：

- 将 `centerline` 的 GT / prediction 解析为折线
- 按 `meter_per_pixel` 转成米制坐标
- 使用 `shapely.geometry.LineString(...).buffer(buffer_size)` 计算两条线的 buffer IoU
- 使用 Hungarian 匹配，阈值默认 `0.33`
- 输出实例级和长度级 `precision / recall / f1`

依赖：

- `shapely`
- `scipy`

示例：

```bash
python scripts/centerline_eval_metrics.py \
  --summary-json outputs/summary.json \
  --meter-per-pixel 1 \
  --buffer-size 1 \
  --match-threshold 0.33
```

### state update 推理

- [scripts/infer_centerline_state_update.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/infer_centerline_state_update.py)

作用：

- 按 row-major 顺序处理 patch
- 从左邻/上邻预测结果抽取 traces，不使用 GT
- `--include-intersections` 时，从左邻/上邻预测出的 `is_cut=true` 路口抽取 1 到 3 个边界点
- centerline incoming trace 保留 1 到 3 个点；1 个点表示只有边界锚点，不能丢成空提示
- 把 traces 和 intersection hints 注入当前 user prompt
- 输出局部预测和合并后的全局坐标结果
- `--eval-centerline` 时在 summary 中写入 shapely buffer-IoU + Hungarian 的中心线评估指标

示例：

```bash
python scripts/infer_centerline_state_update.py \
  --checkpoint-dir outputs/my_checkpoint \
  --patch-json data/my_patch_dataset/test.jsonl \
  --image-folder data/my_patch_dataset \
  --output-json outputs/state_update_summary.json \
  --output-dir outputs/state_update_patches \
  --include-intersections \
  --eval-centerline
```

### 可视化

- [scripts/visualize_state_update_global.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/visualize_state_update_global.py)

作用：

- 读取 state-update summary
- 将所有 patch-local 预测合并后的 `merged_global.lines` 画成整图 PNG

示例：

```bash
python scripts/visualize_state_update_global.py \
  --summary-json outputs/state_update_summary.json \
  --output outputs/state_update_global.png \
  --draw-grid
```

- [scripts/visualize_centerline.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/visualize_centerline.py)

作用：

- 读取新旧 schema
- 绘制 `centerline`
- 绘制 `intersection`

## 8. 当前验证状态

2026-05-14 已在本地新副本 `/media/q/data2/jjh/project/unimapgen_mllm` 完成一次最小 debug 验证。

数据验证：

- 由 1 张 4096 图生成 256 个 `256x256` patch
- patch 顺序为 `tile_id -> row -> col`
- `cut/inside` 不再按“落在 patch 边界就 cut”判断，而是按 clipping 来源判断
- incoming trace 点数为 1 到 3，1 个点表示只有边界锚点、方向信息不足但仍存在邻接延续

两卡 ZeRO-3 debug 训练：

- GPU: `CUDA_VISIBLE_DEVICES=1,2`
- 模型: Qwen3-VL-2B + DINOv3-B
- DeepStack: `[3, 6, 9, 11]`
- 数据: `data/av2_patch_256_fullimage_cutflag_test_v2/sft.jsonl`
- 配置: `scripts/deepspeed_zero3.json`
- 训练: `max_steps=1`, `train_sample_limit=2`, LoRA
- 输出: `/tmp/unimapgen_zero3_debug_train`
- 结果: 训练完成，生成 `adapter_model.safetensors`, `non_lora_trainables.bin`, `qwen_multimodal_checkpoint.json`

两卡真实推理 debug：

- checkpoint: `/tmp/unimapgen_zero3_debug_train`
- 数据: 同一张 4096 图裁剪出的前 2 个 patch
- 输出: `/tmp/unimapgen_zero3_debug_infer_256`
- 结果: `summary_rank0.json` 和 `summary_rank1.json` 均生成，每个 rank 1 条样本，`parse_ok=True`

注意：

- 这是最小 smoke，不代表模型质量；只验证 ZeRO-3 训练、checkpoint 保存、LoRA 推理、DeepStack 加载和新 JSON schema 解析链路能跑通。
- 训练和推理都使用本地已有权重路径，没有复制权重到新项目。

## 9. 当前未做的工作

以下工作仍未完成：

- 用云端完整数据集生成 Phase A / Phase B 训练数据
- 启动新 schema 的正式训练
- 评估 `intersection` 标注的学习效果
- 用真实正式训练 checkpoint 跑 `scripts/infer_centerline_state_update.py`

## 10. 明确不该先动的地方

第一阶段不建议动这些文件：

- [llava/model/llava_arch.py](/media/q/data2/jjh/project/unimapgen_mllm/llava/model/llava_arch.py)
- [llava/model/language_model/llava_qwen3.py](/media/q/data2/jjh/project/unimapgen_mllm/llava/model/language_model/llava_qwen3.py)
- [llava/model/multimodal_encoder/dinov3_encoder.py](/media/q/data2/jjh/project/unimapgen_mllm/llava/model/multimodal_encoder/dinov3_encoder.py)
- DeepStack 主结构
- projector 主结构

原因：

- 当前缺口主要在 **任务流程和数据接口**
- 不是 backbone 表达能力

## 11. 下一步建议执行顺序

推荐按下面顺序继续：

1. 用云端完整数据集生成 Phase A / Phase B 训练数据
2. 用 Phase A 数据先训练，确认新 schema 输出稳定
3. 再用带 traces 的 Phase B 数据训练
4. 用真实 checkpoint 跑 `scripts/infer_centerline_state_update.py`
5. 用 `scripts/visualize_state_update_global.py` 拼接并查看整图 PNG
6. 专门评估 `intersection` 闭合线质量

## 12. 当前重要结论

当前已经明确的设计结论如下：

- 模型只预测 patch 局部坐标
- `state update` 是推理时外部脚本，不进模型主干
- 训练仍然是单 patch SFT
- 训练样本最好逐步加入 `incoming traces`
- `intersection` 是闭合线，不带 `start_type/end_type`
- `centerline` 才带 `cut|inside`
- 第一版只看 `left` 和 `top` 邻接 traces
- 推理时 `left/top` 输入来自前面 patch 的预测值，不来自 GT
- 推理顺序固定为从上到下、每行从左到右
- 训练样本可以 shuffle

## 13. 接手时优先看的文件

建议按这个顺序阅读：

1. [REPRODUCTION_PLAN.md](/media/q/data2/jjh/project/unimapgen_mllm/REPRODUCTION_PLAN.md)
2. [scripts/data/build_sft_dataset.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/data/build_sft_dataset.py)
3. [configs/数据样本.json](/media/q/data2/jjh/project/unimapgen_mllm/configs/数据样本.json)
4. [llava/conversation.py](/media/q/data2/jjh/project/unimapgen_mllm/llava/conversation.py)
5. [scripts/infer_centerline_checkpoint.py](/media/q/data2/jjh/project/unimapgen_mllm/scripts/infer_centerline_checkpoint.py)
