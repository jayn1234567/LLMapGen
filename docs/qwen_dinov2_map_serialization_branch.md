# Qwen + DINOv2 Map Serialization Branch

## 1. 当前范围

这一分支对齐你现在要求的论文主干：

1. 冻结 `DINOv2 ViT-L/14` 提取卫星图 token
2. 线性投影到 `Qwen2.5-1.5B` hidden size
3. 文本 prompt 做 Qwen tokenization
4. `satellite prefix embeddings + prompt embeddings + map target tokens` 一起送入 Qwen
5. Qwen 自回归生成离散 map token
6. 用 detokenizer 把离散 token 还原成 polyline map
7. 用序列化后的卫星图真值监督 Qwen 输出段

当前仍未纳入：

- 对齐后的正式卫星数据
- 指标优化

当前已经纳入的 State Update 范围：

- 训练时：按 patch 扫描顺序累积 GT 全局状态，再为当前 patch 抽取局部邻接 `state` prefix，属于 teacher forcing
- 推理时：新增 scan 版本脚本，优先使用“历史全局线 -> 投影到当前 patch -> state prefix -> 当前 patch 预测 -> 回投全局 -> merge”
- 状态前缀支持 `all / cut_only / cut_points / cut_traces`，当前默认使用 `cut_traces`
- 状态 prompt 显式说明 `<s_cut>/<e_cut>` 的边界连接语义
- 推理阶段加入基于序列语法的 constrained decoding
- 跨 patch 几何状态增加“densify + overlap margin + endpoint primitive”投影补偿
- 全局 merge 增加基于 cut 端点邻近的线段拼接
- 当前 patch 的 state 只从真正相邻的局部历史中提取，而不是直接使用全部全局历史
- endpoint primitive 已升级为 short trace primitive，保留局部方向信息

## 2. 关键实现

- Dataset:
  - `/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_dataset.py`
- Hybrid tokenizer:
  - `/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_tokenizer.py`
- Model:
  - `/mnt/data/project/jn/UniMapGen/unimapgen/models/qwen_map_generator.py`
- Pipeline helpers:
  - `/mnt/data/project/jn/UniMapGen/unimapgen/qwen_map_pipeline.py`
- Train / Eval / Predict:
  - `/mnt/data/project/jn/UniMapGen/unimapgen/train_qwen_map.py`
  - `/mnt/data/project/jn/UniMapGen/unimapgen/eval_qwen_map.py`
  - `/mnt/data/project/jn/UniMapGen/unimapgen/predict_qwen_map.py`
- State-scan predict:
  - `/mnt/data/project/jn/UniMapGen/unimapgen/predict_qwen_state_scan.py`
- State helpers:
  - `/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_dataset.py`

## 3. 设计要点

### 3.1 词表处理

旧实现的问题是把 Qwen 词表直接 resize 成 map 自定义词表大小，这会破坏预训练语言模型的基础词表语义。

现在的实现改成：

1. 保留完整 Qwen 原始词表
2. 把 map serialization token 追加为新增 token
3. prompt 走 Qwen 正常文本编码
4. map GT 走新增 map token id

这样更接近论文里“视觉 prefix + 文本指令 + 大模型输出离散地图 token”的形式。

### 3.2 损失定义

训练时送入：

1. DINOv2 产生的卫星 prefix embeddings
2. prompt token embeddings
3. map target token embeddings

labels 只在 map token 段有效，prefix 和 prompt 段都置 `-100`，因此损失只监督地图序列输出。

### 3.3 推理约束

推理时只允许 Qwen 在 map token 集合里选下一个 token，避免退化成普通自然语言输出。

### 3.4 State Update

当前按论文主旨复现成两部分：

1. `GT previous state -> current patch` 的训练路径
2. `predicted previous state -> current patch` 的扫描推理路径

训练阶段：

- 数据集按 `splits_meta.json` 中的 patch 顺序组织样本
- 当前 patch 的 previous state 不再直接等同于“上一张 patch 的全量线”
- 而是按扫描顺序维护 GT 全局状态，并从中抽取与当前 patch 真正相邻的局部历史
- previous state 再序列化，并保留 `cut_traces` 或其它前缀模式
- 最后追加 `<state>` token，作为 Qwen 当前 patch 生成的条件
- 由于当前 small-set 没有真实 patch 网格坐标，这里的 `patch_scan` 先用 `splits_meta.json` 顺序近似代替

推理阶段：

- `predict_qwen_state_scan.py` 不再使用 GT previous state
- 如果提供 `patch_geometry_json`，会把历史预测线回投到全局坐标，再投影到当前 patch 形成 state prefix
- 当前 patch 预测完成后，再回投全局并做去重 merge
- 如果没有几何元数据，则退回“上一 patch 预测 -> 当前 patch state prefix”的顺序版本
- 几何投影时会对全局线做 densify，并使用 overlap margin 放宽 patch 重叠判定
- 即使整条线没有稳定重叠，也会把邻近边界的 cut endpoint 投影成 short trace primitive，作为 state prefix
- merge 时会优先尝试用端点距离做跨 patch 线段连接，再做签名去重
- 对同一原始卫星图的 patch，优先用 `crop_region` 而不是 `gps_center` 做投影
- endpoint primitive 的邻近判定采用扩展窗口，允许轻度裁剪误差

这和论文里的“利用相邻 patch 的 cut points 迭代更新当前 map state”的思路是一致的。

### 3.5 Stable Decoding

当前解码不再只是“从全部 map token 中选一个”。

加入了两层稳定化约束：

1. `repetition_penalty`
2. 基于 serialization 语法的 token 级约束

语法约束要求输出遵循：

`<bos> -> <line> -> <cat_*> -> <s_*> -> <e_*> -> <pts> -> <x_*> <y_*> ... -> <eol> / <eos>`

同时要求：

- 每条 polyline 至少输出 2 个点后才能 `eol`
- 超过 `max_lines` 后只能输出 `eos`

这样能显著减少“很快退出”或“生成结构无效 token 序列”的情况。

## 4. Smoke 配置

- config:
  - `/mnt/data/project/jn/UniMapGen/configs/qwen_dinov2_map_serialization_smoke.yaml`
- state config:
  - `/mnt/data/project/jn/UniMapGen/configs/qwen_dinov2_map_serialization_state_smoke.yaml`
- script:
  - `/mnt/data/project/jn/UniMapGen/scripts/run_qwen_dinov2_map_serialization_smoke.sh`
  - `/mnt/data/project/jn/UniMapGen/scripts/run_qwen_dinov2_map_serialization_state_smoke.sh`

当前 smoke 配置为了兼容本机 CPU 环境，做了保守裁剪：

- `image_size=224`
- `coord_num_bins=224`
- `max_seq_len=192`
- train/val 只取极少样本

这不是最终论文设置，只用于验证这条主线代码能跑通。

## 5. 后续路线

拿到对齐后的卫星图后，建议按这个顺序继续：

1. 先把 `opensatmap_root/ann_json/split_dir` 切到对齐后的正式数据
2. 再把 `image_size / coord_num_bins / max_seq_len / max_lines / max_points_per_line` 调回论文目标设置
3. 把当前 small-set 的顺序扫描改成真实 patch 坐标驱动的 `left-to-right, top-to-bottom` 扫描
4. 再补更严格的跨 patch 几何裁剪与 merge
5. 最后再做解码策略和指标优化
