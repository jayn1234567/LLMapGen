# UniMapGen 阶段性复现总结（2026-03-06）

## 1. 当前已经完成

### 1.1 卫星图编码分支

- 已接入本地预训练 `DINOv2 ViT-L/14`
- 冻结 backbone，仅训练下游头或投影层
- 已验证 `satellite -> DINOv2 token` 的训练与推理链路可运行

对应实现：

- `/mnt/data/project/jn/UniMapGen/unimapgen/models/encoders/satellite_encoder.py`
- `/mnt/data/project/jn/UniMapGen/unimapgen/models/dino_lane_seg.py`

### 1.2 论文主干的 Qwen 序列化分支

- 已接入本地 `Qwen2.5-1.5B`
- 已实现 `DINOv2 token -> linear projection -> Qwen hidden`
- 已实现文本 prompt tokenization
- 已实现 `satellite prefix + prompt prefix + map target tokens` 联合输入 Qwen
- 已实现离散 map token 的生成、反序列化与训练监督

对应实现：

- `/mnt/data/project/jn/UniMapGen/unimapgen/models/qwen_map_generator.py`
- `/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_tokenizer.py`
- `/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_dataset.py`
- `/mnt/data/project/jn/UniMapGen/unimapgen/train_qwen_map.py`
- `/mnt/data/project/jn/UniMapGen/unimapgen/eval_qwen_map.py`
- `/mnt/data/project/jn/UniMapGen/unimapgen/predict_qwen_map.py`

### 1.3 State Update

- 已实现训练侧 state prefix
- 已实现推理侧 `state scan`
- 已实现按 patch 扫描顺序逐步构造全局状态
- 已实现从全局状态抽取当前 patch 的局部邻接 state
- 已实现 `cut` 属性建模
- 已实现从 cut endpoint 扩展得到 short trace primitive
- 已实现跨 patch 全局 merge 与端点连接

对应实现：

- `/mnt/data/project/jn/UniMapGen/unimapgen/state_geometry.py`
- `/mnt/data/project/jn/UniMapGen/unimapgen/predict_qwen_state_scan.py`

### 1.4 可运行性

- GPU 环境已可用
- 本地 DINOv2 / Qwen 权重可直接加载
- 训练、评估、推理、state-scan 推理均可运行
- 当前 smoke 数据上已经验证 state prefix 会进入后续 patch

## 2. 哪些和论文是一致的

### 2.1 总体范式一致

- 使用卫星图作为主输入
- 使用视觉特征作为大模型前缀条件
- 使用离散 vector token 进行地图序列生成
- 使用 detokenizer 将 token 还原为 polyline map
- 使用 state update 做逐 patch 迭代生成

### 2.2 State Update 机制一致

- 使用 `start/end/cut` 属性表示 patch 边界连接关系
- 推理时采用 patch 扫描更新
- 当前 patch 参考 previous state map 中与自己相邻的 cut 状态
- 目标是保持跨 patch 的连续性和一致性

### 2.3 训练目标一致

- prefix 段不计入 loss
- loss 只监督当前 patch 的 map token 段
- 训练侧使用 teacher forcing 的 previous state

## 3. 哪些是近似实现

### 3.1 大模型部分

- 当前使用的是 `Qwen2.5-1.5B` 文本 LLM
- 不是论文里完整的多模态 Qwen2-VL 视觉主干
- 视觉输入通过外部 DINOv2 编码后投影接入，而不是论文原生 VL 视觉塔

### 3.2 数据部分

- 当前使用的是小规模样例 patch
- 不是论文规模的 OpenSatMap20 正式训练集
- 当前目标是跑通 pipeline，不是追论文指标

### 3.3 patch 几何与扫描

- 当前 small-set 缺少完整正式 patch 网格元数据
- 已优先利用 `crop_region` 与 `gps_center` 做工程近似几何
- 扫描顺序和相邻关系在这个阶段是近似实现，不是最终正式版

### 3.4 state prefix 形式

- 论文强调的是邻接 cut points
- 当前实现将其增强为 `short trace primitive`
- 这是为了给模型提供更强的局部方向约束，属于工程增强，不是论文原文逐字照搬

## 4. 当前达到的阶段

如果按“代码复现”而不是“指标复现”来评估，当前已经完成到：

- 论文主链可运行
- 状态更新可运行
- 训练侧和推理侧都接入了局部邻接 state
- 状态已经可以从前一 patch 传递到后一 patch

如果按“论文完整复现”来评估，当前还没有完成：

- 正式对齐卫星数据训练
- 论文规模数据增强
- 论文最终指标
- 更完整的 patch 几何拓扑约束

## 5. 当前 smoke 结果应如何理解

- 当前样本上模型常常只生成很短的线段
- 因此跨 patch 的 `projected lines` 仍然偏少
- 但 `state_len`、`num_candidate_lines`、`num_endpoint_primitives` 已经证明状态更新链路本身是有效的

所以现在的问题主要是：

- 数据规模太小
- 模型还没有学到稳定、长程的几何生成

而不是：

- state update 代码没接通
- Qwen/DINOv2 分支跑不起来

## 6. 建议的下一阶段

1. 使用后续对齐后的正式卫星数据替换当前 small-set
2. 保留现有 `local adjacent state` 与 `short trace primitive` 机制
3. 放大训练规模与训练轮数
4. 再评估是否需要继续增强跨 patch 几何约束
