# UniMapGen 复现路线（代码审计后）

## 1. 当前代码已实现能力（已核对）

1. 序列化与生成闭环
- 已有 map 向量线的序列化/反序列化（`serialize_annotation` + `MapSequenceTokenizer`）。
- 已有 `train/eval/predict` 全流程，并支持自回归生成。

2. 状态更新机制（工程近似）
- 已有 `G_{n-1}` 前缀训练方式（`<state>` 分隔，loss 只算 `G_n`）。
- 已有 `patch_scan` 全局扫描推理（`infer_state_scan.py`）。
- 已有 `start/end/cut` 端点属性 token。

3. 多模态输入骨架
- 已支持 BEV + PV + 文本提示的前缀注入。
- 已提供 paper 架构入口（`model.arch: paper`），可切换到 DINOv2/Qwen（或 fallback）。

## 2. 与论文关键差距（按优先级）

1. P0 可复现性
- 环境侧缺失 Python/Git 命令，当前无法直接在本机会执行训练验证命令。

2. P1 机制对齐
- 论文的坐标/角度离散化为 10000 bins；原实现按 `image_size`(512) 和 `360` bins，离散精度不足。
- 论文包含三阶段训练 recipe（SFT 预训练 -> VLM 对齐 -> 地图增量训练），现实现以单阶段 smoke 为主。

3. P2 数据与规模
- 论文使用 OpenSatMap20 大规模数据与增强；当前主要在 nuScenes patch 上迭代。

## 3. 本轮已落地改造（开始复现）

1. 新增论文级量化能力（核心）
- `MapSequenceTokenizer` 新增可配置参数：
  - `coord_num_bins`
  - `angle_num_bins`
- 当配置为 10000 时，坐标/角度 token 空间可与论文量化策略对齐。

2. 全链路打通
- `DatasetConfig` 新增上述两个字段。
- `train.py / eval.py / predict.py / infer_state_scan.py` 已全部透传该配置，避免训练与推理词表不一致。
- `configs/unimapgen_paper_scaffold.yaml` 已默认启用：
  - `coord_num_bins: 10000`
  - `angle_num_bins: 10000`

## 4. 接下来的执行路线（可直接开工）

1. 阶段 A（本周）：论文量化 + 状态更新稳定性
- 目标：确认 10k 量化下训练、推理、状态扫描都能稳定运行。
- 产物：`paper_scaffold` 的 smoke 指标与可视化样例。

2. 阶段 B（下周）：训练配方对齐
- 引入与论文一致的多阶段训练配置文件（先小规模仿真版本）。
- 加入更明确的 teacher-forcing 比例、解码反塌缩策略、以及状态前缀采样策略。

3. 阶段 C：数据规模对齐
- 从 nuScenes patch 过渡到 OpenSatMap20 流程（或兼容数据管线）。
- 补全数据增强与采样策略并做消融。

## 5. 建议先跑的命令（论文骨架 smoke）

```bash
# 10k 量化，无状态更新
python -m unimapgen.train --config configs/unimapgen_paper_token10k_smoke.yaml
python -m unimapgen.eval --config configs/unimapgen_paper_token10k_smoke.yaml --checkpoint outputs/unimapgen_paper_token10k_smoke/latest.pt

# 10k 量化 + 状态更新
python -m unimapgen.train --config configs/unimapgen_paper_token10k_state_smoke.yaml
python -m unimapgen.predict --config configs/unimapgen_paper_token10k_state_smoke.yaml --checkpoint outputs/unimapgen_paper_token10k_state_smoke/latest.pt --split val --max_samples 8 --output outputs/paper_token10k_state_preds.json
```

> 注：如无可用 Python 解释器或离线模型权重，需先补齐运行环境再执行。

## 6. 三阶段复现命令（新增）

```bash
# Stage 1: map generation pretrain (SFT)
python -m unimapgen.train --config configs/unimapgen_paper_stage1_sft_smoke.yaml

# Stage 2: multi-modal alignment (loads Stage1 latest.pt)
python -m unimapgen.train --config configs/unimapgen_paper_stage2_align_smoke.yaml

# Stage 3: state update training (loads Stage2 latest.pt)
python -m unimapgen.train --config configs/unimapgen_paper_stage3_state_smoke.yaml

# Stage3 evaluation and global state scan
python -m unimapgen.eval --config configs/unimapgen_paper_stage3_state_smoke.yaml --checkpoint outputs/unimapgen_paper_stage3_state_smoke/latest.pt
python -m unimapgen.infer_state_scan --config configs/unimapgen_paper_stage3_state_smoke.yaml --checkpoint outputs/unimapgen_paper_stage3_state_smoke/latest.pt --split val --scene_limit 1 --max_patches_per_scene 8 --output outputs/paper_stage3_state_scan.json
```
