# UniMapGen 复现路线图（v1）

## 已完成（当前轮）

1. 论文要点落地到代码骨架
- 等距采样（默认 6m）
- 线重排（首点到原点距离）
- 序列化 token 生成与反序列化

2. 数据集接入
- 读取 `nuscenes_map_infos_temporal_{split}.pkl`
- 使用 `token_satellite.png` 匹配卫星图
- 兼容当前 `512x512` 卫星图

3. 训练闭环
- `unimapgen.train` 可训练
- `unimapgen.eval` 可评估
- `unimapgen.predict` 可自回归导出预测 json
4. 多模态第一步
- 已接入 `BEV + CAM_FRONT(PV)` 轻量版本（`configs/unimapgen_v1_pv_smoke.yaml`）
- 当前使用单帧 PV（L=1），用于快速验证多模态链路
 - 已支持训练阶段 BEV/PV 随机模态 mask（可配置 `train.modality_mask`）
5. 状态更新第一步（v2）
- 已实现 `G_{n-1}` 前缀条件输入（`<state>` 分隔）
- 训练时仅对 `G_n` token 计算 loss（前缀不计入）
- 可运行配置：`configs/unimapgen_v2_state_smoke.yaml`
- 状态来源升级：支持 `patch_scan` 模式（scene 内按全局坐标近似 left->right, top->bottom）
6. 线端点属性
- 已加入 line 属性 token：`start/end/cut`（编码为 `<s_*>` 与 `<e_*>`）
- 当前以端点是否接近 patch 边界作为 `cut` 判定
7. 指标与对比
- 已增加工程近似指标：`APC@2/4/8px`、`mean_chamfer_px`、`continuity_pred/gap`
- 自动输出对比表：`docs/v1_v2_comparison.md`

## 与论文的差异清单（v1）

1. 模型主干差异
- 论文：DINOv2-L + Qwen2-VL-ViT + Qwen2.5-1.5B
- 当前：轻量 CNN + Transformer Decoder（为了快速跑通）

2. 多模态差异
- 论文支持 BEV/PV/Text 任意组合，并做模态随机 mask
- 当前先做 BEV-only（后续补 PV/Text）

3. 状态更新差异
- 论文：基于上一状态地图 `G_{n-1}` 的全局迭代更新
- 当前：单 patch 训练，未启用跨 patch 状态更新训练

4. 数据规模与增强
- 论文：OpenSatMap20 4096 图 + 896 patch + 70万增强 patch
- 当前：nuScenes 对应卫星 patch 直接训练，增强策略未完整实现

## 下一阶段（v2）

1. 加入 PV 分支（先轻量实现）
- 扩展到多帧 PV（L>1）与时间采样
- 引入 PV 位置和朝向 token（point+angle）

2. 完整状态更新训练（下一步）
- 把上一状态 cut 点/线作为条件输入 token（替换当前“整图前缀”近似）
- 在推理阶段实现真实 patch 级增量更新与拼接（已落地 `unimapgen.infer_state_scan` 工程版）

8. 全局推理（工程版）
- 新增 scene 级扫描推理与全局坐标拼接输出：`outputs/state_scan_global_smoke.json`
- 当前策略：历史全局线投影到当前 patch 形成 `G_{n-1}` 条件前缀，再自回归生成当前 patch 线并回投全局
- 新增 `cut_only` 前缀修复：按“投影后几何边界”重判 cut，并在空前缀时启用小规模 fallback，避免后续 patch 始终无状态输入

3. 评估指标对齐
- 增加 Chamfer AP 与 mIoU 近似评估
- 增加线连续性与端点连接质量统计

## 增量进展（当前轮）

1. 多模态鲁棒性（训练侧）
- 新增训练阶段 BEV/PV 随机模态 mask（`train.modality_mask`）。
- 约束“至少保留一种视觉模态”，避免样本全空导致训练异常。
- 训练日志新增每轮模态屏蔽比例统计。

2. PV 时序输入升级
- 数据集已从单帧 PV 升级到多帧时序（按 `prev` 链回溯，支持 `pv_num_frames>1`）。
- 增加三帧 smoke 配置：`configs/unimapgen_v1_pv_multiframe_smoke.yaml`。
- 端到端训练已跑通（CPU smoke）。

3. 可复现闭环增强
- 训练输出新增：
  - `config_snapshot.yaml`
  - `run_meta.txt`
  - `metrics.jsonl`
- 支持后续迭代严格对比与回溯。

4. 本轮快速对比（4样本，quick）
- 单帧 vs 多帧 PV 对比文件：`docs/pv_single_vs_multiframe_quick.md`
- 结论：当前极小训练规模下两者仍处于欠拟合阶段，尚未观察到稳定提升；需扩展训练步数与样本规模后再评估。

5. 新增进展（继续迭代）
- 新增轻量数据增强（训练侧）：90/180/270 随机旋转 + 随机水平/垂直翻转，并同步变换向量坐标。
- 新增增强 smoke 配置：`configs/unimapgen_v1_aug_smoke.yaml`，训练与评估已跑通。
- 新增文本提示最小通道（prompt type embedding）并打通 `train/eval/predict` 链路，配置：`configs/unimapgen_v1_pv_text_smoke.yaml`。
- 新增 `BEV+PV` vs `BEV+PV+Text` quick 对比：`docs/pv_vs_pv_text_quick.md`。
- 回归检查：`unimapgen.infer_state_scan` 在最新代码下可正常运行，输出 `outputs/state_scan_global_quick_latest.json`。

6. 小规模阶段实验（stage1）
- 新增配置：`configs/unimapgen_v1_pv_text_stage1.yaml`（64 train / 32 val, 3 epochs）。
- 训练趋势（`outputs/unimapgen_v1_pv_text_stage1/metrics.jsonl`）：
  - `val_loss`: 6.0024 -> 5.3662 -> 5.3073
  - `val_tok_acc`: 0.1444 -> 0.1969 -> 0.1990
- 预测对比（`docs/pv_text_smoke_vs_stage1.md`，8样本 quick）：
  - `mean_chamfer_px`: 67.2773 -> 64.0000（略有改善）
  - 但仍存在“预测线数接近 0”的退化，说明当前生成仍偏塌缩，需要进一步优化解码与训练配方。

7. 坐标化文本提示 token 已落地
- 新增 tokenizer token：
  - `target_xy`: `<txt_xy> <x_*> <y_*> <x_*> <y_*> <txt_end>`
  - `trace_points`: `<txt_trace> (<x_*> <y_*> <a_*>)... <txt_end>`
- 新增角度 token 空间：`<a_0> ... <a_359>`（用于 trace 点朝向）。
- 数据集侧已实现 prompt 构造：
  - `target_xy`：从最长 GT 线提取起终点。
  - `trace_points`：从最长 GT 线采样 trace 点和角度。
- 训练损失接入方式：
  - prompt token 作为 decoder prefix 条件输入；
  - 仅对 `G_n` 目标段计算 loss（prefix 不计入 loss）。
- 对比评估产物：
  - `docs/text_prompt_full_vs_target_xy.md`
  - `docs/text_prompt_full_vs_trace.md`
  - `docs/text_prompt_target_xy_vs_trace.md`
  - 当前 smoke 规模下三者指标接近（均欠拟合，预测线趋近 0）。

8. 论文主干骨架接入（开始阶段）
- 新增 `unimapgen/models/unimapgen_paper.py` 与 `model.arch: paper` 工厂路由。
- 目标骨架：
  - BEV 编码：`facebook/dinov2-large`
  - 生成主干：`Qwen/Qwen2.5-1.5B-Instruct`
  - BEV/PV/Text prompt token 采用 prefix 融合。
- 现阶段阻塞：
  - 环境无法访问 `huggingface.co`，权重下载失败（网络不可达），导致论文主干无法在当前环境真正启动训练。
