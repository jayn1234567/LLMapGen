# UniMapGen 基础复现框架（OpenSatMap优先，无 maptrv2_pkl）

## 1. 基础代码框架（已落地）

### 1.1 数据层
1. `data.source: opensatmap`  
2. 数据读取：`annotrainval20.json + picuse20trainvaltest/{train,val}`  
3. 适配类：`unimapgen/data/opensatmap_dataset.py`  
4. 序列化：
   - 类别归一化：`Lane line/Curb/Virtual line -> lane_line/curb/virtual_line`
   - 按 `meter_per_pixel` 把 `6m` 转成采样像素间隔
   - 保留 `start/end/cut` 端点语义

### 1.2 模型层
1. 入口：`unimapgen/models/__init__.py` 的 `arch: paper`  
2. 结构（当前基础版）：
   - BEV Encoder（DINOv2 接口，离线 fallback 可跑）
   - LLM Decoder（Qwen2.5 接口，离线 fallback 可跑）
   - Prefix 融合接口预留给 PV/Text/State
3. 目标：先在 OpenSatMap BEV-only 上拿到稳定可训练曲线，再逐步补齐论文全模态。

### 1.3 训练/评估层
1. 训练：`python -m unimapgen.train --config configs/unimapgen_opensatmap_bev_smoke.yaml`
2. 评估：`python -m unimapgen.eval --config ... --checkpoint ...`
3. 推理导出：`python -m unimapgen.predict --config ... --checkpoint ...`
4. 一键脚本：`scripts/run_opensatmap_bev_smoke.sh`

### 1.4 数据体检层
1. 脚本：`python -m unimapgen.check_data --config <yaml>`
2. 对 `opensatmap` 支持检查：
   - split 文件是否存在
   - 图片是否可读
   - 注释是否匹配且非空

## 2. 复现路线（从可跑到论文对齐）

### 阶段 A：BEV-only 可跑基线（当前）
1. 使用 OpenSatMap20 train/val（1180/393）直接训练。  
2. 配置：`configs/unimapgen_opensatmap_bev_smoke.yaml`。  
3. 目标：拿到稳定下降的 train/val loss，导出可视化预测结果。  

### 阶段 B：扩展到论文训练设置
1. 输入分辨率升级到 `896x896`。  
2. 加入 overlapped + inclined crop 数据增强流程。  
3. 增大样本规模（接近论文 700k patch 方向）。  

### 阶段 C：多模态对齐（PV/Text）
1. PV 对齐：
   - 先实现 `GPS_info_all.json` 到 nuScenes 帧的样本配对工具（离线缓存成配对索引）。
   - 训练 `BEV+PV`，再开模态随机 mask。
2. Text 对齐：
   - 先做 `target_xy` 与 `trace_points` 两类 prompt。
   - 复用当前 prefix-loss 机制（prefix 不计 loss，只训练目标地图段）。

### 阶段 D：State Update 与全图构建
1. patch 扫描顺序按论文（left->right, top->bottom）。  
2. 用前状态图 `G_{n-1}` 引导当前 patch 生成。  
3. 输出全图并统计跨 patch 连续性指标。  

### 阶段 E：指标对齐
1. 从工程近似指标迁移到论文指标：`mIoU / Mask AP / Chamfer AP`。  
2. 补齐关键消融：`state update / reorder / augmentation / modality`。  

## 3. 你现在可以直接跑

```bash
cd /mnt/data/project/jn/UniMapGen
bash scripts/run_opensatmap_bev_smoke.sh
```

如果只想先验证数据：

```bash
python -m unimapgen.check_data --config configs/unimapgen_opensatmap_bev_smoke.yaml
```
