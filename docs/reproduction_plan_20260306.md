# UniMapGen 复现路线（2026-03-06）

## 1. 论文关键设定（从 `UniMapGen.pdf` 提取）

1. 任务形式：自回归生成向量地图，支持 `BEV/PV/Text/Previous Map` 任意组合输入。  
2. 序列化：  
   - 等距采样，默认间隔 `6m`。  
   - 线按“首点到原点距离”重排。  
   - 端点有 `start/end/cut` 语义用于状态更新。  
3. 状态更新：推理按“左到右、上到下”扫描，当前 patch 由上一步地图提示。  
4. 主干：`DINOv2-Large`（BEV）+ `Qwen2-VL-ViT`（PV）+ `Qwen2.5-1.5B`（生成）。  
5. 数据与训练：  
   - OpenSatMap20：`1180 train / 393 val`，`4096x4096`。  
   - patch 尺寸 `896x896`。  
   - 通过 overlapped + inclined crop + 旋转扩增到约 `700k` patch。  
   - 训练配置：6 epochs，batch 32，AdamW(weight_decay=0.1)，peak lr=2e-5，warmup=100。  
6. 指标：`mIoU`、`Mask AP`、`Chamfer AP`（论文定义）。

## 2. 现有代码审计结论

已对齐：
1. 序列化闭环（等距采样、重排、token 化、反序列化）。  
2. `train/eval/predict/infer_state_scan` 全流程。  
3. `state update` 与 `cut` 端点机制（工程近似版）。  
4. `paper` 架构入口 + 10k 量化 token 空间配置。  

未完全对齐：
1. 当前主要是 nuScenes 风格数据管线，尚非 OpenSatMap 官方训练管线。  
2. PV 编码器目前是轻量近似实现，不是论文同款 Qwen2-VL-ViT 链路。  
3. 评估脚本仍是工程近似指标，不是论文官方评测实现。  
4. 配置路径原先硬编码；当前轮已改为环境变量可覆盖。  

## 3. 本轮已落地改造

1. `load_yaml` 支持环境变量：
   - `${VAR}`
   - `${VAR:-default}`
2. 新增数据自检脚本：`python -m unimapgen.check_data --config <yaml>`  
3. 新增 Linux 一键脚本：`scripts/run_paper_3stage_smoke.sh`  
4. paper 相关配置改为环境变量可覆盖：
   - `NUSCENES_ROOT`
   - `NUSCENES_MAP_PKL_DIR`
   - `SATMAP_ROOT`

## 4. 可执行复现路线

### 阶段 A：环境与数据打通（P0）
1. 准备路径并导出环境变量：
```bash
export NUSCENES_ROOT=/your/path/nuscenes
export NUSCENES_MAP_PKL_DIR=/your/path/maptrv2_pkl
export SATMAP_ROOT=/your/path/opensatmap_nuscenes_cropped
```
2. 先做数据体检（必须通过）：
```bash
python -m unimapgen.check_data --config configs/unimapgen_paper_stage1_sft_smoke.yaml
```

### 阶段 B：机制 smoke（P0->P1）
1. 跑三阶段 smoke：
```bash
bash scripts/run_paper_3stage_smoke.sh
```
2. 产物检查：
   - `outputs/unimapgen_paper_stage{1,2,3}_smoke/latest.pt`
   - `outputs/unimapgen_paper_stage3_state_smoke/metrics.jsonl`
   - `outputs/paper_stage3_state_scan.json`

### 阶段 C：论文参数对齐（P1）
1. 输入尺寸从 `512` 升级到 `896`。  
2. 增加 overlapped+inclined crop 数据准备流程。  
3. 用论文指标替换工程近似指标（mIoU/APM/APCD）。  
4. 将 `use_fallback` 逐步切换为真实 backbone（需本地权重或联网下载）。

### 阶段 D：结果逼近（P1->P2）
1. 按论文训练 recipe 放大 batch/epoch。  
2. 做关键消融：`state update / reorder / augmentation / modality mask`。  
3. 形成最终复现实验表格与可视化。

## 5. 当前阻塞

在本机 `/mnt/data/project/jn/UniMapGen` 下，默认路径：
- `/media/winkness/data/nuscenes`
- `/media/winkness/data/nuscenes/maptrv2_pkl`
- `/media/winkness/code/jn/HRMapNet/data/opensatmap_nuscenes_cropped`

均不存在，且缺 `nuscenes_map_infos_temporal_{train,val}.pkl`。  
因此目前无法在真实数据上继续训练，需先补齐路径与标注文件。
