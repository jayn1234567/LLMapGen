# DINOv2 ViT-L/14 + Map Serialization Pipeline

## 1. 目标范围

本阶段先对齐论文主干并保证可运行：

1. 卫星图特征分支：`DINOv2 ViT-L/14`
2. Map Serialization：折线重采样 + token 序列化/反序列化
3. 自回归预测道路线（可训练、可评估、可推理）

暂不纳入：

- `State Update` 训练/推理闭环（后续在大模型路径中加入）

## 2. 关键实现

### 2.1 DINOv2 ViT-L/14 分支

- `unimapgen/models/encoders/satellite_encoder.py`
  - 支持 `facebook/dinov2-vitl14`
  - 输入归一化（mean/std）
  - 可选去掉 CLS token
  - Patch token 自适应池化到固定 token 网格（默认 `8x8`）
  - 若本地没有 DINO 权重，自动回退到轻量 CNN 分支，保证 pipeline 可跑

### 2.2 Map Serialization

- `unimapgen/data/serialization.py`
  - polyline 重采样
  - `start/end/cut` 边界语义
  - 类别 token + 坐标 token + 角度 token
  - `encode/decode` 全流程

### 2.3 论文主干模型入口

- `unimapgen/models/unimapgen_paper.py`
- `unimapgen/models/__init__.py`
  - `arch: paper` 下默认走 DINOv2 配置项

## 3. 样例可运行配置

- 配置：`configs/unimapgen_dinov2_l14_mapser_unaligned_smoke.yaml`
- 一键脚本：`scripts/run_dinov2_mapser_unaligned_smoke.sh`

执行：

```bash
cd /mnt/data/project/jn/UniMapGen
bash scripts/run_dinov2_mapser_unaligned_smoke.sh
```

流程包含：

1. 数据检查
2. 训练
3. 评估
4. 推理导出

## 4. 后续与 State Update 衔接

等你提供对齐后的卫星数据集后，直接在现有 pipeline 上继续：

1. 先切换对齐数据路径
2. 保持 DINOv2 + Map Serialization 主干
3. 再接入 `use_state_update=true` 的数据前缀构造与训练
