# 基于未对齐卫星图的样例复现路线

## 1. 当前可用样例数据

已从未对齐裁剪卫星图中筛出并标注一批样例：

- 数据根目录：`/mnt/data/project/jn/UniMapGen/data_samples/unaligned_sat_examples`
- 训练集：`train/`（18 张）
- 验证集：`val/`（6 张）
- 标注文件：`annotations.json`
- 划分记录：`splits_meta.json`
- 叠加预览：`preview/`

说明：

- 标注来源于 OpenSatMap 真值提取脚本，类别包含 `Lane line / Curb / Virtual line`。
- 该样例集用于先打通 UniMapGen 端到端复现流程，不用于最终精度结论。

## 2. 直接运行复现流程

```bash
cd /mnt/data/project/jn/UniMapGen
bash scripts/run_unaligned_sat_samples_tiny.sh
```

该脚本会依次执行：

1. `check_data`
2. `train`
3. `eval`
4. `predict`

默认配置文件：

- `configs/unimapgen_unaligned_sat_samples_tiny.yaml`

## 3. 后续切换到对齐数据集

当你提供对齐后的卫星数据集时，只需替换以下三项路径并保持同一配置结构：

- `data.opensatmap_root`
- `data.opensatmap_ann_json`
- `data.opensatmap_split_dir`

其他训练/模型配置可先保持不变，再逐步扩大样本量与分辨率。
