# UniMapGen Stage A Portable Min Discrete

这是一份按 `UniMapGen-StageA-Portable-Min` 目录风格整理的离散 token 便携包。

目标是：

- 拿到这整个目录后，不需要再依赖原始仓库
- 只要把模型和数据放到约定位置
- 就能直接使用：
  - 全参训练
  - LoRA 训练
  - 评估 / 推理 / 可视化

当前默认对齐的离散 token 主线配置：

- `Qwen2.5-VL-3B`
- `Stage A`
- `shared_numbers`
- `coord_num_bins = 896`
- `disable legacy text prompt tokens`

目录说明：

- `ckpts/`
  - 放基础模型
- `dataset/`
  - 放训练 / 验证数据集和 `images/`
- `configs/`
  - 放环境变量样例
- `scripts/`
  - 放训练 / 评估入口
- `unimapgen/`
  - 放这份包自己需要的最小代码
- `outputs/`
  - 默认训练和评估输出
- `logs/`
  - 默认日志目录

主要入口：

- 全参训练：
  - `scripts/launch_stagea_discrete_train.sh`
- LoRA 训练：
  - `scripts/launch_stagea_discrete_train_lora.sh`
- 评估 / 推理 / 可视化：
  - `scripts/launch_stagea_discrete_eval.sh`

建议先读：

- `STAGEA_PORTABLE_USAGE.md`

