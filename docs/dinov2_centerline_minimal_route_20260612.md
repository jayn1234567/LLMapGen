# DINOv2 中心线预测最小可运行路线

更新日期：2026-06-12

这份整理版只保留 DINOv2 路线里最有效、最稳定的中心线 JSON SFT 分支：

1. 输入：RC road-structure patch 图像。
2. 视觉编码：DINOv2 ViT-L/14。
3. 桥接：DINOv2 visual tokens -> Qwen hidden states。
4. 输出：中心线 JSON，格式为 `{"lines":[{"points":[[x,y],[x,y]]}]}`。
5. 真值处理：Douglas 简化 + 近端点近方向 fragment merge。

## 推荐入口

只推荐使用两个脚本：

1. 训练：[scripts/train_dinov2_centerline.py](../scripts/train_dinov2_centerline.py)
2. 推理：[scripts/predict_dinov2_centerline.py](../scripts/predict_dinov2_centerline.py)

其它历史脚本仍保留在仓库里用于追溯实验，但不再作为这条路线的主入口。

## 代码结构

核心整理代码：

1. [unimapgen/dinov2_centerline/data.py](../unimapgen/dinov2_centerline/data.py)
2. [unimapgen/dinov2_centerline/model.py](../unimapgen/dinov2_centerline/model.py)
3. [unimapgen/dinov2_centerline/train.py](../unimapgen/dinov2_centerline/train.py)
4. [unimapgen/dinov2_centerline/predict.py](../unimapgen/dinov2_centerline/predict.py)

复用的稳定底座：

1. [unimapgen/models/qwen3_rc_dinov2_centerline_json_sft.py](../unimapgen/models/qwen3_rc_dinov2_centerline_json_sft.py)
2. [unimapgen/data/rc_centerline_json_sft_dataset.py](../unimapgen/data/rc_centerline_json_sft_dataset.py)
3. [unimapgen/data/rc_centerline_douglas_utils.py](../unimapgen/data/rc_centerline_douglas_utils.py)
4. [unimapgen/rc_llm_runtime.py](../unimapgen/rc_llm_runtime.py)

## 数据处理

训练脚本内置 trainroot 清理能力：

1. `--prepare-trainroot`：启用数据清理。
2. `--douglas-epsilon-px 2.5`：默认 Douglas 采样参数。
3. `--merge-endpoint-tol-px 6.0`：默认近端点拼接阈值。
4. `--merge-heading-tol-deg 22.5`：默认近方向拼接阈值。

输入 trainroot 需要包含：

```text
train.jsonl
meta_train.jsonl
val.jsonl
meta_val.jsonl
```

如果启用 `--prepare-trainroot`，会生成一个新的 prepared trainroot，不会覆盖原数据。

## Tiny Smoke 数据

仓库内提供了一个很小的合成 RC-style trainroot，用于验证这条路线可以独立读取数据并启动训练流程：

[data_samples/dinov2_centerline_tiny_trainroot](../data_samples/dinov2_centerline_tiny_trainroot)

内容：

1. `train.jsonl`：6 条训练样本。
2. `meta_train.jsonl`：训练样本中心线真值。
3. `val.jsonl`：2 条验证样本。
4. `meta_val.jsonl`：验证样本中心线真值。
5. `images/`：8 张 `512x512` 合成 RC 风格 PNG。

已验证：

1. `--prepare-trainroot` 可以成功把 tiny 数据重写成 Douglas + merge trainroot。
2. DINOv2 centerline model 相关 Python 模块可以成功导入。
3. 新增入口脚本和依赖模块通过 `py_compile`。

注意：

1. tiny 数据不包含模型权重。
2. 真实 `max_steps=1` 训练需要传入完整 Hugging Face CausalLM 权重和 DINOv2 权重。
3. 当前超算上可见的 `Qwen3-8B` / `Qwen3-VL-*` 目录不是完整 HF 权重目录，因此本次只能验证到数据准备和模型导入层；有完整 Qwen CausalLM 后可直接运行下面训练命令。

## 训练示例

```bash
cd /file_storage01/home/mingli/project/jn/UniMapGen

python scripts/train_dinov2_centerline.py \
  --model-name-or-path /path/to/Qwen3-8B \
  --dinov2-model-name-or-path /path/to/dinov2_vitl14 \
  --trainroot /file_storage01/home/mingli/data/rc_centerline_trainroot \
  --prepare-trainroot \
  --prepared-trainroot /file_storage01/home/mingli/data/rc_centerline_trainroot_douglas_merge \
  --output-dir /file_storage01/home/mingli/data/outputs/dinov2_centerline_json_sft_minimal \
  --num-train-epochs 6 \
  --learning-rate 1e-4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 2 \
  --bf16 \
  --gradient-checkpointing \
  --save-strategy steps \
  --save-steps 1000 \
  --save-total-limit 3 \
  --local-files-only
```

多卡训练时仍可用 `torchrun` 或 sbatch 外层启动，例如：

```bash
torchrun --nproc_per_node=4 scripts/train_dinov2_centerline.py ...
```

## 推理示例

```bash
cd /file_storage01/home/mingli/project/jn/UniMapGen

python scripts/predict_dinov2_centerline.py \
  --checkpoint-dir /file_storage01/home/mingli/data/outputs/dinov2_centerline_json_sft_minimal/checkpoint-27000 \
  --trainroot /file_storage01/home/mingli/data/rc_centerline_trainroot_douglas_merge \
  --split val \
  --output-jsonl /file_storage01/home/mingli/data/outputs/dinov2_centerline_json_sft_minimal/pred_val100.jsonl \
  --max-samples 100 \
  --shuffle \
  --seed 42 \
  --local-files-only
```

输出 jsonl 每行包含：

1. `id`
2. `image`
3. `gt_lines`
4. `pred_lines`
5. `gt_json`
6. `pred_json`
7. `raw_prediction_text`
8. `parse_ok`
9. `num_gt_lines`
10. `num_pred_lines`

该格式与之前中心线评估和可视化脚本兼容。

## 保留与删除的边界

保留：

1. DINOv2 visual encoder。
2. DINOv2->Qwen visual token bridge。
3. centerline JSON SFT。
4. Douglas + merge 真值处理。
5. LoRA 训练。
6. checkpoint 推理。

不放入最小路线：

1. caption bridge 实验。
2. road-structure JSON/Bezier/quad4 实验。
3. adaptive keypoint GRPO 实验。
4. continuous coordinate head 实验。
5. 各种临时评估、smoke、可视化恢复脚本。

这些分支不是没价值，而是不应该混在“DINOv2 中心线预测最小可运行代码”里。
