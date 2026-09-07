# UniMapGen ms-swift GRPO 交接说明

日期：2026-09-04

本文说明 Context512/ROI256 路线的 ms-swift GRPO 实验入口。它是独立实验，
不改变现有 SFT 入口、奖励实现或自定义 Ray/vLLM GRPO 路线。

## 1. 目标与边界

当前实现针对：

```text
Context512/ROI256 图像
  -> 原始 DINOv2-Large（518 输入，倒数第二层 patch token）
  -> mlp2x_gelu projector
  -> CapRL-Qwen3VL-4B 派生文本 LLM
  -> ms-swift GRPO
```

- 不使用 Qwen3-VL 原生视觉塔。
- 不启用 DeepStack，也不压缩 DINOv2 token。
- 默认冻结 DINOv2 和 projector，只给 LLM 注入新的 LoRA；这与当前第一版
  GRPO 的低风险实验目标一致。
- 可选地从已有 UniMapGen SFT checkpoint 启动，然后再挂载 GRPO LoRA。

## 2. 文件职责

| 文件 | 职责 |
|---|---|
| `scripts/rl/swift_unimapgen_plugin.py` | 注册 Swift 模型、模板和地图 ORM；复用项目的 DINOv2、projector 和模型 loader |
| `mllm/reward/swift_map_reward.py` | 计算格式、line F1 和坐标精度奖励 |
| `scripts/tools/prepare_swift_grpo_dataset.py` | 将训练集推理 JSONL 转成 Swift 的 `messages/images/solution` 格式 |
| `scripts/tools/validate_swift_grpo_dataset.py` | 训练前检查图片、GT、坐标合同和唯一 ID |
| `scripts/npu/train/train_swift_grpo_stage_a_context512_roi256_550k_npu.sh` | DI/NPU 一条命令入口 |

## 3. 数据准备

默认直接使用 Context512/ROI256 550k 原始训练集，不需要先运行 SFT 模型推理：

```text
obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/context512_roi256/context512_roi256_550k.tar
```

入口会自动下载、解压并定位 `phase_a/train.jsonl`。也可以直接设置
`DATASET_ROOT` 或 `TRAIN_JSONL`。若后续要按 SFT 错误构建 hard pool，才使用
训练集推理结果；其中模型的 `prediction/raw_prediction` 只用于审计，不能作为
GRPO 的参考答案：

```bash
bash scripts/npu/test/test_stage_a_lane_intersection_context512_roi256_550k_trainset_npu.sh
```

然后入口脚本会自动执行：

```text
phase_a/train.jsonl 或 train_predictions.jsonl
  -> prepare_swift_grpo_dataset.py
  -> validate_swift_grpo_dataset.py
  -> Swift GRPO
```

转换后的每行至少包含：

```json
{
  "messages": [{"role": "user", "content": "<image>\n..."}],
  "images": ["images/train/...png"],
  "solution": "{\"lines\": [...]}",
  "ground_truth": "{\"lines\": [...]}",
  "coord_config": {
    "coord_mode": "norm1000",
    "coord_range": 1000,
    "patch_width": 256,
    "patch_height": 256
  },
  "map_task": "lane_intersection",
  "sample_id": "..."
}
```

推理输出同时有绝对 `image` 和相对 `image_relpath` 时，转换器优先使用
`image_relpath`，所以可以把 JSONL 和数据根目录分别搬到 DI 节点。

## 4. 奖励

第一版只保留三个互补项：

```text
reward = 0.75 * line_f1
       + 0.20 * coordinate_quality
       + 0.05 * format_reward
```

- `line_f1` 是 instance-level F1 与 length-level F1 的平均值，已经同时惩罚
  漏线和多线，因此不再额外添加数量 penalty。
- `coordinate_quality` 对一对一匹配的中心线做弧长重采样，按 GT 线长加权，
  用米制平均距离的高斯衰减奖励坐标精度。
- `format_reward` 只有在输出是无围栏、无额外文本的 JSON 对象且含
  `lines` 数组时为 1。
- 解析失败返回 `-1.0`，让格式错误候选明显劣于可解析候选。
- 路口 polygon 暂不参与这一版 reward；先验证中心线 reward 是否能改善
  instance F1，再扩展奖励面。

## 5. DI 入口

DI 镜像中需要预装并固定可用的 `ms-swift`，入口不会在运行时改写 torch、
torch_npu 或 CANN。下面是 smoke 示例；路径必须替换成 DI 实际挂载路径：

```bash
INFERENCE_JSONL=/cache/outputs/<train-inference>/train_predictions.jsonl \
IMAGE_ROOT=/cache/datasets/context512_roi256_550k \
MODEL_PATH=/cache/models/caprl-qwen3vl-4b-derived \
UNIMAPGEN_VISION_TOWER=/cache/models/facebook_dinov2-large \
ACTIVATE_SCRIPT=/path/to/activate_mllm_npu.sh \
MAX_STEPS=20 \
NUM_TRAIN_EPOCHS=1 \
NUM_GENERATIONS=4 \
OUTPUT_DIR=/cache/outputs/unimapgen_swift_grpo_smoke \
bash scripts/npu/train/train_swift_grpo_stage_a_context512_roi256_550k_npu.sh
```

正式实验将 `MAX_STEPS` 改为 `-1`，并显式设置 epoch、输出目录和所需的
`PER_DEVICE_TRAIN_BATCH_SIZE`。入口默认值为：

| 参数 | 默认值 |
|---|---:|
| `NUM_GENERATIONS` | 4 |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | 1 |
| `GRADIENT_ACCUMULATION_STEPS` | 1 |
| `MAX_LENGTH` | 4096 |
| `MAX_COMPLETION_LENGTH` | 2048 |
| `LEARNING_RATE` | 1e-6 |
| `MAX_STEPS` | 20 |
| `NUM_TRAIN_EPOCHS` | 1 |
| `SAVE_STEPS` | 100 |

如果从已有 SFT 结果继续，在上面的命令中增加：

```bash
UNIMAPGEN_START_CHECKPOINT=/cache/models/unimapgen_sft/checkpoint-xxxx
```

`MODEL_PATH` 仍应指向基础 CapRL 文本模型。插件会识别单文件全参 checkpoint、
标准 LoRA adapter，以及没有 `config.json` 但包含
`adapter_config.json + non_lora_trainables.bin` 的历史 LoRA 导出；缺少后者
时会直接失败，不会静默退回基础模型。

## 6. 输出与复现

入口在 `WORK_ROOT` 下保存：

```text
swift_grpo_conversion_summary.json
swift_grpo_dataset_validation.json
swift_command.txt
swift_grpo.jsonl
```

Swift 的模型 checkpoint 位于 `OUTPUT_DIR`。入口开始和结束都会打印：

```text
DI_throughput: ... samples/s/npu
```

训练过程先写入本地 `OUTPUT_ROOT`。训练成功退出后，若 DI 提供了
`OUTPUT_URL`，入口会按现有训练脚本约定将完整目录复制到
`OUTPUT_URL/<RUN_ID>`；也可以用 `GRPO_RESULT_OBS` 显式指定 OBS 目标。
因此 GRPO adapter、转换数据、校验报告和 `swift_command.txt` 会一起保存。

## 7. 当前验证状态与限制

已完成：

- 插件、转换器、校验器和 reward 的 Python 静态编译。
- reward、GT/预测隔离、相对图片路径、缺图 fail-closed 的 7 个轻量测试。

尚未在本 Windows 工作站完成真实 Swift + Ascend NPU/DI 运行。因此第一次
上 DI 必须先使用 `MAX_STEPS=20` 的 smoke，重点确认：

1. Swift 能导入 external plugin 并看到 `unimapgen_qwen3_dinov2`。
2. 模型日志显示 DINOv2 tower、projector 和正确的 NPU device。
3. ORM 收到 `solution`、`coord_config`，并打印非恒定 reward。
4. 首个 checkpoint 能正常写出，且输出包含 GRPO adapter。

真实 smoke 通过后再扩大样本量或开启更长训练。
