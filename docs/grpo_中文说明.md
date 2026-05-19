# GRPO 强化学习说明

本文档记录当前项目里的 GRPO 训练方式、奖励设计、数据格式、脚本入口和已完成的 smoke test。当前 GRPO 是项目自定义的 image-aware 实现；它不是直接调用 TRL/HF 原生 `GRPOTrainer` 完成训练。

## 当前实现

入口：

```bash
python -m mllm.train.train_grpo
```

这个入口是兼容 wrapper。真实实现也可以直接这样启动：

```bash
python -m mllm.train.rl.grpo
```

核心文件：

- `mllm/train/rl/grpo.py`：GRPO 训练主逻辑。
- `mllm/train/train_grpo.py`：兼容旧脚本的入口 wrapper。
- `mllm/reward/map_schema.py`：解析模型输出 JSON。
- `mllm/reward/map_reward.py`：计算 map reward。
- `infer_index/line_eval.py`：中心线几何评估指标，GRPO reward 也复用这里的线匹配逻辑。

训练时会加载两份模型：

- `policy model`：当前要训练的模型，默认只在 LoRA 参数上更新。
- `reference model`：冻结的 SFT 参考模型，用于 KL 约束。

每个 prompt 会采样 `NUM_GENERATIONS` 个候选输出，对每个候选计算 reward，再做 group-relative advantage。当前实现是轻量 smoke/debug 版本，重点是跑通图像输入、JSON 输出、reward、LoRA 保存和 checkpoint 推理闭环。

## TRL/HF 后端状态

`--grpo_backend custom` 是当前默认路径，也是实际验证过的路径。

`--grpo_backend trl` 目前只做 TRL 是否安装的检查，然后仍然走项目自己的 image-aware batch 适配逻辑。原因是原生 TRL trainer 不知道本项目的 `images` / `image_sizes` 多模态输入格式，也不知道 Qwen 多模态 checkpoint 的保存元数据。因此在真正切到 TRL 原生实现前，还需要专门写 adapter 和验证 checkpoint 兼容性。

## 数据格式

GRPO 数据沿用 SFT JSONL 格式：

```json
{
  "id": "sample_id",
  "image": "images/xxx.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nPlease construct the complete road map in the current BEV (Bird's Eye View) image patch.\n..."
    },
    {
      "from": "gpt",
      "value": "{\"lines\":[{\"category\":\"centerline\",\"start_type\":\"cut\",\"end_type\":\"inside\",\"points\":[[0,120],[80,130]]}]}"
    }
  ]
}
```

支持两个任务：

- `lane`：只预测中心线，输出 `category=centerline`。
- `lane_intersection`：同时预测中心线和路口多边形，输出 `category=centerline` 与 `category=intersection`。

debug 数据生成脚本：

```bash
python scripts/gpu/build_grpo_debug_data.py --limit 20 --test-count 4
```

生成路径：

- `data/grpo_debug_lane20/train.jsonl`
- `data/grpo_debug_lane20/test.jsonl`
- `data/grpo_debug_lane_intersection20/train.jsonl`
- `data/grpo_debug_lane_intersection20/test.jsonl`

其中 `lane_intersection` debug 数据里的路口是 synthetic polygon，只用于验证代码链路，不代表真实路口标注质量。

## Reward 组成

`lane` 默认 reward：

- `format`：输出是否能解析为合法 JSON，字段和点坐标是否合法。
- `centerline_instance_f1`：中心线实例匹配 F1，来自 `infer_index/line_eval.py`。
- `centerline_length_f1`：中心线长度匹配 F1，来自 `infer_index/line_eval.py`。
- `cut_type`：预测线段的 `start_type/end_type` 是否和 GT 对齐。
- `cut_continuity`：如果预测为 `cut`，端点是否贴近 patch 边界。

`lane_intersection` 在上述基础上增加：

- `intersection`：当前先用路口数量差的简单分数做 smoke reward。后续可以升级为 polygon IoU、边界 cut 连续性、路口类型分类等。

常用参数：

| 参数 | 含义 |
|---|---|
| `MAP_TASK` | `lane` 或 `lane_intersection`。 |
| `COORD_MODE` | 坐标模式，默认 `auto`，会从数据 `meta.coord_mode` 读取；新数据默认是 `norm1000`。 |
| `COORD_RANGE` | 归一化坐标范围，新数据默认 `1000`。 |
| `NUM_GENERATIONS` | 每个 prompt 采样候选数。debug 默认为 2，正式可调大。 |
| `KL_BETA` | policy 与 frozen reference 的 KL 约束权重。 |
| `CLIP_RANGE` | GRPO/PPO 风格 ratio clip 范围。 |
| `REWARD_*_WEIGHT` | reward 各组件权重。 |
| `LORA_TARGET_SCOPE` | LoRA 插入模块，可选 `llm`、`projector`、`vision`、`deepstack`、`all`。 |

坐标说明：GRPO reward 会先按数据坐标模式解析预测和 GT；如果是 `norm1000`，会转换回 patch 像素坐标后再调用 `infer_index/line_eval.py`。因此强化学习优化的几何指标和推理后的评估指标保持一致。

## 正式 NPU 脚本

正式训练优先使用 DINOv2 + Qwen3VL + no-DeepStack 的云端脚本：

```bash
bash scripts/npu/train_grpo_dinov2_qwen3vl-8b_lora_nodeepstack_npu.sh
```

这个脚本是当前推荐的 production GRPO 入口，默认是 LoRA GRPO，内部已经包含：

- `SFT_CHECKPOINT_OBS` / `SFT_CHECKPOINT`：从稳定 SFT checkpoint 开始强化学习。
- `DATASET_PHASE`：选择 `phase_a` 或 `phase_b` 数据。
- `MAP_TASK`：选择 `lane` 或 `lane_intersection`。
- `TRAINING_BRANCH`：严格指定 `phase_a_lane`、`phase_b_lane`、`phase_a_lane_intersection` 或 `phase_b_lane_intersection`。
- `COORD_MODE=auto`、`COORD_RANGE=1000`：自动兼容新 `norm1000` 数据和旧 pixel 数据。
- `eval.jsonl`：直接使用数据处理阶段按原始样本级切好的验证集；不再从 `test.jsonl` 动态切分。
- `LORA_TARGET_SCOPE`、`LORA_R`、`LORA_ALPHA`、`LORA_DROPOUT`：LoRA 插入范围和规模。
- `NUM_GENERATIONS`、`KL_BETA`、`REWARD_*_WEIGHT`：GRPO 采样、KL 和 reward 权重。
- `DEEPSPEED_CONFIG=scripts/deepspeed_zero3.json`：NPU 多卡 ZeRO-3 训练。

还有两个 DINOv3 no-DeepStack 模板脚本，主要用于后续切 DINOv3 实验时参考：

```bash
bash scripts/npu/train_grpo_dinov3_qwen3vl-8b_lora_nodeepstack_auto_lane_npu.sh
bash scripts/npu/train_grpo_dinov3_qwen3vl-8b_lora_nodeepstack_auto_lane_intersection_npu.sh
```

这些脚本同样不要通过外部一次性环境变量临时覆盖核心参数；正式实验前直接编辑脚本顶部参数块，保证实验配置可复现。

## GPU Debug 脚本

当前重点脚本是 DINOv2 + Qwen3VL + no DeepStack：

```bash
# 单卡 lane
bash scripts/gpu/train_grpo_debug_lane_dinov2_qwen3vl_nodeepstack_gpu.sh

# 单卡 lane + intersection
bash scripts/gpu/train_grpo_debug_lane_intersection_dinov2_qwen3vl_nodeepstack_gpu.sh

# GPU0 + GPU2 DeepSpeed lane
bash scripts/gpu/train_grpo_debug_lane_dinov2_qwen3vl_nodeepstack_deepspeed_gpu.sh

# GPU0 + GPU2 ZeRO-3 lane
bash scripts/gpu/train_grpo_debug_lane_dinov2_qwen3vl_nodeepstack_zero3_gpu.sh

# GPU0 + GPU2 ZeRO-3 lane + intersection
bash scripts/gpu/train_grpo_debug_lane_intersection_dinov2_qwen3vl_nodeepstack_zero3_gpu.sh
```

这些脚本内部已经写好主要参数，不需要在命令行外面临时传环境变量。需要改模型、数据、学习率、LoRA 范围、reward 权重时，直接改脚本顶部参数块。

脚本默认 checkpoint：

- `SFT_CHECKPOINT=/media/q/data2/jjh/project/MLLM_project/outputs/test_qwen3vl`
- `VISION_TOWER=/media/q/data2/jjh/project/MLLM_project/checkpoints/facebook_dinov2-large`

正式实验时建议把 `SFT_CHECKPOINT` 改成已经 SFT 稳定收敛的 checkpoint，而不是 base model。

## 本地 NPU Debug 脚本

本地 NPU smoke test 入口：

```bash
bash scripts/npu/train_grpo_debug_lane_dinov2_qwen3vl_nodeepstack_local_npu.sh
```

这个脚本不安装依赖、不下载 OBS，只使用本地路径。运行前需要直接编辑脚本顶部参数块：

- `CONDA_ENV`：必须是已经安装 `torch_npu` 的 NPU 环境。
- `NPU_IDS` / `NPROC_PER_NODE`：本地使用的 NPU 编号和进程数。
- `SFT_CHECKPOINT`：稳定 SFT checkpoint。
- `VISION_TOWER`：DINOv2 checkpoint。
- `TRAIN_JSONL` / `TEST_JSONL` / `IMAGE_FOLDER`：debug 数据和图片目录。

默认是 1 step、LoRA、lane-only、no DeepStack。为了减少单 NPU 显存占用，默认 `KL_BETA=0.0`，不会额外加载 reference model。若要测试 LoRA ZeRO3 reference KL，可以在脚本内设置多 NPU、`DEEPSPEED_CONFIG=scripts/deepspeed_zero3.json`，并把 `KL_BETA` 调到大于 0。

## Checkpoint 和推理

GRPO 默认保存 LoRA checkpoint。正常输出目录包含：

- `checkpoint-*/adapter_model.safetensors`
- `checkpoint-*/adapter_config.json`
- `checkpoint-*/non_lora_trainables.bin`
- `checkpoint-*/config.json`
- `checkpoint-*/qwen_multimodal_checkpoint.json`

推理脚本可以直接加载 `checkpoint-*`：

```bash
python scripts/infer_centerline_checkpoint.py \
  --checkpoint-dir outputs/grpo_debug_lane_dinov2_qwen3vl_nodeepstack_gpu/checkpoint-1 \
  --vision_tower /media/q/data2/jjh/project/MLLM_project/checkpoints/facebook_dinov2-large \
  --input_image_size 518 \
  --disable_deepstack \
  --test-json data/grpo_debug_lane20/test.jsonl \
  --image-folder data/av2_patch_256_fullimage_cutflag_test_v2 \
  --prompt-mode dataset \
  --conv-template conv_qwen_3_Dinov2_huawei \
  --device cuda \
  --eval-centerline
```

`--eval-centerline` 默认会在推理结束后写出 `summary_centerline_eval.json`；如果传入
`--eval-output-json`，指标会写到指定文件。云端 NPU 测试脚本约定输出为
`test_results/summary.json`、`test_results/centerline_eval.json`、`test_results/json/`
和 `test_results/viz/`。

## 已验证情况

已完成 smoke test：

- 单卡 `lane`：`DINOv2 + Qwen3VL + no DeepStack`，训练 1 step，checkpoint 推理通过。
- 单卡 `lane_intersection`：`DINOv2 + Qwen3VL + no DeepStack`，训练 1 step，checkpoint 推理通过。
- 多卡 DeepSpeed：物理 `GPU0,2`，`DINOv2 + Qwen3VL + no DeepStack`，ZeRO-2，训练 1 step，checkpoint 推理通过。
- 多卡 ZeRO-3：物理 `GPU0,2`，`DINOv2 + Qwen3VL + no DeepStack`，`lane` 与 `lane_intersection` 均训练 1 step，checkpoint 推理通过。
- DINOv3 单卡 `lane` 和 `lane_intersection` debug 脚本也完成了训练和 checkpoint 推理 smoke test。

当前 smoke test 只验证工程链路。由于只训练 1 step，模型输出仍可能不是合法 JSON，中心线评估结果为 0 是正常现象，不代表正式 GRPO 训练效果。

ZeRO-3 注意事项：

- LoRA ZeRO-3 现在可以使用 reference KL：不会额外加载一份 frozen reference model，而是在同一个 DeepSpeed-wrapped policy 上临时关闭 LoRA adapter，把“base SFT 模型”当作 reference。
- 因此 LoRA ZeRO-3 debug 脚本可以保持 `KL_BETA=0.02`。
- 全参 ZeRO-3 仍然不能直接使用 reference KL；如果 `--lora_enable False` 且 `--kl_beta > 0`，代码会提前报错。全参场景后续需要单独实现 DeepSpeed-wrapped reference engine。
- 单卡和 ZeRO-2 debug 脚本继续使用独立 frozen reference model。

Reward 权重原则：

- `lane` 阶段不会启用路口 reward。即使 `intersection` component 会被记录，只有 `map_task` 为 `lane_intersection`、`intersection` 或 `all` 时才会乘以 `reward_intersection_weight` 加入总 reward。
- `lane` 默认权重为：format 0.08，infer_index instance F1 0.37，infer_index length F1 0.45，cut type 0.05，cut continuity 0.05，intersection 0.00。线匹配指标合计 0.82，是主信号。
- `lane_intersection` 脚本默认权重为：format 0.07，infer_index instance F1 0.33，infer_index length F1 0.42，cut type 0.04，cut continuity 0.04，intersection 0.10。线匹配指标合计 0.75，路口只作辅助。

训练分支：

- 当前数据里的 A/B 不是 lane 到 lane+路口。A/B 表示是否带 state-update hints：`phase_a` 是空 incoming hints，`phase_b` 是 left/top incoming hints。
- `lane` / `lane_intersection` 是另一条独立维度，表示输出目标是只预测中心线，还是中心线+路口。
- 因此严格分支有 4 个：`phase_a_lane`、`phase_b_lane`、`phase_a_lane_intersection`、`phase_b_lane_intersection`。
- `auto_lane` 和 `auto_lane_intersection` 只检查任务类型，不检查 phase，适合当前 GRPO debug JSONL 这种没有明确 A/B 元数据的数据。
- 训练入口会在样本里存在 `map_task`、`task`、`phase` 或 `debug_phase` 元数据时做校验，避免把 Phase A 数据误喂给 Phase B 脚本，或把 lane+intersection 数据误喂给 lane-only 脚本。

推荐课程顺序：

1. 先做 `phase_a_lane`，学会单 patch 中心线 JSON 和局部几何。
2. 再做 `phase_b_lane`，学习中心线 state-update / 拼接。
3. 然后扩展到 `phase_a_lane_intersection`，加入路口输出但不加 state-update 难度。
4. 最后做 `phase_b_lane_intersection`，同时处理中心线、路口和 left/top hints。

## A/B 阶段关系

GRPO 数据格式与 SFT 一致，可以用于 Phase A 或 Phase B 数据：

- Phase A：输入 prompt 中 incoming lane/intersection hints 为空，目标是先学会单 patch 的局部几何和 JSON 输出格式。
- Phase B：输入 prompt 中包含 left/top 来的 incoming traces/intersections，目标是学习 state-update 场景下的连续性。

目前已实际跑通的是 SFT A/B smoke 链路；GRPO debug 数据也按 `lane` 与 `lane_intersection` 两种任务组织。正式接入 GRPO 时建议先用 Phase A 的稳定 SFT checkpoint 做起点，再切 Phase B 做连续性奖励。

## 后续建议

1. 先用稳定 SFT checkpoint 做 GRPO 起点，不建议从 base model 直接做 GRPO。
2. 先在 `lane` 上把 JSON 合法率、中心线 F1 拉起来，再扩展到 `lane_intersection`。
3. 正式 GRPO 前抽固定 validation set，每隔固定 step 推理并跑 `infer_index/line_eval.py`，不要只看训练 reward。
4. 路口 reward 后续应从简单数量分数升级到 polygon IoU 和 cut 边界一致性。
5. 如果要接入 TRL 原生 GRPO，需要先实现多模态 collator、generate 包装、checkpoint metadata 保存和 LoRA/DeepSpeed 兼容验证。
