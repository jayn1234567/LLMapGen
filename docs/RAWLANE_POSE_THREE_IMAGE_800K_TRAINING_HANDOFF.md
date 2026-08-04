# Raw-Lane + Pose 三图 800k 训练交接

日期：2026-08-04

代码分支：`MLLM`

数据构建说明：`docs/DATASET_V2_RAWLANE_POSE_THREE_IMAGE_800K.md`

## 1. 训练目标

基于同一批 800,000 个 Stage A patch，分别完成两个独立实验：

| 实验 | 数据集 | 输入尺寸 | 监督区域 |
|---|---|---:|---:|
| A | `rawlane_pose_three_image_local256_800k` | 三张 256x256 图 | 整张 256x256 |
| B | `rawlane_pose_three_image_context512_roi256_800k` | 三张 512x512 图 | 中心 256x256 ROI |

两个数据集不能在同一次训练中混合。它们的 train/eval/test patch ID、
真值和固定评估大图保持一致，适合做“局部输入 vs 上下文输入”的对照实验。

每个样本的模型输入顺序固定为：

1. 干净 BEV 道路结构图，不叠加 Raw-Lane；
2. `patch_tif/0_lane.tif` 生成的黑底白线 Raw-Lane 图；
3. `patch_tif/0_pose.tif` 生成的黑底白线历史车辆轨迹图。

不得把第二张图重新画回第一张图，也不得交换 Raw-Lane 与 Pose 的顺序。

## 2. 数据产物

默认构建目录：

```text
D:\data\fulldata_rawlane_pose_three_image_800k\
  output_rawlane_pose_three_image_800k\
    rawlane_pose_three_image_local256_800k\
    rawlane_pose_three_image_context512_roi256_800k\
  packages_rawlane_pose_three_image_800k\
    rawlane_pose_three_image_local256_800k.tar
    rawlane_pose_three_image_context512_roi256_800k.tar
```

每个 TAR 内含一个同名顶层目录。DI 脚本解压后必须解析实际数据根目录，
不能假设 `phase_a` 一定位于解压目录的第一层。

开始训练前必须确认以下文件存在：

```text
dataset_root/
  images/{train,eval,test}/...
  raw_lane_images/{train,eval,test}/...
  pose_images/{train,eval,test}/...
  phase_a/{train,eval,test}.jsonl
  phase_a/meta_{train,eval,test}.jsonl
  dataset_info.json
  split_manifest.json
  three_image_validation.json
```

整个构建任务完成的判据不是“目录已出现”，而是：

- `three_image_800k_build_summary.json` 中 `status` 为 `passed`；
- 两个 `three_image_validation.json` 均为 `passed`；
- 两个 TAR 均存在且大小非零；
- train JSONL 各有 800,000 条记录。

数据尚未提供最终 OBS 地址。创建正式 DI 脚本时保留：

```bash
DATASET_OBS_PATH=${DATASET_OBS_PATH:?DATASET_OBS_PATH is required}
```

拿到两个 TAR 的 OBS 地址后，再分别写入两个正式 recipe 的默认值；不要猜测地址。

## 3. 样本契约

```json
{
  "id": "sample_id",
  "image": "images/train/group/sample.png",
  "images": [
    "images/train/group/sample.png",
    "raw_lane_images/train/group/sample.png",
    "pose_images/train/group/sample.png"
  ],
  "raw_lane_image": "raw_lane_images/train/group/sample.png",
  "pose_image": "pose_images/train/group/sample.png",
  "meta": {
    "coord_mode": "norm1000",
    "coord_range": 1000,
    "raw_lane_overlay": false,
    "raw_lane_separate_image": true,
    "input_image_roles": [
      "bev_road_structure",
      "pv_camera_raw_lane",
      "historical_vehicle_trajectory"
    ]
  },
  "conversations": [
    {"from": "human", "value": "<image>\n<image>\n<image>\n..."},
    {"from": "gpt", "value": "{\"lines\":[...]}"}
  ]
}
```

训练 preflight 必须抽查 train/eval/test，并强制满足：

- `images` 长度等于 3；
- 三个路径依次以 `images/`、`raw_lane_images/`、`pose_images/` 开头；
- user prompt 恰好包含 3 个 `<image>`；
- `record.image == record.images[0]`；
- `record.raw_lane_image == record.images[1]`；
- `record.pose_image == record.images[2]`；
- 三个文件都存在且能被 PIL 解码；
- `meta.raw_lane_overlay == false`；
- assistant target 是合法 `{"lines":[...]}` JSON。

## 4. Prompt

三图 prompt 的开头为：

```text
<image>
<image>
<image>
The first image is the clean BEV road-structure image.
The second image is a lane image predicted by a PV camera model: white lines are predicted lanes on a black background. Do not copy it blindly when it conflicts with the visible BEV evidence.
The third image is a historical vehicle-trajectory image: white lines are historical vehicle trajectories on a black background.
```

随后仍使用 Dataset V2 的道路中心线、路口、norm1000、类型字段和纯 JSON
输出要求。训练脚本不应在运行时覆盖 JSONL 内的 user prompt。

语义白名单：

```text
lane_type:
  common
  right_turn
  waiting_area
  bus_lane
  main_auxiliary_connector
  other

intersection_type:
  common
  t_intersection
  small_untyped
  t_lane_change_area
  other
```

## 5. 数据分布

| 项目 | 数值 |
|---|---:|
| train 样本 | 800,000 |
| empty | 5% |
| easy | 25% |
| medium | 33% |
| hard | 27% |
| very_hard | 10% |
| 固定版含路口样本 | 28% |
| train stride | 128 |
| eval/test stride | 256 |
| 坐标 | norm1000 |

固定评估清单为：

```text
D:\data\fixed_splits\rc_fixed_large_maps_v1.json
```

固定版 800k 在唯一样本、固定难度配额和固定评估大图约束下，含路口样本
最多约为 28.188875%，因此本发布选择可精确满足的 28%，不是 30%。

## 6. 模型基线

除数据输入变为三图外，首先保持 Jiangjihua/当前 MLLM 主线不变：

| 项目 | 基线 |
|---|---|
| 训练入口 | `python -m mllm.train.train_qwen` |
| 视觉塔 | 原始 DINOv2-Large |
| DINO 输入 | 518 |
| 视觉层 | `-2`，patch token |
| Projector | `mlp2x_gelu` |
| LLM | CapRL-Qwen3VL-4B 派生文本 LLM |
| DeepStack | 关闭 |
| 参数训练 | Stage A 全参数 SFT |
| 精度 | BF16 |
| 分布式 | DeepSpeed ZeRO-3 |
| Gradient checkpointing | 开启 |
| LLM LR | `2e-5` |
| Projector LR | `2e-5` |
| Vision LR | `2e-5` |
| Epoch | 8，先完成 smoke 和短程 pilot |
| 目标全局 batch | 128 |

不要为这次数据实验同时引入 DeepStack、视觉层融合、坐标 token、视觉 token
压缩、私有 DINO checkpoint 或新 projector，否则无法判断三图输入本身的收益。

## 7. 显存与序列长度

DINOv2-Large 在 518 输入下生成约 `37 x 37 = 1369` 个 patch token。
三张图会产生约 `3 x 1369 = 4107` 个视觉 token，再加 system/user 文本和
assistant JSON target。`MODEL_MAX_LENGTH=4096` 限制的是 tokenizer 侧文本，
多模态层仍会把三个占位符展开成三段视觉序列。

因此不能直接沿用单图 recipe 的 `PER_DEVICE_TRAIN_BATCH_SIZE=4`。建议首轮：

```text
PER_DEVICE_TRAIN_BATCH_SIZE=1
TARGET_GLOBAL_BATCH_SIZE=128
BF16=true
GRADIENT_CHECKPOINTING=true
DEEPSPEED_CONFIG=scripts/deepspeed_zero3.json
```

梯度累积按实际 world size 自动计算：

```text
gradient_accumulation = ceil(128 / (world_size * per_device_batch))
```

例如 4 节点、每节点 8 卡时，world size 为 32，每卡 batch 1，对应累积 4。

local256 和 context512_roi256 都会在进入 DINOv2 前预处理为 518，因此三段
视觉 token 数基本相同；context 数据的原始图更大，但不会因此产生更多 DINO
patch token。两条 recipe 都应先从每卡 batch 1 开始。

## 8. 其他 Agent 的实现任务

基于最接近的正式 Dataset V2 脚本，新建两个自包含入口：

```text
scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_three_image_local256_800k_original_dinov2_caprl4b_nodeepstack_npu.sh
scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_800k_original_dinov2_caprl4b_nodeepstack_npu.sh
```

参考脚本：

```text
scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_original_dinov2_caprl4b_nodeepstack_npu.sh
scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_rawlane_context512_roi256_200k_stratified_original_dinov2_caprl4b_nodeepstack_npu.sh
```

实现要求：

1. 两条脚本只允许数据路径、run name 和数据 profile 不同，模型基线保持一致。
2. 数据下载、解压、根目录解析和 preflight 必须在每个节点执行一致。
3. 不写死节点数和卡数，继续读取 DI 提供的分布式环境变量。
4. 默认 `PER_DEVICE_TRAIN_BATCH_SIZE=1`，按 world size 推导梯度累积。
5. 先提供 `MAX_STEPS=5` 的 DI-like smoke，再提交正式 8 epoch 任务。
6. smoke 至少跨过一次 forward、backward、optimizer step 和日志打印。
7. 保存前做短程 checkpoint smoke，避免长训练到保存时才暴露 NPU 问题。
8. 日志必须打印 `DI_throughput: ... samples/s/npu`。
9. 数据集检查失败必须退出，不允许降级成只读第一张图。
10. 在 `scripts/npu/train/README.md` 记录两个新入口、默认参数和 OBS 地址。

## 9. 推荐执行顺序

1. 确认两个数据集及 TAR 的完成标志。
2. 将两个 TAR 分别上传 OBS，并记录最终地址与 SHA-256。
3. 在单节点 8 卡 Ascend 上各跑 5 step smoke。
4. 检查每个 batch 实际读取三图，视觉编码器收到的图数为 batch size 的 3 倍。
5. 检查 loss、grad norm、NPU 显存和吞吐量是否稳定。
6. 先运行一个短程 pilot 并完成 checkpoint 重载推理。
7. 再提交正式多节点 8 epoch 训练。
8. 两个实验使用相同评估脚本、同一固定 eval/test patch ID 和同一指标配置。

## 10. 不可变约束

- 不改变 JSON target schema。
- 不改变 norm1000 坐标定义。
- 不改变固定评估大图。
- 不把 Raw-Lane 叠回 BEV。
- 不删除 `Do not copy it blindly...` 提示。
- 不交换三图顺序。
- 不把两套 800k 数据混合训练。
- 不在没有三图 preflight 和 NPU smoke 的情况下直接提交长任务。
