# NPU / DI 训练适配说明

整理当前 LLMapGen DINOv2 + Qwen3 路线在 Ascend NPU 和 DI 训练平台上的适配状态，把训练流程整理成“代码仓 + OBS 数据/模型 + 一条 shell 命令”即可启动的 DI 作业。

## 目标

当前目标是把这条路线迁移到 DI 平台：

```text
分割训练后的 DINOv2
  + Qwen3-8B 对齐层/桥接层
  + 干净 Qwen3-8B
  + 私有数据集
  + LoRA SFT
  + DINOv2 后几层可训练
  + NPU/HCCL 多卡训练
```

训练平台侧期望流程：

```text
1. 代码上传到 CodeHub/GitHub
2. 数据集和模型资产放到 OBS
3. DI 平台配置代码仓地址、数据 OBS 地址、输出 OBS 地址
4. DI 平台执行仓库里的训练脚本
5. 脚本下载数据/模型到 /cache，启动 NPU 训练，最后把结果写回平台输出目录
```

## 当前已经完成的适配

### 1. NPU 训练入口

维护入口：

```text
scripts/train_dinov2_centerline.py
scripts/npu/train/train_dinov2_centerline_qwen_lora_npu.sh
scripts/npu/train/smoke_dinov2_centerline_qwen_random_align_npu.sh
scripts/npu/test/test_dinov2_centerline_qwen_lora_npu.sh
```

训练脚本支持通过环境变量配置：

```bash
TRAINROOT=/path/to/prepared_trainroot
MODEL_NAME_OR_PATH=/path/to/Qwen3-8B
DINOV2_MODEL_NAME_OR_PATH=/path/to/dinov2-large
OUTPUT_DIR=/path/to/output
VISUAL_ENCODER_CHECKPOINT_PATH=/path/to/visual_encoder_checkpoint.pt
BRIDGE_MODULES_STATE_PATH=/path/to/bridge_modules_state.pt
MAP_TASK=lane_intersection
```

核心 NPU 参数：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=8
USE_TORCHRUN=true
KEEP_DISTRIBUTED_ENV=true
BF16=true
GRADIENT_CHECKPOINTING=true
```

训练入口会设置：

```bash
--device-backend npu
--ddp-backend hccl
```

### 2. 单卡/多卡 NPU 分布式适配

已经处理过 DI/NPU 上常见的单进程分布式变量污染问题。

单进程直跑时：

```bash
LLMAPGEN_FORCE_SINGLE_PROCESS_NPU=true
```

脚本会清理：

```bash
RANK
WORLD_SIZE
LOCAL_RANK
LOCAL_WORLD_SIZE
MASTER_ADDR
MASTER_PORT
```

多卡时：

```bash
USE_TORCHRUN=true
KEEP_DISTRIBUTED_ENV=true
NPROC_PER_NODE=8
```

脚本使用：

```bash
python -m torch.distributed.run --nproc_per_node "${NPROC_PER_NODE}"
```

并走 HCCL backend。

相关代码：

```text
unimapgen/runtime/device.py
unimapgen/rc_llm_runtime.py
scripts/npu/train/train_dinov2_centerline_qwen_lora_npu.sh
```

### 3. NPU 环境脚本

已有环境创建/复制脚本：

```text
scripts/npu/setup/create_llmapgen_npu_env.sh
scripts/npu/setup/create_llmapgen_npu_conda_env.sh
scripts/npu/setup/clone_llmapgen_npu_conda_env.sh
```

用途：

- source Ascend toolkit 环境。
- 创建 conda/venv 环境。
- 安装适配版本的 HF 依赖。



```bash
source /home/ma-user/.conda/envs/llmapgen-npu/activate_llmapgen_npu.sh
```

### 4. 数据转换和验证

已有工具：

```text
scripts/tools/inspect_di_dataset.py
scripts/tools/prepare_di_qa_trainroot.py
scripts/tools/validate_di_trainroot.py
```

支持的数据结构：

```text
dataset/
  images/
    train/
    eval/
    test/
  phase_a/
    train.jsonl
    eval.jsonl
    test.jsonl
    meta_train.jsonl
    meta_eval.jsonl
    meta_test.jsonl
  phase_b/
    ...
  dataset_info.json
  split_manifest.json
```

已支持任务：

```text
lane
lane_intersection
```

`lane_intersection` 标签格式沿用数据源结构，不另开 `intersections` 字段：

```json
{
  "lines": [
    {
      "category": "centerline",
      "start_type": "cut",
      "end_type": "inside",
      "points": [[0, 499], [365, 473]]
    },
    {
      "category": "intersection",
      "is_cut": true,
      "points": [[368, 510], [364, 457], [415, 368], [368, 510]]
    }
  ]
}
```

对于 `coord_mode=norm1000` 的数据，转换脚本会统一转到训练使用的 `0..512` 坐标。

### 5. 视觉资产包

已经打包好的正式资产：

```text
dinov2_centerline_assets_qwen3_8b/
  asset_manifest.json
  train_env_template.sh
  visual_encoder_checkpoint.pt
  bridge_modules_state.pt
```

对应本地 tar：

```text
C:\Users\23931\Desktop\dinov2_centerline_assets_qwen3_8b.tar
```

SHA256：

```text
6b0854fe69839ef07ab9e36d6ec0f37b2b2ec48f889ddf1c3bf1ba8c8a237c6c
```

已经切分成 512MB 分卷：

```text
C:\Users\23931\Desktop\dinov2_centerline_assets_qwen3_8b_parts/
  dinov2_centerline_assets_qwen3_8b.tar.part-001
  ...
  dinov2_centerline_assets_qwen3_8b.tar.part-008
  ORIGINAL_SHA256.txt
  SHA256SUMS.txt
  merge_parts_linux.sh
  merge_parts_windows.ps1
```

Linux 合并：

```bash
cd dinov2_centerline_assets_qwen3_8b_parts
bash merge_parts_linux.sh
```

## 当前已验证情况

已经验证过：

- NPU 上随机对齐层 smoke 跑通。
- NPU 多卡 torchrun/HCCL 跑通。
- NPU 上 DINOv2 forward + Qwen3-8B LoRA + dataset loader 跑通。
- NPU 上训练 loss 能正常下降。
- `lane_intersection` 转换逻辑已通过函数级测试并提交。

还需要正式验证：

- 512 `lane_intersection` 数据的完整 trainroot 转换和 validate。
- 正式资产包 + 512 数据 + 多卡 NPU 的长时间训练。
- DI 平台上从 OBS 自动下载数据/模型并上传输出的自包含脚本。

## 当前手动训练流程

### 1. 准备数据

512 patch、中心线+路口数据：

```bash
cd /cache/jn/LLMapGen

python scripts/tools/prepare_di_qa_trainroot.py \
  --input-root /cache/jjh/data/data_lane_intersection_norm_sample_512_33w \
  --phase phase_a \
  --image-root images \
  --task lane_intersection \
  --output-root /cache/jn/prepared_lane_intersection_trainroot
```

验证：

```bash
python scripts/tools/validate_di_trainroot.py \
  --trainroot /cache/jn/prepared_lane_intersection_trainroot \
  --coord-max 512
```

### 2. 随机对齐层 smoke

先不加载正式 DINOv2 分割权重和对齐层，只验证 NPU 训练链路：

```bash
cd /cache/jn/LLMapGen
source /home/ma-user/.conda/envs/llmapgen-npu/activate_llmapgen_npu.sh

export TRAINROOT=/cache/jn/prepared_lane_intersection_trainroot
export OUTPUT_DIR=/cache/jn/outputs/random_align_lane_intersection_smoke
export MODEL_NAME_OR_PATH=/path/to/Qwen3-8B
export DINOV2_MODEL_NAME_OR_PATH=/path/to/dinov2-large
export MAP_TASK=lane_intersection

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export USE_TORCHRUN=true
export KEEP_DISTRIBUTED_ENV=true
export MAX_STEPS=10
export MAX_SAMPLES=64
export MAX_EVAL_SAMPLES=8

bash scripts/npu/train/smoke_dinov2_centerline_qwen_random_align_npu.sh
```

### 3. 正式资产训练

使用分割 DINOv2 + 对齐层资产：

```bash
cd /cache/jn/LLMapGen
source /home/ma-user/.conda/envs/llmapgen-npu/activate_llmapgen_npu.sh
source /path/to/dinov2_centerline_assets_qwen3_8b/train_env_template.sh

export TRAINROOT=/cache/jn/prepared_lane_intersection_trainroot
export OUTPUT_DIR=/cache/jn/outputs/dinov2_qwen3_8b_lane_intersection_lora_npu
export MODEL_NAME_OR_PATH=/path/to/Qwen3-8B
export DINOV2_MODEL_NAME_OR_PATH=/path/to/dinov2-large
export MAP_TASK=lane_intersection

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export USE_TORCHRUN=true
export KEEP_DISTRIBUTED_ENV=true

export NUM_TRAIN_EPOCHS=6
export MAX_STEPS=-1
export MAX_SAMPLES=0
export MAX_EVAL_SAMPLES=0
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=4
export LEARNING_RATE=2e-5
export SAVE_STEPS=1000
export SAVE_TOTAL_LIMIT=10
export LOGGING_STEPS=10
export DATALOADER_NUM_WORKERS=2
export BF16=true
export GRADIENT_CHECKPOINTING=true

export FREEZE_VISION_ENCODER=true
export VISION_TRAIN_LAST_N_LAYERS=2

bash scripts/npu/train/train_dinov2_centerline_qwen_lora_npu.sh
```

## DI 脚本做法

项目里的 `scripts/npu/train` 是更完整的 DI 生产化写法。典型脚本如：

```text
train_sft_stage_a_lane_dinov2_qwen3vl_nodeepstack_npu.sh
train_sft_stage_a_lane_intersection_dinov2_qwen3vl_nodeepstack_npu.sh
train_sft_stage_b_lane_dinov2_qwen3vl_nodeepstack_npu.sh
```

1. 每个脚本是一个固定 recipe。

```bash
DATASET_PHASE=phase_a
MAP_TASK=lane_intersection
VISION_BACKBONE=dinov2
VISION_TOWER_NAME=facebook_dinov2-large
MM_VISION_TOWER_TYPE=dinov2
INPUT_IMAGE_SIZE=518
```

2. 使用 DI 平台注入的 `OUTPUT_URL`。

```bash
CLUSTER_SAVE=${OUTPUT_URL}
OSB_SHARE_PATH="${CLUSTER_SAVE}"
RUN_ID=$(date -u +%Y%m%d_%H%M%S)
LOCAL_MODEL_SAVE_PATH=/cache/local_model_save_path/${RUN_ID}
CLOUD_OUTPUT_PATH=${OSB_SHARE_PATH%/}/${RUN_ID}
```

3. 用 moxing 从 OBS 下载模型和数据。

```bash
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/${VISION_TOWER_NAME}', '${VISION_TOWER}')"
python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ZIP_PATH}')"
unzip -q "${DATASET_ZIP_PATH}" -d "${DATASET_EXTRACT_ROOT}"
python -c "import moxing as mox; mox.file.copy_parallel('${QWEN_MODEL_OBS_PATH}', '${QWEN_PATH}')"
```

4. 在脚本里 source Ascend 环境并设置 HCCL。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export HCCL_SOCKET_IFNAME=eth0
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_WHITELIST_DISABLE=1
export WITHOUT_JIT_COMPILE=1
export COMBINED_ENABLE=1
```

5. 自动识别 DI 多机变量。

```bash
if [[ -z "${MA_VJ_NAME:-}" ]]; then
  NNODES=${NNODES:-1}
  NODE_RANK=${NODE_RANK:-0}
  NPROC_PER_NODE=${NPROC_PER_NODE:-8}
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
else
  NNODES=${NNODES:-$MA_NUM_HOSTS}
  NODE_RANK=${NODE_RANK:-$VC_TASK_INDEX}
  NPROC_PER_NODE=${NPROC_PER_NODE:-$MA_NUM_GPUS}
  MASTER_ADDR=${MASTER_ADDR:-${VC_WORKER_HOSTS%%,*}}
fi
```

6. 自动计算梯度累积。

```bash
TOTAL_DEVICES=$(( NNODES * NPROC_PER_NODE ))
MICRO_BATCH=$(( TOTAL_DEVICES * PER_DEVICE_TRAIN_BATCH_SIZE ))
GRADIENT_ACCUMULATION_STEPS=$(( (TARGET_GLOBAL_BATCH_SIZE + MICRO_BATCH - 1) / MICRO_BATCH ))
```

7. 用 `torchrun` 启动训练。

```bash
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m mllm.train.train_qwen \
  ...
```

8. rank0 把本地输出移动到云输出。

```bash
if [[ "${NODE_RANK}" == "0" ]]; then
  mv "${OUTPUT_PATH}" "${CLOUD_OUTPUT_PATH}"
fi
```

## DI 脚本

已经新增：

```text
scripts/npu/train/train_di_dinov2_centerline_qwen_lora_npu.sh
```

参考 jiangjihua，但服务我们自己的模型结构。

### 需要支持的输入变量

DI 平台配置：

```bash
OUTPUT_URL=obs://bucket/path/to/output
DATASET_OBS_PATH=obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/data/prepared_lane_intersection_trainroot.tar
QWEN_MODEL_OBS_PATH=obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/checkpoint/Qwen3-8B
DINOV2_MODEL_OBS_PATH=obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints/facebook_dinov2-large
ASSET_OBS_PATH=obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/model/dinov2_centerline_assets_qwen3_8b
```

可选训练参数：

```bash
DATASET_KIND=auto
DATASET_DIR_NAME=prepared_lane_intersection_trainroot
DATASET_PHASE=phase_a
MAP_TASK=lane_intersection
NUM_TRAIN_EPOCHS=6
TARGET_GLOBAL_BATCH_SIZE=32
PER_DEVICE_TRAIN_BATCH_SIZE=1
LEARNING_RATE=1e-4
VISION_TRAIN_LAST_N_LAYERS=4
```

当前 `DATASET_OBS_PATH` 指向已经转换好的 trainroot tar，脚本解压后会直接
查找 `train.jsonl` 并用于训练，不会再调用 `prepare_di_qa_trainroot.py`。如果后续
传入的是原始私有数据集 zip/tar，再设置 `DATASET_KIND=raw`，脚本会走转换流程。

### 脚本内部流程

```text
1. 定义 RUN_ID、/cache 路径、输出路径
2. source Ascend 环境
3. 可选安装/激活 conda 环境
4. 用 moxing 下载数据 tar/zip/目录
5. 解压或复制到 /cache/dataset_extract_${RUN_ID}
6. 如果发现 train.jsonl，直接作为 prepared trainroot 使用
7. 否则按 raw dataset 调用 prepare_di_qa_trainroot.py 转成 trainroot
8. 用 moxing 下载 Qwen3-8B、DINOv2-large、资产包
9. 调用 validate_di_trainroot.py 做快速校验
10. 自动识别 DI 单机/多机变量
11. 按 TARGET_GLOBAL_BATCH_SIZE 计算梯度累积
12. torchrun 启动 train_dinov2_centerline.py
13. rank0 把输出移动或上传到 OUTPUT_URL/RUN_ID
```

### 伪代码骨架

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${OUTPUT_URL:?OUTPUT_URL is required}"
: "${DATASET_OBS_PATH:?DATASET_OBS_PATH is required}"
: "${QWEN_MODEL_OBS_PATH:?QWEN_MODEL_OBS_PATH is required}"
: "${DINOV2_MODEL_OBS_PATH:?DINOV2_MODEL_OBS_PATH is required}"
: "${ASSET_OBS_PATH:?ASSET_OBS_PATH is required}"

RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}
OBS_CACHE=${OBS_CACHE:-/cache}
DATASET_EXTRACT_ROOT=${OBS_CACHE}/dataset_extract_${RUN_ID}
PREPARED_TRAINROOT=${OBS_CACHE}/prepared_trainroot_${RUN_ID}
LOCAL_OUTPUT_DIR=${OBS_CACHE}/llmapgen_output_${RUN_ID}
CLOUD_OUTPUT_DIR=${OUTPUT_URL%/}/${RUN_ID}

source /usr/local/Ascend/ascend-toolkit/set_env.sh || true
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_WHITELIST_DISABLE=1
export TOKENIZERS_PARALLELISM=false

python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${OBS_CACHE}/dataset_${RUN_ID}.zip')"
unzip -q "${OBS_CACHE}/dataset_${RUN_ID}.zip" -d "${DATASET_EXTRACT_ROOT}"

python -c "import moxing as mox; mox.file.copy_parallel('${QWEN_MODEL_OBS_PATH}', '${OBS_CACHE}/checkpoints/Qwen3-8B')"
python -c "import moxing as mox; mox.file.copy_parallel('${DINOV2_MODEL_OBS_PATH}', '${OBS_CACHE}/checkpoints/dinov2-large')"
python -c "import moxing as mox; mox.file.copy_parallel('${ASSET_OBS_PATH}', '${OBS_CACHE}/assets/dinov2_centerline_assets_qwen3_8b')"

source "${OBS_CACHE}/assets/dinov2_centerline_assets_qwen3_8b/train_env_template.sh"

python scripts/tools/prepare_di_qa_trainroot.py \
  --input-root "${DATASET_EXTRACT_ROOT}" \
  --dataset-dir-name "${DATASET_DIR_NAME}" \
  --phase "${DATASET_PHASE:-phase_a}" \
  --image-root images \
  --task "${MAP_TASK:-lane_intersection}" \
  --output-root "${PREPARED_TRAINROOT}"

python scripts/tools/validate_di_trainroot.py \
  --trainroot "${PREPARED_TRAINROOT}" \
  --coord-max 512

export TRAINROOT="${PREPARED_TRAINROOT}"
export OUTPUT_DIR="${LOCAL_OUTPUT_DIR}"
export MODEL_NAME_OR_PATH="${OBS_CACHE}/checkpoints/Qwen3-8B"
export DINOV2_MODEL_NAME_OR_PATH="${OBS_CACHE}/checkpoints/dinov2-large"
export MAP_TASK="${MAP_TASK:-lane_intersection}"
export USE_TORCHRUN=true
export KEEP_DISTRIBUTED_ENV=true

bash scripts/npu/train/train_dinov2_centerline_qwen_lora_npu.sh

python -c "import moxing as mox; mox.file.copy_parallel('${LOCAL_OUTPUT_DIR}', '${CLOUD_OUTPUT_DIR}')"
```

## DI 平台启动方式



```bash
cd /cache/jn/LLMapGen
source /home/ma-user/.conda/envs/llmapgen-npu/activate_llmapgen_npu.sh
source /cache/jn/dinov2_centerline_assets_qwen3_8b/train_env_template.sh

export TRAINROOT=/cache/jn/prepared_lane_intersection_trainroot
export OUTPUT_DIR=/cache/jn/outputs/di_lane_intersection_run
export MODEL_NAME_OR_PATH=/cache/models/Qwen3-8B
export DINOV2_MODEL_NAME_OR_PATH=/cache/models/dinov2-large
export MAP_TASK=lane_intersection

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export USE_TORCHRUN=true
export KEEP_DISTRIBUTED_ENV=true

bash scripts/npu/train/train_dinov2_centerline_qwen_lora_npu.sh
```

等 `train_di_dinov2_centerline_qwen_lora_npu.sh` 补完后，DI 平台最终命令应简化为：

```bash
bash scripts/npu/train/train_di_dinov2_centerline_qwen_lora_npu.sh
```
