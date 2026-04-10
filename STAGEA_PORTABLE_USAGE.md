# UniMapGen-StageA 可迁移说明

如果你要使用全量 `fixed16` 非离散 token `LoRA` 线的 `checkpoint-157182` 最小集，请看：

- `STAGEA_FIXED16_CHECKPOINT_157182_USAGE.md`

在开始训练或推理前，请先把下面这些资源放到仓库根目录下的固定相对位置：

- `ckpts/modelscope/Qwen/Qwen2___5-VL-3B-Instruct/`
- `dataset/paper16_patch_only_100img_system/`
- `outputs/llamafactory_qwen2_5vl_3b_paper16_patch_only_100img_lora/`

## Checkpoint 识别约定

- 训练输出目录固定写到：
  - `outputs/llamafactory_qwen2_5vl_3b_paper16_patch_only_100img_lora/`
- 推理默认也从这个目录读取 LoRA / checkpoint
- 如果这个目录本身就直接包含 LoRA 文件，例如：
  - `adapter_config.json`
  - `adapter_model.safetensors`
  那么推理会直接把这个目录当作 adapter 目录使用
- 否则，推理会在该目录下继续查找 `checkpoint-*` 子目录，并自动选择最新的一个
- 如果这两种情况都不存在，推理会直接报错退出，不会模糊尝试

## 入口脚本

- 训练入口：
  - `scripts/launch_stagea_patch_only_100img_train.sh`
- 单个 family 推理评估入口：
  - `scripts/launch_stagea_patch_only_1fam_eval.sh`

## 脚本行为

- 训练和推理入口都会在运行时从当前仓库根目录解析路径
- 训练入口会在 `.runtime/` 下生成一份运行时 `dataset_info.json`
  - 因此不会依赖复制前配置文件里可能残留的绝对路径
- 推理入口默认只会从
  - `outputs/llamafactory_qwen2_5vl_3b_paper16_patch_only_100img_lora/`
  这一个固定位置解析 adapter，除非你显式设置了 `ADAPTER`
- 如果基础模型、数据集或 checkpoint 路径缺失，训练和推理入口都会立刻报错退出

## 可覆盖环境变量

如果你确实需要覆盖默认路径，可以手动设置下面这些环境变量：

- `MODEL_DIR`
- `DATASET_ROOT`
- `ADAPTER`
- `OUTPUT_DIR`
- `DEVICE`
