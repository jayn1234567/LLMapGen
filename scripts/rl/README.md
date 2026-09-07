# 强化学习辅助脚本

本目录保存 hard-pool 构建和 vLLM/LoRA 导出工具。正式 GRPO 训练入口位于 `scripts/npu/train/train_grpo_*.sh`。

| 脚本 | 作用 | 关键参数 |
|---|---|---|
| `build_hard_pool.py` | 根据推理 summary 的 F1、截断和格式情况生成困难样本池 | `--summary`, `--output-dir`, `--source-jsonl`, `--map-task`, `--low-f1-threshold`, `--medium-f1-threshold`, `--max-per-bucket`, `--seed` |
| `export_merged_lora_checkpoint.py` | 将 LoRA adapter 与基础模型合并为可独立加载的 checkpoint | `--adapter-checkpoint`, `--model-base`, `--output-dir`, `--vision-tower`, `--bf16` |
| `export_text_decoder_for_vllm.py` | 从多模态 checkpoint 导出 vLLM rollout 使用的文本 decoder | `--checkpoint`, `--output-dir`, `--overwrite` |

示例：

```bash
python scripts/rl/build_hard_pool.py \
  --summary /path/to/summary.json \
  --source-jsonl /path/to/train.jsonl \
  --output-dir /path/to/hard_pool \
  --map-task lane_intersection \
  --seed 42
```

hard pool 只用于后续 RL/再训练采样；不要用它替代固定评估集。

## ms-swift UniMapGen GRPO

| 文件 | 作用 |
|---|---|
| `swift_unimapgen_plugin.py` | 注册 DINOv2 + projector + CapRL 文本 LLM、Context512 模板和地图 reward |
| `../tools/prepare_swift_grpo_dataset.py` | 将带 GT 的训练集推理 JSONL 转为 Swift GRPO 数据 |
| `../tools/validate_swift_grpo_dataset.py` | 在启动 Swift 前检查图片、GT、坐标合同和唯一 ID |
| `../npu/train/train_swift_grpo_stage_a_context512_roi256_550k_npu.sh` | DI/NPU 一条命令正式入口，自动安装固定版 ms-swift，默认 `use_vllm=false`、4 个候选、1 epoch |

最小入口（默认直接使用 Context512/ROI256 550k 原始训练集）：

```bash
bash scripts/npu/train/train_swift_grpo_stage_a_context512_roi256_550k_npu.sh
```

上面的入口默认执行完整 550K 数据集的 1 epoch 正式训练。首次只做 smoke
时才临时增加 `MAX_STEPS=20`；正式入口不需要额外参数。

入口默认安装 `ms-swift==4.0.0`。安装前后会比较 `torch`、`torch_npu`
和 `transformers` 版本；如果 pip 试图改变这三个 DI 运行时版本，入口会直接失败。
设置 `INSTALL_MS_SWIFT=False` 可以关闭自动安装，但此时环境必须已经有完全匹配的
`ms-swift==4.0.0`。当前正式默认值为 `MAX_STEPS=-1`、`NUM_TRAIN_EPOCHS=1`。

入口默认下载：

```text
obs://yw-ads-training-2-gy1/data/external/personal/h58801830/jn/data/context512_roi256/context512_roi256_550k.tar
```

如需覆盖默认数据、模型或视觉塔，可设置 `DATASET_OBS_PATH`、`MODEL_OBS_PATH`
和 `VISION_TOWER_OBS_PATH`。也可以直接指定已经解压的
`DATASET_ROOT`，或指定原始 `TRAIN_JSONL`；不再要求先生成训练集推理结果。
后续做 hard-pool 时，才使用 `INFERENCE_JSONL` 或 `INFERENCE_OBS_PATH`。

完整的数据合同、奖励公式、SFT checkpoint 续训和 DI 限制见
`docs/SWIFT_GRPO_UNIMAPGEN_HANDOFF.md`。
