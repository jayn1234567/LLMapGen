 # Stage A Portable Usage

## 1. 放模型

把基础模型放到：

- `ckpts/modelscope/Qwen2___5-VL-3B-Instruct/`

## 2. 放数据

把数据集放到：

- `dataset/`

至少要有：

- `dataset/train.jsonl`
- 可选的 `dataset/val.jsonl`
- `dataset/images/`

## 3. 训练

全参训练：

```bash
bash scripts/launch_stagea_discrete_train.sh
```

LoRA 训练：

```bash
bash scripts/launch_stagea_discrete_train_lora.sh
```

## 4. 评估 / 推理 / 可视化

```bash
bash scripts/launch_stagea_discrete_eval.sh
```

## 5. 默认训练配置

全参训练默认：

- `8 GPU`
- `bs/device = 4`
- `grad_accum = 1`
- global batch `32`
- `lr = 1e-4`
- `epoch = 6`
- `shared_numbers`
- `coord_num_bins = 896`

LoRA 训练默认：

- `4 GPU`
- `bs/device = 4`
- `grad_accum = 2`
- global batch `32`
- `lr = 1e-4`
- `epoch = 6`
- `lora_rank = 16`
- `lora_alpha = 32`
- `lora_dropout = 0.05`
- `shared_numbers`
- `coord_num_bins = 896`

## 6. 常改路径

可以直接改这些环境变量：

- `MODEL_DIR`
- `DATASET_ROOT`
- `TRAIN_JSON`
- `EVAL_JSON`
- `OUTPUT_DIR`
- `RUN_DIR`
- `CHECKPOINT_OR_MODEL`
- `PROCESSOR_PATH`


