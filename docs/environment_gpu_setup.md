# UniMapGen Environment Setup

## 1. 独立环境

已在项目目录下创建独立 `conda` 环境：

- `/mnt/data/project/jn/UniMapGen/.envs/unimapgen-gpu`

激活方式：

```bash
cd /mnt/data/project/jn/UniMapGen
bash scripts/activate_unimapgen_gpu_env.sh
```

或者手动：

```bash
source /home/lenovo/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/data/project/jn/UniMapGen/.envs/unimapgen-gpu
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
```

注意：

- 这一行 `LD_LIBRARY_PATH` 目前不是可选项
- 否则 `transformers -> sklearn -> pyarrow` 这条导入链会落到系统 `libstdc++.so.6`
- 然后报 `GLIBCXX_3.4.32 not found`

## 2. 当前环境状态

环境内已经有 CUDA 版 PyTorch：

- `torch==2.10.0+cu130`
- `torchvision==0.25.0+cu130`
- `transformers==5.2.0`

你的终端里已经确认：

```text
torch.cuda.is_available() == True
torch.cuda.device_count() == 1
```

但我当前这条 agent 会话里同一环境仍返回：

```text
torch.cuda.is_available() == False
torch.cuda.device_count() == 0
```

## 3. 关键判断

这说明环境本身已经没问题，差异来自“不同终端/会话的 GPU 可见性”。

我核对到的现象：

1. `/dev/nvidia0`、`/dev/nvidiactl` 等设备节点存在
2. `nvidia-smi` 报错：`Failed to initialize NVML: Unknown Error`
3. 在你的交互终端里 GPU 可用，但在 agent 会话里不可用

因此：

- 新建虚拟环境已经完成
- 你应该在自己的 GPU 终端里运行训练
- 我这边仍可继续负责代码、配置、日志分析和结果排查

## 4. 后续建议

推荐直接在你的终端里执行：

```bash
cd /mnt/data/project/jn/UniMapGen
bash scripts/activate_unimapgen_gpu_env.sh
python -m unimapgen.train_qwen_map --config configs/qwen_dinov2_map_serialization_smoke.yaml
python -m unimapgen.eval_qwen_map --config configs/qwen_dinov2_map_serialization_smoke.yaml --checkpoint outputs/qwen_dinov2_map_serialization_smoke/latest.pt --split val --max_samples 1
python -m unimapgen.predict_qwen_map --config configs/qwen_dinov2_map_serialization_smoke.yaml --checkpoint outputs/qwen_dinov2_map_serialization_smoke/latest.pt --split val --max_samples 1 --output outputs/qwen_dinov2_map_serialization_smoke/predictions.json
```

如果你跑出日志或报错，把输出贴给我，我继续往下修。
