# Two-Stage Finetune Handoff Minimum Checklist

这份文档面向“第一次拿到代码的人”。

目标只有两个：

1. 明确**最小必需文件清单**
2. 明确**实际启动顺序**

只关注当前两阶段微调主线：

- `Stage A`: patch-only
- `Stage B`: fake-state mixture state-update
- rollout inference: `handoff gating + patch-only agreement`

## 1. 必要前提

对方需要先准备好：

- 代码仓库：拉取分支 `codex/two-stage-finetune-only`
- 原始数据集
- `Qwen2.5-VL-3B-Instruct` 本地模型
- conda 环境
- 可用 GPU

如果这 5 项不齐，后面的脚本不能正常跑。

## 2. 最小必需文件清单

下面这些是当前两阶段链路的最小必需代码。

### 2.1 配置文件

Stage A:

- `/mnt/data/project/jn/UniMapGen/configs/llamafactory_paper16_patch_only_100img_system/dataset_info.json`
- `/mnt/data/project/jn/UniMapGen/configs/llamafactory_paper16_patch_only_100img_system/qwen2_5vl_3b_lora_sft.yaml`

Stage B:

- `/mnt/data/project/jn/UniMapGen/configs/llamafactory_paper16_stageb_from_patchonly_fake_mixture/dataset_info.json`
- `/mnt/data/project/jn/UniMapGen/configs/llamafactory_paper16_stageb_from_patchonly_fake_mixture/qwen2_5vl_3b_lora_sft.yaml`

### 2.2 数据构建脚本

- `/mnt/data/project/jn/UniMapGen/scripts/build_opensatmap_paper16_family_manifest.py`
- `/mnt/data/project/jn/UniMapGen/scripts/export_llamafactory_patch_only_from_raw_family_manifest.py`
- `/mnt/data/project/jn/UniMapGen/scripts/export_llamafactory_state_sft_from_raw_family_manifest.py`

### 2.3 推理/评估脚本

- `/mnt/data/project/jn/UniMapGen/scripts/run_qwen2_5vl_lora_small_eval.py`
- `/mnt/data/project/jn/UniMapGen/scripts/rollout_predict_qwen2_5vl_from_raw_family_manifest.py`

### 2.4 自动串联脚本

- `/mnt/data/project/jn/UniMapGen/scripts/run_stagea_resume_stageb_fake_rollout_pipeline.sh`

## 3. 目录放置要求

### 3.1 代码目录

项目代码根目录：

```bash
/mnt/data/project/jn/UniMapGen
```

### 3.2 原始数据目录

原始数据目录：

```bash
/mnt/data/data1/OpenSateMap
```

关键标注文件：

```bash
/mnt/data/data1/OpenSateMap/annotrainval20.json
```

### 3.3 模型目录

Qwen2.5-VL 本地模型目录：

```bash
/mnt/data/project/jn/UniMapGen/ckpts/modelscope/Qwen/Qwen2___5-VL-3B-Instruct
```

### 3.4 输出目录

所有数据集导出、训练结果、日志都写到：

```bash
/mnt/data/project/jn/UniMapGen/outputs
```

## 4. 环境要求

建议有两个 conda 环境：

- `llamafactory-cu128`
- `unimapgen-gpu`

用途：

- `llamafactory-cu128`
  - Stage A 训练
  - Stage B 训练
  - rollout inference
- `unimapgen-gpu`
  - family manifest 构建
  - patch-only / state-update 数据集导出

如果只想跑当前主链，这两个环境都需要。

## 5. 启动顺序清单

按下面顺序执行最稳。

### 第一步：确认路径

确认这三个路径都存在：

- `/mnt/data/project/jn/UniMapGen`
- `/mnt/data/data1/OpenSateMap/annotrainval20.json`
- `/mnt/data/project/jn/UniMapGen/ckpts/modelscope/Qwen/Qwen2___5-VL-3B-Instruct`

### 第二步：构建 family manifest

先生成 `4096 -> paper16 family` 的 manifest。

关键输出：

- `/mnt/data/project/jn/UniMapGen/outputs/paper16_family_manifest_100img.jsonl`

### 第三步：导出 Stage A 数据集

导出 patch-only 数据集。

关键输出：

- `/mnt/data/project/jn/UniMapGen/outputs/paper16_patch_only_100img_system`

### 第四步：训练 Stage A

使用 Stage A 配置训练 patch-only 模型。

关键输出目录：

- `/mnt/data/project/jn/UniMapGen/outputs/llamafactory_qwen2_5vl_3b_paper16_patch_only_100img_lora`

这一步完成后，要确认：

- 根目录下有可用 adapter
或
- 至少有可用 checkpoint 目录

### 第五步：导出 Stage B 数据集

导出 fake-state mixture 数据集。

关键输出：

- `/mnt/data/project/jn/UniMapGen/outputs/paper16_sft_100img_system_paper_serialized_neighborfix_fake_mixture`

### 第六步：训练 Stage B

在 Stage A 基础上继续训练 Stage B。

关键输出目录：

- `/mnt/data/project/jn/UniMapGen/outputs/llamafactory_qwen2_5vl_3b_paper16_stageb_from_patchonly_fake_mixture_lora`

### 第七步：rollout 推理与可视化

使用 Stage B adapter 做 rollout inference。

关键输出目录：

- `/mnt/data/project/jn/UniMapGen/outputs/rollout_eval_stageb_fake_mixture_gated_16fam`

这里会包含：

- `summary.json`
- `predictions/`
- `metrics/`
- `visualizations/`

## 6. 最简手工执行顺序

如果不使用自动 pipeline，最简顺序就是：

1. 准备环境
2. 放好原始数据和 Qwen2.5-VL 模型
3. 构建 `paper16 family manifest`
4. 导出 `Stage A patch-only` 数据集
5. 训练 `Stage A`
6. 导出 `Stage B fake-state mixture` 数据集
7. 训练 `Stage B`
8. 跑 rollout inference 和可视化

## 7. 自动执行入口

如果要一键串起来，当前入口脚本是：

- `/mnt/data/project/jn/UniMapGen/scripts/run_stagea_resume_stageb_fake_rollout_pipeline.sh`

它负责：

1. 导出 fake-state mixture 数据集
2. 续训或接着使用 Stage A
3. 启动 Stage B
4. 跑 gated rollout inference

注意：

- 自动脚本依赖前面的路径都已经放对
- 如果数据集图片损坏，Stage B 会在加载数据时报错

## 8. 交接时必须提醒对方的事

交接时必须明确说明：

1. 仓库里有一些历史代码，但当前两阶段链路只依赖第 2 节列出的文件
2. `outputs/` 和原始数据集不在代码仓库里，需要单独准备
3. `Qwen2.5-VL-3B-Instruct` 必须本地可读，不能只写 HuggingFace 名字
4. `Stage B` 依赖一个真实可用的 `Stage A` adapter/checkpoint
5. rollout inference 不是单 patch 测试，而是顺序 `p00 -> p15` 的链式推理

## 9. 交接结论

如果对方：

- 拉取 `codex/two-stage-finetune-only`
- 按要求放好数据目录
- 放好 Qwen2.5-VL 本地模型
- 装好 conda 环境

那么他需要关注的核心内容只有：

- 第 2 节这些最小必需文件
- 第 5 节这个启动顺序

除此之外，仓库里其他历史代码都不应该成为他启动当前两阶段微调主线的阻碍。
