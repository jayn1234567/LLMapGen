# UniMapGen Stage A Fixed16 `checkpoint-157182` 可迁移说明

这份最小集对应的是超算上这条非离散 token、`LoRA`、全量 `fixed16` 的 `Stage A` 训练线：

- 训练输出根：
  - `outputs/llamafactory_qwen2_5vl_3b_paper16_patch_only_full_trainval_geomdedup_fixed16_empty10_stdce_8gpu_e10/`
- 推荐外部评测 checkpoint：
  - `outputs/llamafactory_qwen2_5vl_3b_paper16_patch_only_full_trainval_geomdedup_fixed16_empty10_stdce_8gpu_e10/checkpoint-157182/`

## 需要放到仓库根目录的资源

- 基础模型：
  - `ckpts/modelscope/Qwen/Qwen2___5-VL-3B-Instruct/`
- 训练集：
  - `dataset/paper16_patch_only_full_trainval_geomdedup_fixed16_empty10_system/`
- `unseen-lite` grouped eval 数据：
  - `dataset/paper16_patch_only_full_trainval_fixed16_unseenlite305515_empty08_from_refs_allboxes/`
- 训练输出 / adapter：
  - `outputs/llamafactory_qwen2_5vl_3b_paper16_patch_only_full_trainval_geomdedup_fixed16_empty10_stdce_8gpu_e10/`

## 入口脚本

- 训练入口：
  - `scripts/launch_stagea_full_trainval_geomdedup_fixed16_train.sh`
- `checkpoint-157182` 的 grouped `unseen-lite` 评测入口：
  - `scripts/launch_stagea_full_trainval_geomdedup_fixed16_unseenlite_eval_ckpt157182.sh`

## 配置目录

- `configs/llamafactory_paper16_patch_only_full_trainval_geomdedup_fixed16_empty10_system/`
  - `dataset_info.json`
  - `qwen2_5vl_3b_lora_sft_stdce_8gpu_e10.yaml`
  - `README.md`

训练入口会像 `StageA-Min` 原分支一样，在 `.runtime/` 下自动生成运行时 `dataset_info.json` 和运行时 yaml，
因此不会依赖配置文件里原本残留的绝对路径。

## 默认路径约定

- 训练默认读取：
  - `dataset/paper16_patch_only_full_trainval_geomdedup_fixed16_empty10_system/`
- 训练默认输出到：
  - `outputs/llamafactory_qwen2_5vl_3b_paper16_patch_only_full_trainval_geomdedup_fixed16_empty10_stdce_8gpu_e10/`
- 评测默认读取固定 checkpoint：
  - `outputs/llamafactory_qwen2_5vl_3b_paper16_patch_only_full_trainval_geomdedup_fixed16_empty10_stdce_8gpu_e10/checkpoint-157182/`

## 已知外部评测结果

`checkpoint-157182` 在 grouped `unseen-lite` 上的记录结果是：

- `mIoU = 0.27603`
- `APM = 0.01084`
- `APC_0.9 = 0.05242`
- `APC_1.5 = 0.08470`
- `APC_3.0 = 0.18934`
- `APC_4.5 = 0.26942`
- `mean_chamfer_px = 38.57`
- `pred_num_lines = 15.37`
- `gt_num_lines = 11.42`

这个结果是 `unseen-lite` 的 mid-difficulty 参考，不是最终 strict unseen authority。

## 可覆盖环境变量

训练入口支持：

- `MODEL_DIR`
- `DATASET_ROOT`
- `OUTPUT_DIR`

评测入口支持：

- `EVAL_ROOT`
- `DATASET_JSONL`
- `META_JSONL`
- `BASE_MODEL`
- `PROCESSOR_PATH`
- `ADAPTER`
- `OUTPUT_DIR`
- `DEVICE`
- `MAX_NEW_TOKENS`
- `MAX_SOURCE_PATCHES`
