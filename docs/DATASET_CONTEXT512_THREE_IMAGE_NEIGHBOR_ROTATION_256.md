# Context512/ROI256 三图邻居旋转数据集

## 1. 数据集身份

这是一版基于已完成的 Context512/ROI256 三图数据集生成的空间增强消融集。
它的目标是让相邻的 256 网格样本看到不同的局部朝向，检验模型是否过度记忆
相邻 Context512 窗口中的重复内容。

| 项目 | 值 |
|---|---|
| dataset_variant | context512_roi256_three_image_neighbor_rotation_256 |
| 输入数据集 | D:\data\rc_dataset_v2_three_image_800k_from_obs\output_three_image_800k\context512_roi256_rawlane_pose_800k |
| 输出目录 | D:\data\context512_roi256_three_image_neighbor_rotation_256 |
| 输出压缩包 | D:\data\context512_roi256_three_image_neighbor_rotation_256.tar |
| 训练网格 | 256 像素 stride |
| 输入图像 | 每条样本 3 张 512x512 图 |
| 监督区域 | 输入图像中心的固定 256x256 ROI |
| 坐标 | ROI 相对 norm1000，范围 0..1000 |
| 旋转角度 | 0°、45°、135° |

构建命令：

~~~powershell
python scripts\tools\build_context512_roi256_three_image_ablation_windows.py --mode neighbor_rotation --input-root "D:\data\rc_dataset_v2_three_image_800k_from_obs\output_three_image_800k\context512_roi256_rawlane_pose_800k" --output-root "D:\data\context512_roi256_three_image_neighbor_rotation_256" --neighbor-angles "0,45,135" --neighbor-grid-stride 256 --neighbor-source-grid-policy filter --copy-mode hardlink --image-resample bilinear --package-path "D:\data\context512_roi256_three_image_neighbor_rotation_256.tar" --progress-every 10000
~~~

最终训练样本数不能根据目录名称猜。如果输入集不是严格的 256 网格，
--neighbor-source-grid-policy filter 会丢弃训练集中 x0 或 y0 未对齐 256
的记录。因此最终训练数以输出目录中的 balance_report.json、
ablation_validation.json 和 dataset_info.json 为准，不要默认假设为 800000。

## 2. 目录与样本格式

压缩包解开后，顶层目录通常为：

~~~text
context512_roi256_three_image_neighbor_rotation_256/
  images/
    train/...
    eval/...
    test/...
  raw_lane_images/
    train/...
    eval/...
    test/...
  pose_images/
    train/...
    eval/...
    test/...
  phase_a/
    train.jsonl
    eval.jsonl
    test.jsonl
    meta_train.jsonl
    meta_eval.jsonl
    meta_test.jsonl
  dataset_info.json
  balance_report.json
  ablation_validation.json
  build_summary.json
  build_complete.json
  split_manifest.json
~~~

每条 Stage A 记录使用三张独立图片，顺序固定为：

~~~text
1. clean BEV road-structure image
2. PV camera model raw-lane image
3. historical vehicle-trajectory (pose) image
~~~

训练 loader 应优先读取记录中的 images 数组并保持顺序。记录中若同时保留
raw_lane_image、pose_image 等辅助字段，它们只能作为冗余索引，不能改变
images 的三图顺序。

用户消息必须包含恰好三个 <image> 标记，assistant target 仍为现有的
{"lines":[...]} JSON schema。邻居旋转构建器不会重新设计 prompt 或标签
语义，只变换图像和几何 target。

## 3. 邻居旋转规则

对于训练记录，按原始全局网格坐标计算：

~~~text
grid_x = x0 // 256
grid_y = y0 // 256
phase = (grid_x + grid_y) % 3
angle = [0, 45, 135][phase]
~~~

这会使水平和垂直直接邻居处于不同 phase。只有三个 phase，因此它不是完整的
八邻居图着色；某个对角方向仍可能共享角度。

每个被选中的训练记录只保留一行，不增加副本：

- 0°：保持原样；
- 45° 或 135°：三张 512x512 图围绕 (255.5, 255.5) 同步旋转；
- clean BEV 使用 --image-resample bilinear；
- Raw-Lane 和 Pose 使用最近邻插值，避免稀疏白像素被插值变粗或变淡；
- 原 ROI 内的中心线和路口 polygon 同步映射到 Context512 像素坐标、旋转、
  裁回固定中心 ROI [128,128,384,384)，再转回 ROI 相对 norm1000；
- 几何 target 发生旋转和裁剪，不能继续使用未变换的旧坐标。

旋转只应用于 train。eval 和 test 保持输入数据集中的原始样本与坐标，
因此可以与未旋转基线复用同一评估集进行比较。

### 全局坐标注意事项

非零角度是局部增强，不再对应原大图中的真实物理位置。构建器会将
global_coordinates_valid 设为 false，并将源坐标放在 source_global_metadata
中供追溯。不要把这些训练记录用于整图拼接、跨 patch 全局几何评估或重新生成
Stage B 的真实邻居状态。

### 与 nonoverlap 的区别

该版本只改变相邻样本的朝向，不消除 Context512 窗口的物理重叠。训练网格
stride 为 256 时，相邻 512 窗口仍有约 50% 的边长重叠。严格消除重叠应使用
单独的 --mode nonoverlap 数据集，不能把本数据集称为无重叠集。

## 4. 标签与坐标契约

该版本继承输入三图数据集的标签，不在转换阶段新增或删除道路类别。当前主线
通常使用以下 taxonomy，但 DI 训练前必须以输出的 dataset_info.json 和代表性
JSONL 为最终依据：

~~~text
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
~~~

坐标规则：

~~~text
x_norm = round(x_roi_pixel / 255 * 1000)
y_norm = round(y_roi_pixel / 255 * 1000)
~~~

x_roi_pixel、y_roi_pixel 是中心 256 ROI 内的坐标，不是整张 512 图的坐标。
模型端 DINOv2 仍按主线配置将每张输入 resize 到视觉编码器要求的 518；这是
模型预处理尺寸，不改变数据集的 512 输入和 ROI256 标注契约。

## 5. 构建结果验收

首先查看以下文件，不要只看文件夹名：

~~~text
dataset_info.json          # 视图、尺寸、三图角色、旋转策略
balance_report.json        # 训练数、过滤数、各角度数量、空样本比例
ablation_validation.json   # 图片解码、三图配对、prompt 和坐标检查
build_summary.json         # 构建汇总与校验结果
build_complete.json        # 是否完整结束
split_manifest.json        # split 计数与 ID 摘要
~~~

Windows 上快速打印核心字段：

~~~powershell
python -c "import json; from pathlib import Path; r=Path(r'D:\data\context512_roi256_three_image_neighbor_rotation_256'); names=['dataset_info.json','balance_report.json','ablation_validation.json','build_summary.json','build_complete.json','split_manifest.json']; [(print('\n###',n), print(json.dumps(json.loads((r/n).read_text(encoding='utf-8')), ensure_ascii=False, indent=2)) if (r/n).is_file() else print('MISSING')) for n in names]"
~~~

至少确认：

1. dataset_info.json.dataset_variant 等于
   context512_roi256_three_image_neighbor_rotation_256；
2. context_size=512、target_size/target_patch_size=256、
   coord_mode=norm1000、coord_range=1000；
3. three_image_input.num_images_per_sample=3，角色顺序为 BEV、Raw-Lane、Pose；
4. train 的 neighbor_rotation_angle_counts 同时记录 0、45、135（如果源数据
   对应 phase 有记录）；
5. neighbor_filtered_train_samples 被明确记录；
6. 抽查的三张图片都能解码为 512x512，且每条 prompt 恰好有三个 <image>；
7. 训练 target 的坐标在 0..1000，中心线/路口 JSON 可解析；
8. eval/test 未被旋转，split 计数和固定评估集策略符合项目约定。

检查压缩包顶层目录：

~~~powershell
python -c "import tarfile; p=r'D:\data\context512_roi256_three_image_neighbor_rotation_256.tar'; print(sorted({m.name.split('/')[0] for m in tarfile.open(p) if m.name}))"
~~~

## 6. DI 训练交接

### 6.1 上传

训练需要上传新的 tar 包，例如：

~~~text
obs://<bucket>/<prefix>/context512_roi256_three_image_neighbor_rotation_256.tar
~~~

dataset_info.json、balance_report.json 等报告建议保留在 tar 内；如果已经在包内，
不需要再单独上传。训练入口只需要 DATASET_OBS_PATH 指向 tar。

上传后先用 tar -tf 或 DI 下载后的目录确认真实顶层目录名。通常应为：

~~~text
context512_roi256_three_image_neighbor_rotation_256
~~~

### 6.2 推荐训练路线

本数据集只改变数据，不改变 Jiangjihua 主线模型结构。推荐先用原始 DINOv2-Large
+ CapRL-Qwen3VL-4B 派生文本 LLM、no-DeepStack、LLM LoRA 的正式入口做 smoke，
再做完整训练：

~~~bash
EXPECTED_DATASET_VARIANTS=context512_roi256_three_image_neighbor_rotation_256 \
DATASET_DIR_NAME=context512_roi256_three_image_neighbor_rotation_256 \
DATASET_OBS_PATH=obs://<bucket>/<prefix>/context512_roi256_three_image_neighbor_rotation_256.tar \
EXPECTED_TRAIN_SAMPLES=<从 balance_report.json 读取的实际 train 数> \
MAX_STEPS=20 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
TARGET_GLOBAL_BATCH_SIZE=128 \
bash scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_three_image_context512_roi256_800k_original_dinov2_caprl4b_nodeepstack_lora_llm_npu.sh
~~~

当前脚本文件名带有 800k，但它只是历史 recipe 名称；邻居旋转包经过 256 网格
过滤后不一定有 800k 条训练记录，所以必须覆盖 EXPECTED_TRAIN_SAMPLES，不能
直接使用默认值。

smoke 通过后，将 MAX_STEPS 改为 -1，再按 DI 实际节点数和显存设置
PER_DEVICE_TRAIN_BATCH_SIZE 与 TARGET_GLOBAL_BATCH_SIZE。三图输入会带来约三倍
视觉 token 流，必须保持 MODEL_MAX_LENGTH>=6144；主线默认 8192，不要为了提高
batch 而降低到无法容纳三路视觉 token 的长度。

入口脚本会做严格数据预检。若报 dataset_variant 不匹配，检查
EXPECTED_DATASET_VARIANTS 是否覆盖新 variant；若报训练记录数不匹配，重新读取
balance_report.json，不要重新采样或修改 tar 包。

### 6.3 DI 训练边界

- DI 使用预先筛选好的 train，不应在启动时再次按难度重采样；
- DINOv2 的视觉输入仍是 518，数据文件仍是 512；
- 三图顺序必须保持一致，不能把 Raw-Lane 或 Pose 拼回 BEV；
- 日志应打印 DI_throughput: ... samples/s/npu 或
  DI_throughput: ... tokens/s/npu；
- 训练 checkpoint 可以用于普通 Stage A 推理，但旋转 train 样本的
  global_coordinates_valid=false 不影响 eval/test 的正常指标计算；
- 和基线比较时必须使用相同的 eval/test 图像、标签、prompt、模型初始化和指标
  脚本，唯一变量应是 train 的邻居旋转策略。

## 7. 这版实验回答什么问题

它主要回答：在保留 256 stride 的数据覆盖率和三图输入的前提下，让相邻样本拥有
不同方向，是否能降低模型对重复 Context512 内容的记忆，并改善跨邻居泛化。它
不能单独回答“更大上下文是否有效”或“去除窗口重叠是否有效”；这两类问题应分别
与普通 Context512/ROI256 和 strict nonoverlap 数据集比较。

