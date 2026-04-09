# 数据集制作四段主链说明



只保留了数据集制作主链

## 目录结构

- `scripts/`
  - 4 个主脚本入口
- `unimapgen/dataset_build_refactor/`
  - 这些主脚本依赖的 helper 包

## 保留的 4 个主脚本

1. `build_opensatmap_paper16_family_manifest.py`
   - 从原始 `4096x4096` 图像定义 `paper16 family manifest`
2. `export_llamafactory_patch_only_from_raw_family_manifest.py`
   - 把 `family manifest + 原始标注` 导成 `patch-only` 数据集
3. `build_patch_only_fixed_grid_targetbox_dataset.py`
   - 把 `patch-only` 数据集展开成 `fixed16 Stage A` 数据集
4. `build_stageb_fixed16_gt_point_angle_dataset.py`
   - 把 `fixed16 Stage A` 数据集再加工成 `Stage B` 数据集

## helper 包的作用

`unimapgen/dataset_build_refactor/` 里是这 4 个脚本共用的逻辑：

- `common.py`
  - json/jsonl 读写、ShareGPT 记录组装、dataset_info 生成
- `geometry.py`
  - 折线裁剪、重采样、方向规范化、排序
- `patch_only.py`
  - patch-only 目标构造
- `fixed16.py`
  - fixed16 box 构造和 box 内目标裁剪
- `stageb.py`
  - Stage B 邻居 trace 提取和 prompt 构造
- `viz.py`
  - 还保留在 helper 里，但当前这 4 个主脚本已经不再走可视化分支

## 使用时的路径约束

这 4 个脚本在代码里会把“脚本所在目录的上一级”当成仓库根目录，然后从那里导入：

- `unimapgen.dataset_build_refactor`

所以这个打包目录里要保持现在这种相对结构：

```text
bundle_root/
  scripts/
  unimapgen/
    dataset_build_refactor/
```

如果改动目录结构，脚本里的 import 路径也要一起调整。

## 备注

- 这份打包目录来自当前整理后的 refactor 版本
- 原仓库里的 `scripts/` 原版脚本没有被改动

