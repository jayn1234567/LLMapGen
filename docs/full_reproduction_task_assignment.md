# UniMapGen 完整复现任务分工文档

## 1. 目标与边界

本项目目标不是“把当前 smoke pipeline 跑通”，而是完成对 `/mnt/data/project/jn/UniMapGen/UniMapGen.pdf` 的工程级完整复现，包含：

- 论文主干模型链路复现
- State Update 完整训练与推理复现
- 正式数据构建与增强流程复现
- 多模态输入分支复现（至少补齐 PV）
- 论文主要实验、消融与评估复现
- 可重复训练、评估、推理和文档交付

当前已经完成的部分见：
- [reproduction_status_20260306.md](/mnt/data/project/jn/UniMapGen/docs/reproduction_status_20260306.md)
- [qwen_dinov2_map_serialization_branch.md](/mnt/data/project/jn/UniMapGen/docs/qwen_dinov2_map_serialization_branch.md)

当前已完成部分可以作为后续工作的基线，而不是推倒重来。

## 2. 当前基线

当前仓库已经具备以下可运行基线：

- 卫星图分支：`DINOv2 ViT-L/14 -> proj -> Qwen2.5-1.5B`
- 地图序列化：polyline -> discrete map token -> detokenizer
- State Update：训练侧 previous state + 推理侧 scan update
- 局部邻接 state：当前 patch 只参考邻近历史，而不是全部全局历史
- 稳定解码：语法约束 + repetition penalty
- GPU 环境与本地权重加载可用

关键代码：
- [satellite_encoder.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/encoders/satellite_encoder.py)
- [serialization.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/serialization.py)
- [qwen_map_tokenizer.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_tokenizer.py)
- [qwen_map_dataset.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_dataset.py)
- [qwen_map_generator.py](/mnt/data/project/jn/UniMapGen/unimapgen/models/qwen_map_generator.py)
- [predict_qwen_state_scan.py](/mnt/data/project/jn/UniMapGen/unimapgen/predict_qwen_state_scan.py)
- [state_geometry.py](/mnt/data/project/jn/UniMapGen/unimapgen/state_geometry.py)

## 3. 与论文的主要剩余差异

还没有完成的核心差异可以归纳为六块：

1. 正式对齐卫星数据与 patch 几何元数据
2. 论文规模的数据构建与增强流程
3. 论文多模态分支，尤其是 PV 分支
4. 更贴近论文的 tokenizer / prompt / state 表达细节
5. 训练策略、正式配置与大规模实验管理
6. 论文最终评估、消融与复现报告

这六块建议拆成独立工作包，由不同角色负责。

## 4. 推荐团队角色

建议按 6 个角色分工；如果团队人数不足，可以合并相邻角色。

### 4.1 角色 A：数据与几何负责人

负责：
- 正式对齐卫星数据接入
- patch 裁剪、重叠裁剪、旋转/倾斜增强
- patch 几何元数据生成
- patch 扫描顺序和邻接关系定义

要求：
- 熟悉数据清洗、几何坐标和批量数据管线
- 能独立做大规模 patch 构建和质量检查

### 4.2 角色 B：地图序列化与状态建模负责人

负责：
- 当前 serialization 分支继续对齐论文
- tokenizer 细节、state prefix 形式
- start/end/cut 语义、跨 patch state merge
- detokenizer 和约束解码完善

要求：
- 熟悉结构化序列建模
- 能同时处理数据表示、解码规则和几何约束

### 4.3 角色 C：大模型与多模态建模负责人

负责：
- Qwen 主干训练策略
- DINOv2 prefix 注入
- 补齐 PV 分支
- 多模态融合、模态 mask、prompt 体系

要求：
- 熟悉 Hugging Face / PyTorch / 大模型训练
- 能处理 prefix tuning、embedding 拼接和多模态输入

### 4.4 角色 D：训练系统与实验平台负责人

负责：
- 环境固化
- 正式训练配置整理
- 日志、checkpoint、恢复训练、脚本管理
- 多卡、长训练、显存和吞吐排障

要求：
- 熟悉训练工程化、脚本化和资源管理
- 能把“能跑”变成“稳定可复现”

### 4.5 角色 E：评估与实验分析负责人

负责：
- 评估脚本
- 论文主指标对齐
- 消融实验设计与结果整理
- 错误案例分析和可视化

要求：
- 熟悉实验设计、统计对比和误差分析
- 能产出汇报级图表和结论

### 4.6 角色 F：项目集成与文档负责人

负责：
- 各模块接口约束
- 每周合并、集成测试
- 文档同步、版本记录
- 阶段交付物验收

要求：
- 能从项目整体视角约束接口和节奏
- 能识别“局部看起来能跑、整体其实不闭环”的问题

## 5. 任务分工

### 工作包 1：正式数据与 patch 管线复现

负责人：角色 A
协作：角色 F

目标：
- 用正式对齐的卫星图替换当前 small-set
- 复现论文的数据构建方式，而不是只保留当前 smoke 版本

具体任务：
- 整理正式卫星图数据目录规范
- 为每个 patch 生成稳定的 `token -> image -> annotation -> geometry` 映射
- 生成标准 patch 几何元数据：中心、裁剪框、原图来源、邻接关系、扫描顺序索引
- 复现 patch 级增强：重叠裁剪、旋转增强、倾斜裁剪
- 生成 train/val/test 划分与元信息文件
- 输出数据健康检查脚本：空标注、重复 patch、越界点、类别异常、几何断裂

交付物：
- 正式 patch 数据目录
- `patch_geometry.json` 或等价元数据文件
- 数据构建脚本和说明文档
- 数据质检报告

验收标准：
- 任意 patch 都能回溯到原图和原始标注
- patch 几何信息能驱动真实 `left-to-right, top-to-bottom` 扫描
- 数据构建脚本可重复运行，输出一致

依赖：
- 你后续提供对齐后的正式卫星数据

### 工作包 2：序列化、detokenizer 与 state 表达对齐论文

负责人：角色 B
协作：角色 A、角色 C

目标：
- 在当前可运行 serialization 基线基础上，进一步贴近论文而不是停留在工程近似

具体任务：
- 审核当前 token 设计和论文附录/正文的差异
- 评估是否保留当前 `<x_i>/<y_i>` bin token，还是改成更贴论文的整词/整数字表示
- 保持 encode/decode 双向可逆
- 明确 state prefix 的最终形式：`cut_points` 还是 `cut_traces`
- 完善 start/end/cut 的几何与语义约束
- 继续提升跨 patch merge 规则：投影、裁剪、连接、去重
- 把训练侧 state 和推理侧 state 完全统一到同一套几何抽取逻辑

当前基线文件：
- [serialization.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/serialization.py)
- [qwen_map_tokenizer.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_tokenizer.py)
- [qwen_map_dataset.py](/mnt/data/project/jn/UniMapGen/unimapgen/data/qwen_map_dataset.py)
- [state_geometry.py](/mnt/data/project/jn/UniMapGen/unimapgen/state_geometry.py)

交付物：
- 定稿版 map serialization 规范
- 定稿版 state serialization 规范
- 单元测试：encode/decode 一致性、state 投影与 merge 一致性
- 更新后的技术文档

验收标准：
- `serialize -> detokenize` 在规范样例上稳定可逆
- 训练侧和推理侧 state 构造逻辑一致
- 与当前 small-set 和正式数据都兼容

### 工作包 3：多模态主干补齐，优先 PV 分支

负责人：角色 C
协作：角色 A、角色 D

目标：
- 从当前卫星图单分支扩展到更接近论文的多模态结构

具体任务：
- 梳理论文中 PV 分支输入形式
- 接入 PV 数据集与相机图像读取
- 实现 PV encoder 与时序聚合
- 设计多模态 prefix 拼接接口：satellite / PV / text / state
- 实现训练时模态随机 mask
- 支持至少以下模式：
  - satellite-only
  - satellite + state
  - PV-only
  - satellite + PV
  - satellite + PV + text + state
- 保持和当前 Qwen 主线兼容

交付物：
- PV 数据读取与特征提取模块
- 多模态 Qwen 训练/推理脚本
- 模态组合配置文件
- smoke 和最小训练日志

验收标准：
- 至少 3 种模态组合可稳定训练和推理
- 模态 mask 可配置可复现
- 不破坏当前 satellite-only 分支

### 工作包 4：训练工程、环境与大规模实验脚本

负责人：角色 D
协作：角色 C、角色 F

目标：
- 把当前“靠手工命令跑通”的状态升级为稳定实验平台

具体任务：
- 固化环境：`requirements.txt` 或 `environment.yml`
- 清理和统一训练脚本、评估脚本、预测脚本入口
- 支持断点恢复、best/latest checkpoint 管理
- 整理正式训练配置：small / base / full
- 增加资源监控：显存、速度、数据吞吐、失败自动定位信息
- 准备多卡训练方案，如果机器条件允许则支持 DDP
- 统一输出目录格式与实验元信息记录

交付物：
- 固化环境文件
- 训练/评估/推理脚本模板
- 正式配置集
- 实验运行手册

验收标准：
- 新成员按文档可独立拉起训练
- 训练异常时能定位到配置、数据或模型模块
- 重跑同一配置能得到一致日志结构和产物目录

### 工作包 5：评估、消融与论文结果对齐

负责人：角色 E
协作：角色 B、角色 C、角色 D

目标：
- 从“代码能跑”转向“论文实验能复核”

具体任务：
- 梳理论文主指标与评估协议
- 实现或校验评估脚本
- 建立标准实验表：
  - 无 state vs 有 state
  - cut_points vs cut_traces
  - 约束解码 vs 非约束解码
  - satellite-only vs satellite+PV
  - 小数据 vs 正式数据
- 建立错误案例池：断裂、重复、短线、跨 patch 不连续
- 形成每周实验摘要和决策建议

交付物：
- 评估脚本与结果表
- 消融对比文档
- 错误案例可视化报告
- 最终复现报告初稿

验收标准：
- 每次正式训练都有固定评估产物
- 主要实验设置和结论可复核
- 可以明确说明与论文结果的差距来自哪里

### 工作包 6：项目集成、接口控制与文档

负责人：角色 F
协作：全员

目标：
- 保证各人做的模块能合起来，而不是形成多个孤立分支

具体任务：
- 维护模块接口文档
- 每周组织一次集成分支合并
- 统一代码风格和目录结构
- 维护阶段性里程碑文档
- 审查“和论文一致”这种表述是否有证据支撑
- 维护风险清单与阻塞项列表

交付物：
- 周报模板
- 里程碑状态文档
- 风险清单
- 集成测试记录

验收标准：
- 关键接口变更都有文档记录
- 任何阶段都能说明：做到了什么、没做到什么、为什么
- 合并后主分支始终可运行

## 6. 推荐里程碑

### M1：正式数据闭环

目标：
- 完成对齐卫星数据接入
- 形成真实 patch geometry
- 当前 satellite + serialization + state 分支切换到正式数据

里程碑判定：
- 训练、推理、state-scan 都能在正式数据上跑通
- 不再依赖当前 small-set 的近似几何

主负责人：角色 A、角色 B

### M2：序列化与 state 机制定稿

目标：
- 确定 tokenizer 方案
- 确定 state prefix 最终形式
- 统一训练和推理状态构造

里程碑判定：
- serialization 规范冻结
- state 构造规范冻结
- detokenizer、merge、约束解码接口稳定

主负责人：角色 B

### M3：多模态主线打通

目标：
- 补齐 PV 分支
- 支持多模态组合训练
- 支持模态随机 mask

里程碑判定：
- 至少 3 种模态组合稳定运行
- 可输出最小可用结果和日志

主负责人：角色 C、角色 D

### M4：正式训练与评估

目标：
- 形成 base/full 配置
- 开始跑论文核心实验和消融

里程碑判定：
- 有稳定训练曲线
- 有成体系对比表
- 能说明当前与论文差距

主负责人：角色 D、角色 E

### M5：复现收口

目标：
- 完成最终代码、文档、结果和风险说明

里程碑判定：
- 代码可交付
- 文档可交付
- 实验可复核
- 与论文差异说明完整

主负责人：角色 F

## 7. 依赖关系

建议按如下依赖推进：

1. 先做工作包 1
2. 同时推进工作包 2 的规范设计
3. 工作包 1 稳定后再放大工作包 4
4. 工作包 3 在数据闭环后进入主线
5. 工作包 5 必须等工作包 2 到 4 稳定后再全面展开
6. 工作包 6 从第一周开始贯穿全程

不要反过来做：
- 在正式数据未稳定前大规模追指标
- 在 tokenizer/state 规范未定稿前做大量消融
- 在训练平台未稳定前跑长程正式实验

## 8. 每周管理建议

建议每周固定输出四类内容：

1. 本周新增能力
- 新增了什么，不是做了什么尝试

2. 当前阻塞项
- 数据、显存、接口、精度、时间成本

3. 下周承诺交付
- 必须是可验收产物，不是开放性研究方向

4. 风险变化
- 哪些风险变小了，哪些风险变大了

## 9. 建议的人员配置映射

如果你有 6 人：
- 1 人做数据与几何
- 1 人做序列化与 state
- 1 人做大模型与多模态
- 1 人做训练工程
- 1 人做评估分析
- 1 人做集成与文档

如果你只有 4 人：
- 数据与几何 + 集成文档 合并
- 序列化与 state 单独保留
- 大模型与多模态 + 训练工程 合并
- 评估分析单独保留

如果你只有 3 人：
- 数据与几何
- 模型、序列化与 state
- 训练工程、评估与集成

## 10. 当前最优先的三件事

按现在的项目状态，优先级最高的是：

1. 接入对齐后的正式卫星数据并生成正式 patch geometry
2. 把训练和推理都彻底切到正式的局部邻接 state 逻辑
3. 开始补 PV 分支的最小可用版本

这三件事决定后面做的是“完整复现”，还是继续停留在“单分支可运行验证”。
