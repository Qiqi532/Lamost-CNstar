<div align="center">

# LAMOST CN-Enhanced Star Detection

基于光谱、物理先验与 Positive–Unlabeled Learning 的 CN 增强星候选体筛选研究

<p>
  <img src="https://img.shields.io/badge/data-LAMOST%20DR13-2455a4" alt="LAMOST DR13">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776ab" alt="Python">
  <img src="https://img.shields.io/badge/notebooks-Jupyter-f37626" alt="Jupyter">
  <img src="https://img.shields.io/badge/task-positive--unlabeled%20learning-6f42c1" alt="PU learning">
</p>

</div>

> 本仓库保存研究代码与实验 Notebook。原始星表、FITS 光谱、缓存、模型权重和生成图像不随仓库同步；请按本地数据路径配置后运行。

## 目录

- [研究问题](#研究问题)
- [当前结论](#当前结论)
- [方法与结果](#方法与结果)
- [数据与预处理](#数据与预处理)
- [仓库结构](#仓库结构)
- [从哪里开始](#从哪里开始)
- [环境与复现](#环境与复现)
- [研究边界与下一步](#研究边界与下一步)

## 研究问题

目标是从 LAMOST DR13 低分辨率光谱中识别 **CN 增强恒星**。CN 星数量极少，而未标注样本中可能仍包含大量未知目标，因此问题不是普通的平衡二分类，而是一个极端不平衡的 **Positive–Unlabeled（PU）学习**任务：

- 光谱范围：约 **3800–4500 Å**；连续谱归一化后为 **700 维**输入
- 最新合并缓存：**41,243 颗恒星 / 91 颗已知 CN 星**，正样本比例约 1:453
- 历史基线缓存：**33,565 颗恒星 / 73 颗已知 CN 星**，正样本比例约 1:459
- 目标不是给未标注样本强行赋予“负类”标签，而是从 U（unlabeled）中挖掘新的 CN 候选体

三个核心分子带为：

| 分子带 | 波长范围 | 研究用途 |
| --- | --- | --- |
| CN3839 | 3830–3883 Å | 最强的 CN 特征之一 |
| CN4142 | 4120–4216 Å | 较宽的 CN 分子带 |
| CH4300 | 4285–4315 Å | 辅助区分 C/N 丰度关系 |

## 当前结论

<div align="center">

| 方法 | ROC-AUC | PR-AUC | Precision@50 | Precision@100 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| **XGBoost PU Bagging（最新）** | **0.9883** | **0.6030** | **0.20** | **0.11** | 当前最稳定的标杆 |
| EndToEndPU + nnPU | 0.9456 | 0.3070 | 0.10 | 0.06 | 有效但对划分敏感 |
| EndToEndPU + 加权 BCE | 0.9044 | 0.1533 | 0.10 | 0.06 | 低于 nnPU |
| Label Spreading | — | — | 1.00* | 0.73* | 有潜力，需独立测试集 |
| DeepSVDD | 0.4495 | — | 0.00 | 0.00 | 单类异常检测路线失败 |

</div>

*Label Spreading 的 Precision@K 来自已知标签上的 pseudo-metrics，不能等同于独立测试集性能。

## 最新实验进展（2026-09）

### CrossFit 基线与新增标签审计

在 91 颗已知 CN 星上训练的原始 CrossFit，对 99 个新增高分辨率标签进行冻结模型外部审计，得到 ROC-AUC **0.911 ± 0.031**、PR-AUC **0.876**。该结果保留为 M2 实验的基线，不应与加入新增标签后的 OOF 指标混为同一评估协议。

### M2 新增标签训练

新增标签包含 **46 个正例**和 **53 个可靠负例**。M2 使用严格的 5 折 OOF 协议：每个新增对象只由未见过它的模型评分，并从正例、可靠负例和普通 U 采样池中同时排除 held-out 标签，避免标签泄漏。

| 结果 | ROC-AUC | PR-AUC | 说明 |
| --- | ---: | ---: | --- |
| 99 个新增标签严格 OOF | **0.939 ± 0.026** | **0.909** | M2 独立评价协议 |
| 190 个已标记对象总体 OOF | **0.926** | **0.971** | 137 正例 + 53 可靠负例 |

基于 137 个确认正例的 OOF 分数，90% 召回阈值为 **0.249882**，保留 124/137 个正例，导出 **393 个**未标注候选。95% 召回阈值为 **0.098071**，实际召回率 **95.62%**，导出 **1,081 个**候选。原 91 颗已知星按90%阈值重新标记后，模型识别 **79 颗**、漏检 **12 颗**。这些分数是 PU 排序分数，不是校准后的后验概率。

完整流程见 [m2_crossfit/M2_crossfit_full_pipeline.ipynb](m2_crossfit/M2_crossfit_full_pipeline.ipynb)，训练引擎见 [m2_crossfit/m2_crossfit_engine.py](m2_crossfit/m2_crossfit_engine.py)，紧凑结果导出见 [m2_crossfit/export_compact_tables.py](m2_crossfit/export_compact_tables.py)。

### 光谱特征二次核验

`feature/` 对冻结的 654 个 XGBoost 候选进行 CN 指数和差分面积两条路径的二次分层。两种方法都支持候选排序、簇内参考比较和已知星召回统计；鲁棒双指标筛选分别保留约 222 个对象。该阶段用于候选分层和光谱证据交叉检查，不把指数法或面积法单独宣称为独立确认分类器。

### 当前研究判断

1. XGBoost PU/CrossFit 仍是当前最稳定的候选排序主线。
2. 新增高分辨率标签显著提升了 M2 的独立 OOF 区分能力，但 91 颗已知星中仍有 12 颗漏检，应优先作为困难星体核验集。
3. 90% 与 95% 阈值适合产生不同规模的观测候选池；实际确认仍需结合光谱质量、CN/CH 指数、差分面积和人工复核。
4. feature 阶段更适合作为 M2 候选的二次证据排序，而不是替代独立高分辨率标签。

目前最重要的研究判断是：

1. **XGBoost PU Bagging 是当前最可靠的候选体排序方法。** 在最新合并缓存上，500 次 PU bagging 得到 ROC-AUC=0.9883、PR-AUC=0.6030，并按已知 CN 星得分分布标定阈值，筛出约 890 个候选体（2.16%）。
2. **CN 物理先验值得保留。** EndToEndPU 的 CN 波段注意力显著优于去掉注意力的消融版本；但深度网络受正样本数量少、参数量大和随机划分影响，训练稳定性仍不足。
3. **图半监督方法可能提供互补候选体。** Label Spreading 在已有标签上的 top-K 指标较好，但必须用独立验证策略排除标签泄漏和评价偏差。
4. **更复杂不一定更好。** DeepSVDD 的 AUROC 低于随机水平，纯物理特征 MLP 也出现明显过拟合，说明后续工作应优先改善样本覆盖、验证设计和物理归纳偏置。

## 方法与结果

### 1. 物理筛选：T-physics

通过 CN 分子带拟合、连续谱估计和面积阈值进行候选体筛选。历史实验得到 1,929 个候选体，但只召回 73 个已知 CN 星中的 38 个（52.1%）。它适合作为可解释的先验筛选与交叉验证方法，不适合作为唯一分类器。

入口 Notebook：[PhaseSummary/01_T_physics/T_physics.ipynb](PhaseSummary/01_T_physics/T_physics.ipynb)

### 2. 经典机器学习：XGBoost PU Bagging

核心做法是每一轮从未标注样本中随机采样伪负样本，与已知正样本组成平衡训练集，重复训练 500 个浅层 XGBoost 分类器并平均输出分数。该方案直接使用 700 维归一化光谱，并可结合 masked-band 聚类和 cluster z-score 降低恒星物理参数造成的偏差。

推荐入口：

- [PhaseSummary/03_ML_XGB/ML_XGB_PU.ipynb](PhaseSummary/03_ML_XGB/ML_XGB_PU.ipynb)：主实验与历史基线
- [PhaseSummary/03_ML_XGB/ML_XGB_PU_threshold.ipynb](PhaseSummary/03_ML_XGB/ML_XGB_PU_threshold.ipynb)：阈值标定实验
- [ML/pu_bagging.py](ML/pu_bagging.py)：PU bagging 核心实现
- [PhaseSummary/03_ML_XGB/tune_xgb_pu.py](PhaseSummary/03_ML_XGB/tune_xgb_pu.py)：参数调优脚本
- [XGB/](XGB/)：匹配、KDE、关键波段和两阶段候选体实验

### 3. 端到端深度 PU：1D ResNet + CN Attention

EndToEndPU 使用约 380 万参数的 1D ResNet，并在输入端加入可学习的 CN 波段波长门控，在残差块中加入 SE 通道注意力。训练流程包含正样本物理增强、hard-negative mining、Mixup、梯度裁剪、权重衰减、余弦退火和早停；同时实现加权 BCE、uPU 和 nnPU 损失。

推荐入口：

- [EndToEndPU/EndToEndPU_Summary.ipynb](EndToEndPU/EndToEndPU_Summary.ipynb)：综合结果与误差分析
- [EndToEndPU/EndToEndPU_Method.ipynb](EndToEndPU/EndToEndPU_Method.ipynb)：方法说明
- [EndToEndPU/nnPU_Tuning_Summary.ipynb](EndToEndPU/nnPU_Tuning_Summary.ipynb)：nnPU 调优
- [EndToEndPU/models/resnet_cn_attention.py](EndToEndPU/models/resnet_cn_attention.py)：模型定义
- [EndToEndPU/pu_loss.py](EndToEndPU/pu_loss.py)：PU 风险估计
- [EndToEndPU/trainer.py](EndToEndPU/trainer.py)：训练循环

### 4. 图半监督、流形与表示学习

ML/LabelSpreading/ 包含 KNN 图上的 Label Spreading、谱聚类、GNN 和 UMAP/KDE 实验；ML/SpectraAE/ 包含卷积自编码器、CN-aware 加权重构和 AE 特征上的 PU bagging。它们主要用于候选体交叉验证、可解释表示学习和后续方法探索。

### 5. 已验证的失败路线

- **DeepSVDD**：全局 latent-space center-distance 无法捕获微弱 CN 信号，AUROC=0.4495。
- **纯物理特征 MLP**：训练集与验证集 PR-AUC 差距约 0.62–0.73，严重过拟合。
- **仅依赖物理阈值**：召回率不足，无法覆盖 CN 星的形态多样性。

这些实验仍保留在 Notebook 中，因为它们构成研究结论的一部分，而不是需要被隐藏的“失败代码”。

## 数据与预处理

共享数据加载和特征计算集中在以下模块：

- [PhaseSummary/shared/data_loader.py](PhaseSummary/shared/data_loader.py)：数据加载、缓存检测、CN 分子带定义和批量指数计算
- [ML/utils.py](ML/utils.py)：14 维物理/光谱特征、cluster z-score、评价指标和 top-K 统计
- [build_dr13_all_cache.py](build_dr13_all_cache.py)：最新合并缓存的构建入口
- [Base/spectra.py](Base/spectra.py)：早期光谱读写与预处理工具

主要预处理步骤如下：

~~~mermaid
flowchart LR
    A[星表与 FITS 光谱] --> B[Schema 标准化]
    B --> C[天球匹配已知 CN 标签]
    C --> D[RV 修正与 700 px 插值]
    D --> E[连续谱归一化]
    E --> F[UID 去重与异常过滤]
    F --> G[物理特征 / PCA / masked-band 聚类]
    G --> H[PU Bagging / 图传播 / 深度模型]
    H --> I[候选体排序与交叉验证]
~~~

特征工程的核心思想不是简单比较全体光谱，而是在相似 Teff、logg、[Fe/H] 的恒星中估计 CN 基线，再分析 CN/CH 指数及其邻域偏差。常用特征包括：

~~~text
teff, logg, feh
CN3839, CN4142, CH4300
delta_CN3839, delta_CN4142, delta_CH4300
knn_center_euclid, knn_center_dist_z
pca_1, pca_2, pca_3
~~~

## 仓库结构

~~~text
Lamost/
├── Base/                    # 初期数据探索、基础光谱工具与早期 Notebook
├── ML/                      # 经典 ML、PU bagging、深度学习与图方法
│   ├── BinaryClassifier/    # Conv1D/MLP 二分类实验
│   ├── Deep/                # DeepSVDD 与深度 PU 实验
│   ├── LabelSpreading/      # 图传播、谱聚类、GNN、UMAP
│   └── SpectraAE/           # 自编码器与 CN-aware 表示学习
├── PhaseSummary/            # 按研究阶段整理的主分析入口
│   ├── 01_T_physics/
│   ├── 02_BinaryClassifier/
│   ├── 03_ML_XGB/
│   ├── 04_SpectraAE/
│   └── shared/
├── EndToEndPU/              # 1D ResNet + CN attention + nnPU
├── XGB/                     # XGBoost 候选体交叉实验
├── crossfit/                # 原始 CrossFit 引擎、Notebook 与测试
├── diagnosis/               # 新增标签诊断、加样本实验与阈值审计
├── feature/                 # CN 指数、差分面积与方法交叉核验
├── m2_crossfit/             # 严格 M2 OOF、候选导出与91星重新标记
├── build_dr13_all_cache.py  # 最新数据缓存构建脚本
└── README.md                # 项目总览与研究结论
~~~

仓库中不上传以下内容：Data/、*_cache/、checkpoints/、模型权重、FITS/CSV 数据、PDF 文献、PNG 图像和临时实验日志。这样可以避免把本地数据路径、二进制模型和大文件混入研究代码版本。

## 从哪里开始

如果只想快速了解目前的研究结论，建议按以下顺序阅读：

1. [m2_crossfit/M2_crossfit_full_pipeline.ipynb](m2_crossfit/M2_crossfit_full_pipeline.ipynb)：最新 M2 训练、严格 OOF 和候选导出。
2. [m2_crossfit/README.md](m2_crossfit/README.md)：M2 结果口径、字段说明和复现命令。
3. [EndToEndPU/EndToEndPU_Summary.ipynb](EndToEndPU/EndToEndPU_Summary.ipynb)：深度 PU 方法总结与对比。
4. [PhaseSummary/03_ML_XGB/ML_XGB_PU.ipynb](PhaseSummary/03_ML_XGB/ML_XGB_PU.ipynb)：历史 XGBoost PU 主实验。
5. [feature/CN_method_crosscheck.ipynb](feature/CN_method_crosscheck.ipynb)：候选光谱特征的交叉核验。

如果需要从代码开始：

~~~text
数据加载       → PhaseSummary/shared/data_loader.py
特征计算       → ML/utils.py
XGBoost PU     → ML/pu_bagging.py
深度模型       → EndToEndPU/models/resnet_cn_attention.py
PU 损失        → EndToEndPU/pu_loss.py
训练循环       → EndToEndPU/trainer.py
~~~

## 环境与复现

项目使用本地 Conda 环境 myenv。建议使用项目约定的解释器：

~~~powershell
D:/Anaconda/envs/myenv/python.exe --version
D:/Anaconda/envs/myenv/python.exe -m notebook
~~~

运行 Notebook 前，需要在本机准备对应的数据和缓存目录，并检查 PhaseSummary/shared/data_loader.py 或具体实验脚本中的路径探测逻辑。不同数据版本、正样本目录和随机种子会明显影响 PR-AUC 与 Precision@K，因此复现实验时应同时记录：

- 数据缓存版本与已知 CN 星数量
- 训练/验证/测试划分和随机种子
- PU bagging 轮数与伪负样本采样策略
- 候选体阈值的标定方式
- 是否使用 cluster z-score 或 CN-band attention

本仓库只保存可审阅的代码和 Notebook，不保证在没有本地数据缓存的环境中“一键运行”。

## 研究边界与下一步

当前结果仍然是候选体筛选研究，不等同于对未知 CN 星的最终天文确认。下一步优先级为：

1. 建立严格的独立验证集和跨目录交叉验证，减少已知标签泄漏。
2. 扩充可靠的 CN 正样本，缓解深度模型的小样本过拟合。
3. 将 CN band–continuum 对比结构显式编码到更轻量的网络中，再评估 nnPU 的稳定性。
4. 用 XGBoost、Label Spreading 和 EndToEndPU 的交集/差集分析建立候选体优先级，而不是只依赖单一模型。
5. 对高优先级候选体进行额外光谱质量检查和天文文献交叉匹配。

## 研究说明

本项目面向科研探索与方法比较。README 中的指标来自不同阶段的实验记录；除特别注明外，指标不应被视为同一测试集上的严格横向比较。最终候选体仍需结合独立数据、光谱质量和天文物理分析进行确认。
