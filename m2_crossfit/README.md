# M2 CrossFit 全流程

该目录实现 91 颗旧证认 CN 星、46 个新增正例、53 个可靠负例和其余 LAMOST DR13 未标注对象的 M2 CrossFit 训练、评估和候选导出。

## 主要入口

- `M2_crossfit_full_pipeline.ipynb`：已执行的完整训练 Notebook。
- `m2_crossfit_engine.py`：三类标签 CrossFit、严格新标签 OOF、全标签 ensemble 和导出逻辑。
- `results/m2_candidates_recall90.csv`：90% 正例召回阈值下的普通 U 候选。
- `results/known91_recognition_labels.csv`：91 颗旧证认星的模型识别标签。
- `results/m2_labelled_oof_scores.csv`：99 个新增高分辨率标签的严格 OOF 分数（46 正例、53 可靠负例）。
- `results/m2_all_scores.npz`：41,243 个对象的全量 OOF、重复训练、fold、状态及最终模型分数数组。

## 正式运行结果

以下结果来自 5 折、3 次重复、每折 100 个 PU bag、每个模型 50 轮 XGBoost 的正式执行版本：

| 分析环节 | ROC-AUC | PR-AUC | 说明 |
|---|---:|---:|---|
| 原 91 星模型对 99 个新增标签的独立审计 | 0.911 ± 0.031 | 0.876 | 冻结旧模型，复现既有结果 |
| 严格 M2 新标签 5 折 OOF | 0.939 ± 0.026 | 0.909 | 每个新增标签仅由未见过它的模型评分 |
| M2 全部 190 个标签的 OOF | 0.926 | 0.971 | 137 正例与 53 可靠负例的总体区分结果 |

- 90% 正例召回阈值：`0.2498818354`；137 个确认正例中保留 124 个，实际召回率 `90.51%`。
- 普通未标注样本候选：393 个；全部 137 个正例和 53 个可靠负例均已排除。
- 原 91 颗证认星：模型可识别 79 颗（`label=1`），漏检 12 颗（`label=0`）。
- 重复训练稳定性：3 次重复分别产生 488、395、421 个候选，对应阈值为 0.2035、0.2600、0.2349。

候选入选及 91 星分组均以无标签泄漏的 `m2_oof_score` 为准；`final_model_score` 只用于所有标签参与最终训练后的候选优先级补充排序。

## 标签定义

`known91_recognition_labels.csv` 中：

- `ground_truth_label=1`：该对象是已证认 CN 星；
- `label=1`：M2 OOF score 达到 90% 正例召回阈值；
- `label=0`：M2 OOF score 低于该阈值，属于模型漏检对象。

模型 label 只描述当前模型在固定阈值下能否识别，不修改天文真实标签。

## 运行

生成 Notebook：

```powershell
& 'D:\Anaconda\envs\myenv\python.exe' m2_crossfit\build_m2_notebook.py
```

Smoke 执行：

```powershell
$env:M2_SMOKE='1'
& 'D:\Anaconda\envs\myenv\python.exe' -m jupyter nbconvert --to notebook --execute m2_crossfit\M2_crossfit_full_pipeline.ipynb --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=myenv
Remove-Item Env:M2_SMOKE
```

正式执行：

```powershell
& 'D:\Anaconda\envs\myenv\python.exe' -m jupyter nbconvert --to notebook --execute m2_crossfit\M2_crossfit_full_pipeline.ipynb --inplace --ExecutePreprocessor.timeout=10800 --ExecutePreprocessor.kernel_name=myenv
```

测试：

```powershell
& 'D:\Anaconda\envs\myenv\python.exe' -m unittest m2_crossfit.test_m2_crossfit_engine -v
```

## 结果解释

- `m2_oof_score`：候选入选和 91 星识别标签的权威分数。
- `final_model_score`：全部 137 正例和 53 可靠负例参与训练后的补充排序分数。
- `m2_recall90_threshold`：使 137 个确认正例中至少 124 个被保留的 OOF 顺序阈值。
- 这些分数来自人为构造的 PU 采样比例，不是校准后的巡天后验概率。

## 严格 OOF 修正

旧的加样本实验将 held-out 新标签保留在普通 U 池中，存在被抽为伪负样本的可能。本目录的严格 OOF 评估会把全部 held-out 新标签从训练正例、可靠负例和普通 U 池同时排除。
