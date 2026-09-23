"""Build the fully annotated M2 cross-fit training notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "M2_crossfit_full_pipeline.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        r"""
# M2 CrossFit：新增高分辨率标签训练、外部评估与候选导出

本 Notebook 从项目缓存开始执行完整流程，输出两个核心结果：

1. 用原 91 颗证认 CN 星训练原始 crossfit，在 99 个新增高分辨率标签上复现加入前的外部评估；
2. 将 46 个新增正例和 53 个可靠负例纳入 M2，使用严格 OOF 分数定义 90% 正例召回阈值，并导出候选表与 91 星识别表。

## 指标解释

- **0.911 ± 0.031 / 0.876**：原 91 星 crossfit 对 99 个新增标签的外部 ROC-AUC（± Hanley-McNeil 标准误）和 PR-AUC。
- **M2 新标签 OOF 指标**：99 个新标签做 5 折；被评分对象不出现在训练正例、可靠负例或 U 采样池中。
- **90% 阈值**：137 个确认正例的 OOF 分数中第 `ceil(137×0.90)=124` 高分。它是固定召回阈值，不是 90% 后验概率。
- **候选入选和 91 星 label**：只使用 OOF 分数。全标签 ensemble 分数仅作为补充排序列。
"""
    ),
    md("## 0. 环境、路径与配置"),
    code(
        r"""
from pathlib import Path
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    RocCurveDisplay, PrecisionRecallDisplay,
    roc_curve, precision_recall_curve,
)

def find_project_root(start):
    current = Path(start).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "build_dr13_all_cache.py").exists():
            return candidate
    raise FileNotFoundError("cannot locate project root")

PROJECT_ROOT = find_project_root(Path.cwd())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_dr13_all_cache import load_dr13_all_cache
from crossfit import crossfit_engine as baseline_engine
from m2_crossfit.m2_crossfit_engine import (
    M2Config, build_status, build_output_tables, evaluate_scores,
    map_new_labels, prepare_features, run_m2_crossfit,
    run_newlabel_oof_m2, save_outputs, train_full_ensemble,
)

HERE = PROJECT_ROOT / "m2_crossfit"
RESULTS_DIR = HERE / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SMOKE = os.environ.get("M2_SMOKE", "0") == "1"
if SMOKE:
    baseline_config = baseline_engine.CrossFitConfig(
        u_to_p_ratio=2, n_splits=3, n_repeats=1, n_bags=2,
        num_boost_round=3, target_recall=0.90, random_seed=42,
    )
    m2_config = M2Config(
        u_to_p_ratio=2, n_splits=3, n_repeats=1, n_bags=2,
        num_boost_round=3, target_recall=0.90, random_seed=42,
    )
    newlabel_config = M2Config(
        u_to_p_ratio=2, n_splits=3, n_repeats=1, n_bags=2,
        num_boost_round=3, target_recall=0.90, random_seed=42,
    )
else:
    baseline_config = baseline_engine.CrossFitConfig(
        u_to_p_ratio=10, max_depth=3, min_child_weight=1, reg_lambda=1.0,
        n_splits=5, n_repeats=3, n_bags=100, num_boost_round=50,
        target_recall=0.90, random_seed=42, standardize=True,
    )
    m2_config = M2Config(
        u_to_p_ratio=10, max_depth=3, min_child_weight=1, reg_lambda=1.0,
        n_splits=5, n_repeats=3, n_bags=100, num_boost_round=50,
        target_recall=0.90, random_seed=42, standardize=True,
    )
    # 与已有 add-sample 正式实验保持 80 bags，但修正 held-out 标签泄漏。
    newlabel_config = M2Config(
        u_to_p_ratio=10, max_depth=3, min_child_weight=1, reg_lambda=1.0,
        n_splits=5, n_repeats=1, n_bags=80, num_boost_round=50,
        target_recall=0.90, random_seed=42, standardize=True,
    )

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})
print("Python       :", sys.executable)
print("Project root :", PROJECT_ROOT)
print("Output       :", RESULTS_DIR)
print("Smoke mode   :", SMOKE)
print("Baseline cfg :", baseline_config)
print("M2 cfg       :", m2_config)
"""
    ),
    md(
        r"""
## 1. 加载数据并冻结标签身份

这里进行 UID 级校验。新增标签必须满足：99 个 UID 唯一、全部能映射到 41,243 条母样本、标签只含 0/1、与原 91 颗正例零重叠。
"""
    ),
    code(
        r"""
data = load_dr13_all_cache()
stars = pd.DataFrame(data["stars_clean"]).reset_index(drop=True)
X_model = prepare_features(data["X_clean"], standardize=True)
wave = np.asarray(data["common_wave"])

highres = pd.read_csv(PROJECT_ROOT / "diagnosis" / "cands_label.csv")
mapping = map_new_labels(stars, highres)
old_positive = np.asarray(mapping["old_positive"], dtype=int)
new_positive = np.asarray(mapping["new_positive"], dtype=int)
new_negative = np.asarray(mapping["new_negative"], dtype=int)
status = build_status(len(stars), old_positive, new_positive, new_negative)

print(f"母样本: {len(stars):,}")
print(f"旧正例: {old_positive.size}")
print(f"新增正例: {new_positive.size}")
print(f"可靠负例: {new_negative.size}")
print(f"普通 U: {(status == 0).sum():,}")
assert len(stars) == 41243
assert old_positive.size == 91
assert new_positive.size == 46
assert new_negative.size == 53
assert np.unique(np.concatenate([old_positive, new_positive, new_negative])).size == 190
"""
    ),
    md(
        r"""
## 2. 原始 91 星 crossfit：复现 99 标签外部指标

这一段重新训练原始 crossfit。99 个高分辨率标签不作为正例或可靠负例；它们只在各自 OOF 分数生成后用于外部评估。
"""
    ),
    code(
        r"""
baseline_started = time.perf_counter()
baseline_y = baseline_engine.label_array(stars)
baseline_result = baseline_engine.run_crossfit_pu(
    X_model, baseline_y, baseline_config, progress=True
)
new_label_table = mapping["labels"].copy()
new_rows = new_label_table["source_index"].to_numpy(dtype=int)
new_y = new_label_table["label"].to_numpy(dtype=int)
baseline_new_scores = np.asarray(baseline_result["score_mean"])[new_rows]
baseline_audit = evaluate_scores(new_y, baseline_new_scores)
pd.DataFrame(baseline_result["history"]).to_csv(
    RESULTS_DIR / "baseline_training_history.csv", index=False
)
print(json.dumps(baseline_audit, indent=2, ensure_ascii=False))
print(f"baseline elapsed: {time.perf_counter() - baseline_started:.1f}s")
if not SMOKE:
    assert abs(baseline_audit["roc_auc"] - 0.9109926168990976) < 1e-10
    assert abs(baseline_audit["pr_auc"] - 0.8763481932761269) < 1e-10
"""
    ),
    md(
        r"""
### 2.1 基线 ROC、PR 与分数分布

ROC-AUC 描述排序能力；PR-AUC 更关注稀有正类。标准误只附在 ROC-AUC 上，PR-AUC 不使用相同的解析标准误公式。
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
RocCurveDisplay.from_predictions(new_y, baseline_new_scores, ax=axes[0], color="#185FA5")
axes[0].set_title(
    f"原91星 crossfit：99标签 ROC\n"
    f"AUC={baseline_audit['roc_auc']:.3f} ± {baseline_audit['roc_auc_se']:.3f}"
)
PrecisionRecallDisplay.from_predictions(
    new_y, baseline_new_scores, ax=axes[1], color="#1D9E75"
)
axes[1].set_title(f"99标签 PR\nPR-AUC={baseline_audit['pr_auc']:.3f}")
bins = np.linspace(0, 1, 35)
axes[2].hist(baseline_new_scores[new_y == 0], bins=bins, alpha=.65,
             label="确认非增丰 n=53", color="#BA7517")
axes[2].hist(baseline_new_scores[new_y == 1], bins=bins, alpha=.65,
             label="确认增丰 n=46", color="#E24B4A")
axes[2].set_xlabel("baseline OOF score")
axes[2].set_ylabel("count")
axes[2].set_title("新增标签分数分布")
axes[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "01_baseline_external_audit.png", dpi=170, bbox_inches="tight")
plt.show()
"""
    ),
    md(
        r"""
## 3. 严格 M2：99 个新增标签 5 折 OOF

旧实验把 held-out 新标签保留在普通 U 池中，因此它们可能被抽成伪负样本。本流程把所有 99 个新标签从普通 U 池移除；每折只允许其余 4 折的新标签进入正例或可靠负例训练集合。
"""
    ),
    code(
        r"""
newlabel_oof = run_newlabel_oof_m2(
    X_model,
    old_positive,
    new_positive,
    new_negative,
    newlabel_config,
    progress=True,
)
newlabel_audit = evaluate_scores(
    np.asarray(newlabel_oof["labels"]), np.asarray(newlabel_oof["oof_score"])
)
newlabel_oof["fold_metrics"].to_csv(
    RESULTS_DIR / "m2_newlabel_fold_metrics.csv", index=False
)
print(json.dumps(newlabel_audit, indent=2, ensure_ascii=False))
display(newlabel_oof["fold_metrics"])
"""
    ),
    code(
        r"""
m2_new_y = np.asarray(newlabel_oof["labels"])
m2_new_score = np.asarray(newlabel_oof["oof_score"])
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
RocCurveDisplay.from_predictions(m2_new_y, m2_new_score, ax=axes[0], color="#534AB7")
axes[0].set_title(
    f"严格M2：99标签 OOF ROC\n"
    f"AUC={newlabel_audit['roc_auc']:.3f} ± {newlabel_audit['roc_auc_se']:.3f}"
)
PrecisionRecallDisplay.from_predictions(
    m2_new_y, m2_new_score, ax=axes[1], color="#1D9E75"
)
axes[1].set_title(f"严格M2 OOF PR\nPR-AUC={newlabel_audit['pr_auc']:.3f}")
axes[2].scatter(baseline_new_scores, m2_new_score,
                c=np.where(new_y == 1, "#E24B4A", "#BA7517"), alpha=.8)
axes[2].plot([0, 1], [0, 1], color="gray", ls=":")
axes[2].set_xlabel("原91星 baseline OOF score")
axes[2].set_ylabel("严格 M2 OOF score")
axes[2].set_title("加入新标签前后分数变化")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "02_strict_m2_newlabel_oof.png", dpi=170, bbox_inches="tight")
plt.show()
"""
    ),
    md(
        r"""
## 4. 全样本 M2 CrossFit

137 个确认正例、53 个可靠负例和 41,053 个普通 U 一起进入三类分层 crossfit。可靠负例在训练折中被强制加入每个 bag；holdout 中的任何对象都不会进入该模型训练。
"""
    ),
    code(
        r"""
m2_result = run_m2_crossfit(X_model, status, m2_config, progress=True)
print("labelled OOF ROC-AUC:", m2_result["labelled_roc_auc"])
print("labelled OOF PR-AUC :", m2_result["labelled_pr_auc"])
print("recall90 threshold  :", m2_result["threshold"])
print("required positives  :", m2_result["required_positive"])
print("achieved recall     :", m2_result["achieved_positive_recall"])
print("repeat thresholds   :", np.asarray(m2_result["repeat_thresholds"]).round(6))
print("repeat candidates   :", np.asarray(m2_result["repeat_candidate_counts"]))
assert int(m2_result["required_positive"]) == 124
assert float(m2_result["achieved_positive_recall"]) >= 0.90
"""
    ),
    md(
        r"""
### 4.1 OOF 分数、顺序阈值与重复稳定性

阈值只由确认正例的 OOF 分数决定。候选数量的重复波动反映 137 个正例对决策边界的稳定程度。
"""
    ),
    code(
        r"""
score = np.asarray(m2_result["score_mean"])
threshold = float(m2_result["threshold"])
pos_score = score[status == 1]
neg_score = score[status == -1]
u_score = score[status == 0]
ordered = np.sort(pos_score)[::-1]

fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
axes[0].hist(u_score, bins=90, color="gray", alpha=.65, label=f"U n={len(u_score):,}")
axes[0].hist(pos_score, bins=35, color="#E24B4A", alpha=.75, label="确认正例 n=137")
axes[0].hist(neg_score, bins=25, color="#BA7517", alpha=.75, label="可靠负例 n=53")
axes[0].axvline(threshold, color="black", ls="--", label=f"threshold={threshold:.4f}")
axes[0].set_xlabel("M2 OOF score")
axes[0].set_ylabel("count")
axes[0].set_title("全样本 OOF 分数")
axes[0].legend(fontsize=8)

axes[1].plot(np.arange(1, len(ordered)+1), ordered, marker=".", lw=1)
axes[1].axvline(124, color="#E24B4A", ls="--", label="第124高分")
axes[1].axhline(threshold, color="black", ls=":")
axes[1].set_xlabel("确认正例降序排名")
axes[1].set_ylabel("M2 OOF score")
axes[1].set_title("90% 正例召回顺序统计量")
axes[1].legend()

repeat_id = np.arange(1, m2_config.n_repeats + 1)
axes[2].bar(repeat_id - .16, m2_result["repeat_candidate_counts"], width=.32,
            color="#185FA5", label="候选数量")
ax2 = axes[2].twinx()
ax2.plot(repeat_id, m2_result["repeat_thresholds"], marker="o", color="#E24B4A",
         label="阈值")
axes[2].set_xlabel("repeat")
axes[2].set_ylabel("候选数量")
ax2.set_ylabel("阈值")
axes[2].set_title("重复间稳定性")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "03_m2_threshold_and_stability.png", dpi=170, bbox_inches="tight")
plt.show()
"""
    ),
    md(
        r"""
## 5. 使用全部标签训练最终 ensemble

该 ensemble 使用全部 137 个正例和 53 个可靠负例。它的 `final_model_score` 可用于候选内部的补充排序；候选是否入选仍由上一节的 OOF score 和 OOF 阈值决定。
"""
    ),
    code(
        r"""
final_score = train_full_ensemble(X_model, status, m2_config, progress=True)
print("final score range:", float(final_score.min()), float(final_score.max()))
"""
    ),
    md("## 6. 导出候选表和 91 星识别标签表"),
    code(
        r"""
paths = save_outputs(
    RESULTS_DIR,
    stars,
    mapping,
    status,
    m2_result,
    final_score,
    baseline_audit,
    newlabel_oof,
)
tables = build_output_tables(stars, old_positive, status, m2_result, final_score)
candidates = tables["candidates"]
known91 = tables["known91"]

print("候选数:", len(candidates))
print("91星可识别:", int(known91["label"].sum()))
print("91星漏检:", int((known91["label"] == 0).sum()))
display(known91.head(20))
display(candidates.head(20))

assert len(known91) == 91
assert set(known91["label"].unique()).issubset({0, 1})
assert (candidates["m2_status"] == 0).all()
assert not set(candidates["uid"].astype(str)) & set(
    stars.iloc[np.flatnonzero(status != 0)]["uid"].astype(str)
)
for name, path in paths.items():
    print(f"{name:12s}: {path}")
"""
    ),
    md(
        r"""
### 6.1 91 颗旧证认星：识别与漏检特征

导出表中的 `ground_truth_label` 恒为 1；新 `label` 表示模型在 M2 90% 正例召回阈值下能否识别。这里重点检查温度、金属丰度、重力和 SNR 是否与漏检有关。
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, column, label_text in zip(
    axes.ravel(),
    ["teff", "logg", "feh", "snru"],
    ["Teff", "log g", "[Fe/H]", "SNRu"],
):
    recognized = known91.loc[known91["label"] == 1, column].dropna()
    missed = known91.loc[known91["label"] == 0, column].dropna()
    ax.hist(recognized, bins=18, alpha=.65, color="#1D9E75", label=f"识别 n={len(recognized)}")
    ax.hist(missed, bins=10, alpha=.8, color="#E24B4A", label=f"漏检 n={len(missed)}")
    ax.set_xlabel(label_text)
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
fig.suptitle("91颗旧证认星：识别与漏检参数分布")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "04_known91_recognition_parameters.png", dpi=170, bbox_inches="tight")
plt.show()

missed91 = known91.loc[known91["label"] == 0, [
    "uid", "ra", "dec", "teff", "logg", "feh", "snru",
    "m2_oof_score", "m2_repeat_std", "m2_selection_frequency",
    "positive_rank", "recognition_status"
]]
display(missed91)
"""
    ),
    md("### 6.2 候选参数分布"),
    code(
        r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
u_mask = status == 0
for ax, column, text_label in zip(
    axes,
    ["teff", "logg", "feh"],
    ["Teff", "log g", "[Fe/H]"],
):
    ax.hist(stars.loc[u_mask, column], bins=60, density=True, alpha=.35,
            color="gray", label="普通 U")
    ax.hist(candidates[column], bins=35, density=True, alpha=.70,
            color="#185FA5", label=f"候选 n={len(candidates):,}")
    ax.hist(stars.iloc[np.flatnonzero(status == 1)][column], bins=25,
            density=True, histtype="step", lw=2, color="#E24B4A", label="确认正例")
    ax.set_xlabel(text_label)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
fig.suptitle("M2 recall90 候选与母样本参数分布")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "05_candidate_parameters.png", dpi=170, bbox_inches="tight")
plt.show()
"""
    ),
    md("### 6.3 最高分候选光谱"),
    code(
        r"""
top = candidates.sort_values("m2_oof_score", ascending=False).head(20).reset_index(drop=True)
fig, axes = plt.subplots(4, 5, figsize=(18, 11), sharex=True)
for ax, (_, row) in zip(axes.ravel(), top.iterrows()):
    idx = int(row["source_index"])
    ax.plot(wave, np.asarray(data["X_clean"])[idx], color="#185FA5", lw=.75)
    for left, right, colour, name in [
        (3830, 3883, "royalblue", "CN3839"),
        (4120, 4216, "green", "CN4142"),
        (4285, 4315, "red", "CH4300"),
    ]:
        ax.axvspan(left, right, color=colour, alpha=.09)
    ax.set_title(
        f"#{int(row['candidate_rank'])} {row['uid']}\n"
        f"OOF={row['m2_oof_score']:.3f}, final={row['final_model_score']:.3f}\n"
        f"T={row['teff']:.0f}, logg={row['logg']:.2f}, FeH={row['feh']:.2f}",
        fontsize=7,
    )
    ax.set_xlim(3800, 4500)
fig.supxlabel("Wavelength (Angstrom)")
fig.supylabel("Normalized flux")
fig.suptitle("M2 recall90 最高分候选光谱", y=1.01)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "06_top20_candidate_spectra.png", dpi=170, bbox_inches="tight")
plt.show()
"""
    ),
    md(
        r"""
## 7. 结果摘要与使用限制

1. `m2_oof_score` 是候选入选的权威分数；`final_model_score` 只用于候选内部补充排序。
2. 90% 指已确认正例的 OOF 召回，不是模型后验概率或候选真实性置信度。
3. 91 星表中的 `label=0` 表示模型在本阈值下漏检，不改变其真实的证认星身份。
4. 99 个新增标签来自追随观测，M2 的 OOF 指标描述该标签群体，不能直接外推为整个 DR13 的总体精度。
5. 下一批二次高分结果应作为新的独立外部测试集，避免继续用同一批 99 个标签调参与评估。
"""
    ),
    code(
        r"""
summary = json.loads((RESULTS_DIR / "m2_summary.json").read_text(encoding="utf-8"))
display(pd.json_normalize(summary, sep="."))
print("结果目录:", RESULTS_DIR.resolve())
print("图像目录:", FIGURE_DIR.resolve())
"""
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "myenv",
            "language": "python",
            "name": "myenv",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
