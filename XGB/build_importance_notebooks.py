"""Build the two importance-sampling experiment notebooks.

Generated notebooks:
    - ML_XGB_PU_matched.ipynb   # experiment 1: 3D-binned stratified nearest-neighbor sampling
    - ML_XGB_PU_kde.ipynb       # experiment 2: KDE density-ratio weighted sampling

Both reuse the shared module XGB/importance_sampling.py and compare against the
uniform-sampling baseline in the same notebook (same split, seed=42).

Run: D:/Anaconda/envs/myenv/python.exe XGB/build_importance_notebooks.py
"""

import nbformat as nbf
from pathlib import Path

BASE = Path(__file__).resolve().parent  # XGB/
PROJECT_ROOT = BASE.parent              # Lamost/


def nb():
    return nbf.v4.new_notebook(metadata={
        "kernelspec": {"display_name": "myenv", "language": "python", "name": "myenv"},
        "language_info": {"name": "python", "version": "3.12.0"},
    })


def md(src):
    return nbf.v4.new_markdown_cell(src)


def code(src):
    return nbf.v4.new_code_cell(src)


# ═══════════════════════════════════════════════════════════════════════
# 共享单元格
# ═══════════════════════════════════════════════════════════════════════

SHARED_IMP = '''# 共享数据加载 + 导入重要性采样模块

import sys, os, time
from pathlib import Path

# 定位项目根目录（向上查找直到同时存在 ML/ 与 Data/）
_PROJECT_ROOT = Path(os.getcwd())
for _ in range(5):
    if (_PROJECT_ROOT / "ML").exists() and (_PROJECT_ROOT / "Data").exists():
        break
    _PROJECT_ROOT = _PROJECT_ROOT.parent
for p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "XGB")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 10})
import warnings
warnings.filterwarnings("ignore")

# 核心逻辑（采样器 + PU Bagging + 评估）统一在 XGB/importance_sampling.py
import importance_sampling as imp

t0 = time.time()
S = imp.load_and_split()
X_clean = S["X_clean"]
stars_clean = S["stars_clean"]
common_wave = S["common_wave"]
X_all = S["X_spec_scaled"]
X_tr = S["X_tr"]
X_te = S["X_te"]
y_tr = S["y_tr"]
y_te = S["y_te"]
n_pos_tr = S["n_pos_tr"]
unl_tr_idx = S["unl_tr_idx"]
tr_idx = S["tr_idx"]

feh_all = stars_clean["feh"].values.astype(float)

print(f"数据加载完成 ({time.time()-t0:.0f}s)")
print(f"总样本 {len(S['y_all']):,} | 训练 {len(tr_idx):,} (P={n_pos_tr}, U={len(unl_tr_idx):,}) | 测试 {len(S['test_idx']):,}")
print(f"物理参数列: teff/logg/feh（全样本无缺失）")
'''

RUN_BOTH = '''# 运行 PU Bagging（重要性采样 vs 标准等概率采样）

T = 500

print(f"[{name}] {label} (T={T}) ...")
te_method, p_method = imp.run_pu_bagging(
    X_tr, X_te, X_all, y_tr, y_te, n_pos_tr, method_sampler, T,
    label=name, report_every=100)

print()
print("[base] 标准 PU 等概率采样 (T=500) ...")
te_base, p_base = imp.run_pu_bagging(
    X_tr, X_te, X_all, y_tr, y_te, n_pos_tr, imp.UniformSampler(unl_tr_idx), T,
    label="base", report_every=100)

res_method = imp.metrics(y_te, te_method)
res_base = imp.metrics(y_te, te_base)

print()
print(f"{label:28s}  ROC={res_method['roc']:.4f}  PR={res_method['pr']:.4f}  "
      f"P@50={res_method['p50']:.4f}  P@100={res_method['p100']:.4f}")
print(f"{'标准 PU(等概率)':28s}  ROC={res_base['roc']:.4f}  PR={res_base['pr']:.4f}  "
      f"P@50={res_base['p50']:.4f}  P@100={res_base['p100']:.4f}")
'''

COMPARE = '''# 结果对比：指标表 + PR/ROC 曲线叠加

from sklearn.metrics import precision_recall_curve, roc_curve, auc

comp = pd.DataFrame({
    "采样方式": ["标准 PU (等概率)", label],
    "ROC-AUC": [res_base["roc"], res_method["roc"]],
    "PR-AUC": [res_base["pr"], res_method["pr"]],
    "P@50": [res_base["p50"], res_method["p50"]],
    "P@100": [res_base["p100"], res_method["p100"]],
})
print(comp.to_string(index=False))

base_rate = y_te.sum() / len(y_te)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

pr_base, rc_base, _ = precision_recall_curve(y_te, te_base)
pr_m, rc_m, _ = precision_recall_curve(y_te, te_method)
ax1 = axes[0]
ax1.plot(rc_base, pr_base, "b-", linewidth=2, label=f"标准 PU (AP={res_base['pr']:.3f})")
ax1.plot(rc_m, pr_m, "darkorange", linewidth=2, label=f"{label} (AP={res_method['pr']:.3f})")
ax1.axhline(base_rate, color="gray", linestyle="--", linewidth=1.0, label=f"Random ({base_rate:.3f})")
ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision")
ax1.set_title("Precision-Recall Curve")
ax1.legend(fontsize=9); ax1.grid(alpha=0.2)
ax1.set_xlim(0, 1.02); ax1.set_ylim(0, 1.02)

fpr_base, tpr_base, _ = roc_curve(y_te, te_base)
fpr_m, tpr_m, _ = roc_curve(y_te, te_method)
ax2 = axes[1]
ax2.plot(fpr_base, tpr_base, "darkred", linewidth=2, label=f"标准 PU (AUC={auc(fpr_base, tpr_base):.3f})")
ax2.plot(fpr_m, tpr_m, "darkorange", linewidth=2, label=f"{label} (AUC={auc(fpr_m, tpr_m):.3f})")
ax2.plot([0, 1], [0, 1], "gray", linestyle="--", linewidth=1.0)
ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve")
ax2.legend(fontsize=9); ax2.grid(alpha=0.2)
ax2.set_xlim(0, 1.02); ax2.set_ylim(0, 1.02)

fig.suptitle("标准 PU vs " + label, fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
'''

THRESHOLD_EXPORT = '''# 已知 CN 星标定阈值 + 候选体 [Fe/H] 分解

recall_quantile = 0.05
thr_method, ncand_method, nknown_method = imp.threshold_candidates(stars_clean, p_method, recall_quantile)
thr_base, ncand_base, nknown_base = imp.threshold_candidates(stars_clean, p_base, recall_quantile)

def feh_split(prob, thr):
    cand = (stars_clean["label"].values == -1) & (prob >= thr)
    f = feh_all[cand]
    return dict(n=int(cand.sum()), gt=int((f > -1.2).sum()), le=int((f <= -1.2).sum()),
                median=float(np.median(f)))

fs_method = feh_split(p_method, thr_method)
fs_base = feh_split(p_base, thr_base)

print(f"标定阈值 (保留~{(1-recall_quantile)*100:.0f}% 已知 CN 星):")
print(f"  标准 PU(等概率): thr={thr_base:.4f}  候选={ncand_base}  已知CN>=thr={nknown_base}")
print(f"  {label}: thr={thr_method:.4f}  候选={ncand_method}  已知CN>=thr={nknown_method}")
print()
print("候选体 [Fe/H] 分解:")
for nm, fs in [("标准 PU(等概率)", fs_base), (label, fs_method)]:
    print(f"  {nm:22s} n={fs['n']:5d}  feh>-1.2: {fs['gt']:5d}  feh<=-1.2: {fs['le']:5d}  中位feh={fs['median']:.3f}")

outpath = str(_PROJECT_ROOT / "XGB" / "@@FNAME@@")
n_exp = imp.export_candidates(stars_clean, p_method, thr_method, outpath)
print(f"\\n候选体已导出: {outpath}  ->  {n_exp} 颗")
'''


# ═══════════════════════════════════════════════════════════════════════
# 实验 1：三维分箱分层匹配近邻采样
# ═══════════════════════════════════════════════════════════════════════

PREP_MATCHED = '''# 标准化物理参数 + 构造"三维分箱分层匹配近邻"采样器

name = "matched"
label = "三维分箱分层匹配近邻"

B, K = 5, 10   # 每维分位箱数 / 同箱内最近邻个数

phys_all, mu, sd = imp.standardize_physics(stars_clean)
phys_tr = phys_all[tr_idx]
phys_pos = phys_tr[np.where(y_tr == 1)[0]]   # 训练正样本 (n_pos_tr, 3)
phys_unl = phys_tr[unl_tr_idx]               # 训练 U (n_unl, 3)

cands, n_cell = imp.build_match_candidates(phys_pos, phys_unl, unl_tr_idx, B=B, K=K)
method_sampler = imp.MatchSampler(cands, imp.random_seed)

# 有效负采样池 = 所有正样本候选的并集（用于诊断其物理参数分布）
match_pool_idx = np.unique(np.concatenate([np.atleast_1d(c) for c in cands]))
match_sizes = np.array([len(c) for c in cands])

print(f"物理参数标准化: mu(teff/logg/feh) = {mu.round(1)}, sd = {sd.round(2)}")
print(f"分箱 B={B}（3D 共 {B**3} 箱）, 同箱近邻 K={K}")
print(f"分层命中（仅靠同箱即取满 K 个近邻）: {n_cell}/{len(cands)}")
print(f"每正样本候选池大小: min={match_sizes.min()}  med={np.median(match_sizes):.0f}  max={match_sizes.max()}")
print(f"有效负采样池大小（候选并集）: {len(match_pool_idx):,}  (U 池共 {len(unl_tr_idx):,})")
'''

DIAG_MATCHED = '''# 诊断：匹配采样是否把负样本的 [Fe/H] 分布"拉向"正样本

feh_pos = feh_all[tr_idx[np.where(y_tr == 1)[0]]]
feh_unl = feh_all[tr_idx[unl_tr_idx]]
feh_match = feh_all[tr_idx[match_pool_idx]]   # 匹配负采样池的 feh

bins = np.linspace(-2.6, -0.6, 41)
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.hist(feh_unl, bins=bins, density=True, alpha=0.45, color="steelblue",
        label=f"U 池（等概率采样, n={len(feh_unl):,}）")
ax.hist(feh_match, bins=bins, density=True, alpha=0.65, color="darkorange",
        label=f"匹配负采样池（分箱+近邻, n={len(feh_match):,}）")
ax.hist(feh_pos, bins=bins, density=True, alpha=0.8, color="crimson",
        label=f"已知 CN 正样本（n={len(feh_pos)}）")
ax.axvline(-1.2, color="k", linestyle="--", linewidth=1.0, label="[Fe/H] = -1.2")
ax.set_xlabel("[Fe/H]")
ax.set_ylabel("密度")
ax.set_title("物理参数对齐：匹配采样后负样本 [Fe/H] 分布 vs 正样本")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()

for nm, f in [("U 池(等概率)", feh_unl), ("匹配负采样池", feh_match), ("正样本", feh_pos)]:
    print(f"{nm:16s} 中位 feh={np.median(f):.3f}  feh>-1.2 占比={np.mean(f > -1.2)*100:.1f}%")
'''


# ═══════════════════════════════════════════════════════════════════════
# 实验 2：KDE 核密度加权抽样
# ═══════════════════════════════════════════════════════════════════════

PREP_KDE = '''# 标准化物理参数 + 构造"KDE 核密度加权"采样器

name = "kde"
label = "KDE 核密度加权"

phys_all, mu, sd = imp.standardize_physics(stars_clean)
phys_tr = phys_all[tr_idx]
phys_pos = phys_tr[np.where(y_tr == 1)[0]]   # 训练正样本 (n_pos_tr, 3)
phys_unl = phys_tr[unl_tr_idx]               # 训练 U (n_unl, 3)

w, ess = imp.build_kde_weights(phys_pos, phys_unl, unl_tr_idx)
method_sampler = imp.WeightedSampler(unl_tr_idx, w, imp.random_seed)

print(f"物理参数标准化: mu(teff/logg/feh) = {mu.round(1)}, sd = {sd.round(2)}")
print(f"KDE 权重: max/median = {np.max(w)/np.median(w):.1f}  有效样本量 ESS = {ess:.0f} / {len(unl_tr_idx):,}")
print(f"权重集中度说明: ESS 越小，加权采样越'集中'在少数 U 上（物理分布越接近正样本）")
'''

DIAG_KDE = '''# 诊断：KDE 加权是否把负样本的 [Fe/H] 分布"拉向"正样本

feh_pos = feh_all[tr_idx[np.where(y_tr == 1)[0]]]
feh_unl = feh_all[tr_idx[unl_tr_idx]]

bins = np.linspace(-2.6, -0.6, 41)
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.hist(feh_unl, bins=bins, density=True, alpha=0.45, color="steelblue",
        label=f"U 池（等概率采样, n={len(feh_unl):,}）")
ax.hist(feh_unl, bins=bins, weights=w, density=True, alpha=0.65, color="darkorange",
        label="KDE 加权后（密度比重加权）")
ax.hist(feh_pos, bins=bins, density=True, alpha=0.8, color="crimson",
        label=f"已知 CN 正样本（n={len(feh_pos)}）")
ax.axvline(-1.2, color="k", linestyle="--", linewidth=1.0, label="[Fe/H] = -1.2")
ax.set_xlabel("[Fe/H]")
ax.set_ylabel("密度")
ax.set_title("物理参数对齐：KDE 加权后负样本 [Fe/H] 分布 vs 正样本")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()

print(f"U 池(等概率)       中位 feh={np.median(feh_unl):.3f}  feh>-1.2 占比={np.mean(feh_unl > -1.2)*100:.1f}%")
print(f"KDE 加权(有效分布) 中位 feh={np.average(feh_unl, weights=w):.3f}  feh>-1.2 占比={np.sum(w[feh_unl > -1.2])*100:.1f}%")
print(f"正样本             中位 feh={np.median(feh_pos):.3f}  feh>-1.2 占比={np.mean(feh_pos > -1.2)*100:.1f}%")
'''


def build_matched():
    n = nb()
    n.cells = [
        md("""# XGBoost PU Bagging — 三维分箱分层匹配近邻采样（重要性采样 I）

**实验动机：** 现有结果显示，[Fe/H] > -1.2 的候选星中有相当一部分 N 丰度较低的"误判"，而 [Fe/H] 更贫的部分识别更好。这说明标准 PU 的**等概率负采样**可能让模型学会了用物理参数（尤其 [Fe/H]）区分正负的"捷径"，而非真正依赖 CN 分子带形态。

**方法：**
1. 在 (teff, logg, feh) 标准化空间做 **3D 分位分箱**（每维 5 箱，共 125 箱，分层）
2. 每个正样本只在**同箱**内取 K=10 个**最近邻** U 作为负样本候选（分箱分层 + 近邻匹配）
3. 每轮 bag 每个正样本从自己的候选池采 1 个负样本 → **1:1 物理参数对齐**

**预期：** 负样本的物理参数分布被"拉向"正样本后，模型无法再用 [Fe/H] 偷懒，只能靠 CN 分子带区分；若识别性能变化显著，说明原结果确实受物理参数混淆影响。"""),

        code(SHARED_IMP),

        md("""## 1. 构造匹配采样器

标准 PU 从全部 U 中等概率采样；本实验把负样本限制为"与正样本物理参数最接近"的 U。分箱保证每个正样本的负样本落在同一物理参数区域（分层），箱内再取最近邻（近邻匹配），实现严格的物理参数对齐。"""),

        code(PREP_MATCHED),

        md("""## 2. 采样分布诊断（[Fe/H]）

关键检验：匹配采样是否把负样本的 [Fe/H] 分布拉向正样本分布。若橙色曲线明显向红色（正样本）靠拢，说明物理参数对齐生效。"""),

        code(DIAG_MATCHED),

        md("""## 3. 运行 PU Bagging（T=500）

在匹配采样器与标准等概率采样器上分别跑 T=500，其余完全一致。"""),

        code(RUN_BOTH),

        md("""## 4. 结果对比（指标 + PR/ROC 曲线）"""),

        code(COMPARE),

        md("""## 5. 已知 CN 星标定阈值 + 候选体 [Fe/H] 分解

直接观察重要性采样是否改变了候选体在 [Fe/H] > -1.2 与 <= -1.2 两个区域的分布——这是本实验最关心的"误判"指标。"""),

        code(THRESHOLD_EXPORT.replace("@@FNAME@@", "XGB_PU_matched_candidates_threshold.csv")),

        md("""## 6. 结论

**三维分箱分层匹配近邻采样 vs 标准等概率采样：**

1. **物理参数对齐是否生效**：看第 2 节直方图，匹配负采样池的 [Fe/H] 分布应明显更接近正样本
2. **是否影响识别**：对比第 4 节 ROC-AUC / PR-AUC / P@50 / P@100
3. **是否缓解 feh>-1.2 误判**：看第 5 节候选体 [Fe/H] 分解，若重要性采样后 feh>-1.2 候选占比下降，说明原方法在该区域确实混入了参数驱动的误判
4. **调参**：`B`（箱数）越大分层越细、`K`（近邻数）越小匹配越紧；可尝试 B=7 / K=5 观察更激进的对齐效果"""),
    ]
    return n


def build_kde():
    n = nb()
    n.cells = [
        md("""# XGBoost PU Bagging — KDE 核密度加权抽样（重要性采样 II）

**实验动机：** 与实验一相同——[Fe/H] > -1.2 的候选星中有相当一部分 N 丰度较低的"误判"，怀疑标准 PU 的等概率负采样让模型学会了用物理参数区分正负的捷径。

**方法：**
1. 对正样本与 U 池分别拟合 **3D 高斯核密度** f_pos / f_unl（(teff, logg, feh) 空间）
2. 以重要性权重 **w = f_pos / f_unl** 对 U 加权采样：抬高"正样本密度相对更高"区域的采样概率，压低"只有 U、没有正样本"区域的概率
3. 等价于用**密度比做协变量漂移校正**，让负样本的物理参数分布逼近正样本

**与实验一的区别：** 实验一是"硬匹配"（每个正样本找物理最近邻）；本实验是"软加权"（密度比连续重加权，不显式配对），两者互为印证。"""),

        code(SHARED_IMP),

        md("""## 1. 构造 KDE 加权采样器

密度比 w = f_pos / f_unl 是协变量漂移校正的标准重要性权重。ESS（有效样本量）越小，说明权重越集中在少数 U 上，即物理分布被拉得越接近正样本。"""),

        code(PREP_KDE),

        md("""## 2. 采样分布诊断（[Fe/H]）

关键检验：KDE 加权是否把负样本的有效 [Fe/H] 分布拉向正样本分布。橙色曲线是权重重加权的直方图，应明显向红色（正样本）靠拢。"""),

        code(DIAG_KDE),

        md("""## 3. 运行 PU Bagging（T=500）

在 KDE 加权采样器与标准等概率采样器上分别跑 T=500，其余完全一致。"""),

        code(RUN_BOTH),

        md("""## 4. 结果对比（指标 + PR/ROC 曲线）"""),

        code(COMPARE),

        md("""## 5. 已知 CN 星标定阈值 + 候选体 [Fe/H] 分解

直接观察重要性采样是否改变候选体在 [Fe/H] > -1.2 与 <= -1.2 两个区域的分布。"""),

        code(THRESHOLD_EXPORT.replace("@@FNAME@@", "XGB_PU_kde_candidates_threshold.csv")),

        md("""## 6. 结论

**KDE 核密度加权抽样 vs 标准等概率采样：**

1. **物理参数对齐是否生效**：看第 2 节直方图，加权后的 [Fe/H] 分布应更接近正样本
2. **是否影响识别**：对比第 4 节 ROC-AUC / PR-AUC / P@50 / P@100
3. **是否缓解 feh>-1.2 误判**：看第 5 节候选体 [Fe/H] 分解
4. **与实验一对比**：若两种重要性采样结论一致（同升或同降），则结论更稳健；若不一致，说明匹配的"紧/松"程度对结果敏感
5. **调参**：`gaussian_kde` 的带宽可调（如 `bw_method` 缩放因子）；带宽越小，权重越尖锐、对齐越激进"""),
    ]
    return n


# ═══════════════════════════════════════════════════════════════════════
# 实验 3：feh-only 重要性采样（消融：是否只是 feh 的影响）
# ═══════════════════════════════════════════════════════════════════════

PREP_FEHONLY = '''# feh-only 标准化 + 构造两个 feh-only 采样器

name_fm = "feh_matched"
name_fk = "feh_kde"
B, K = 5, 10

# 只用 [Fe/H] 一列做匹配 / 加权（cols=["feh"]），其余物理参数不参与
phys1, mu1, sd1 = imp.standardize_physics(stars_clean, cols=["feh"])
phys1_tr = phys1[tr_idx]
phys1_pos = phys1_tr[np.where(y_tr == 1)[0]]   # (n_pos_tr, 1)
phys1_unl = phys1_tr[unl_tr_idx]               # (n_unl, 1)

# ① feh-only 分箱匹配近邻（一维分箱）
cands_f, n_cell_f = imp.build_match_candidates(phys1_pos, phys1_unl, unl_tr_idx, B=B, K=K)
fm_sampler = imp.MatchSampler(cands_f, imp.random_seed)
feh_match_pool = np.unique(np.concatenate([np.atleast_1d(c) for c in cands_f]))

# ② feh-only KDE 加权（一维核密度）
w_f, ess_f = imp.build_kde_weights(phys1_pos, phys1_unl, unl_tr_idx)
fk_sampler = imp.WeightedSampler(unl_tr_idx, w_f, imp.random_seed)

print(f"feh-only 标准化: mu={mu1[0]:.3f}, sd={sd1[0]:.3f}")
print(f"① feh 分箱匹配: 分箱 B={B}, 近邻 K={K}, 分层命中 {n_cell_f}/{len(cands_f)}, "
      f"有效负池 {len(feh_match_pool):,}")
print(f"② feh KDE 加权: max/median={np.max(w_f)/np.median(w_f):.1f}, ESS={ess_f:.0f}/{len(unl_tr_idx):,}")
'''

DIAG_FEHONLY = '''# 诊断：feh-only 采样后负样本的 [Fe/H] 分布

feh_pos = feh_all[tr_idx[np.where(y_tr == 1)[0]]]
feh_unl = feh_all[tr_idx[unl_tr_idx]]
feh_fmatch = feh_all[tr_idx[feh_match_pool]]

bins = np.linspace(-2.6, -0.6, 41)
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.hist(feh_unl, bins=bins, density=True, alpha=0.45, color="steelblue",
        label=f"U 池（等概率, n={len(feh_unl):,}）")
ax.hist(feh_fmatch, bins=bins, density=True, alpha=0.6, color="darkorange",
        label=f"feh 分箱匹配负池（n={len(feh_fmatch):,}）")
ax.hist(feh_unl, bins=bins, weights=w_f, density=True, alpha=0.6, color="green",
        label="feh KDE 加权（有效分布）")
ax.hist(feh_pos, bins=bins, density=True, alpha=0.8, color="crimson",
        label=f"已知 CN 正样本（n={len(feh_pos)}）")
ax.axvline(-1.2, color="k", linestyle="--", linewidth=1.0, label="[Fe/H] = -1.2")
ax.set_xlabel("[Fe/H]"); ax.set_ylabel("密度")
ax.set_title("feh-only 重要性采样：负样本 [Fe/H] 分布对齐情况")
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()

print(f"U 池(等概率)     中位 feh={np.median(feh_unl):.3f}  feh>-1.2 占比={np.mean(feh_unl > -1.2)*100:.1f}%")
print(f"feh 分箱匹配负池 中位 feh={np.median(feh_fmatch):.3f}  feh>-1.2 占比={np.mean(feh_fmatch > -1.2)*100:.1f}%")
print(f"feh KDE 加权     中位 feh={np.average(feh_unl, weights=w_f):.3f}  feh>-1.2 占比={np.sum(w_f[feh_unl > -1.2])*100:.1f}%")
print(f"正样本           中位 feh={np.median(feh_pos):.3f}  feh>-1.2 占比={np.mean(feh_pos > -1.2)*100:.1f}%")
'''

RUN_FEHONLY = '''# 运行 PU Bagging：标准等概率 vs feh-only 匹配 vs feh-only KDE

T = 500

print("[base] 标准 PU 等概率采样 (T=500) ...")
te_base, p_base = imp.run_pu_bagging(X_tr, X_te, X_all, y_tr, y_te, n_pos_tr,
                                     imp.UniformSampler(unl_tr_idx), T, label="base", report_every=100)

print()
print("[feh_matched] feh-only 分箱匹配近邻 (T=500) ...")
te_fm, p_fm = imp.run_pu_bagging(X_tr, X_te, X_all, y_tr, y_te, n_pos_tr,
                                 fm_sampler, T, label="feh_matched", report_every=100)

print()
print("[feh_kde] feh-only KDE 加权 (T=500) ...")
te_fk, p_fk = imp.run_pu_bagging(X_tr, X_te, X_all, y_tr, y_te, n_pos_tr,
                                 fk_sampler, T, label="feh_kde", report_every=100)

res_base = imp.metrics(y_te, te_base)
res_fm = imp.metrics(y_te, te_fm)
res_fk = imp.metrics(y_te, te_fk)

print()
for nm, r in [("标准 PU(等概率)", res_base), ("feh 分箱匹配近邻", res_fm), ("feh KDE 加权", res_fk)]:
    print(f"{nm:20s} ROC={r['roc']:.4f}  PR={r['pr']:.4f}  P@50={r['p50']:.4f}  P@100={r['p100']:.4f}")
'''

COMPARE_FEHONLY = '''# 三路对比：指标表 + PR/ROC 曲线

from sklearn.metrics import precision_recall_curve, roc_curve, auc

comp = pd.DataFrame({
    "采样方式": ["标准 PU (等概率)", "feh 分箱匹配近邻", "feh KDE 加权"],
    "ROC-AUC": [res_base["roc"], res_fm["roc"], res_fk["roc"]],
    "PR-AUC": [res_base["pr"], res_fm["pr"], res_fk["pr"]],
    "P@50": [res_base["p50"], res_fm["p50"], res_fk["p50"]],
    "P@100": [res_base["p100"], res_fm["p100"], res_fk["p100"]],
})
print(comp.to_string(index=False))

base_rate = y_te.sum() / len(y_te)
series = [("标准 PU", te_base, res_base, "b"), ("feh 匹配", te_fm, res_fm, "darkorange"), ("feh KDE", te_fk, res_fk, "green")]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax1 = axes[0]
for nm, te, r, c in series:
    pr, rc, _ = precision_recall_curve(y_te, te)
    ax1.plot(rc, pr, color=c, linewidth=2, label=f"{nm} (AP={r['pr']:.3f})")
ax1.axhline(base_rate, color="gray", linestyle="--", linewidth=1.0, label=f"Random ({base_rate:.3f})")
ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision")
ax1.set_title("Precision-Recall Curve")
ax1.legend(fontsize=9); ax1.grid(alpha=0.2); ax1.set_xlim(0, 1.02); ax1.set_ylim(0, 1.02)

ax2 = axes[1]
for nm, te, r, c in series:
    fpr, tpr, _ = roc_curve(y_te, te)
    ax2.plot(fpr, tpr, color=c, linewidth=2, label=f"{nm} (AUC={auc(fpr, tpr):.3f})")
ax2.plot([0, 1], [0, 1], "gray", linestyle="--", linewidth=1.0)
ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve")
ax2.legend(fontsize=9); ax2.grid(alpha=0.2); ax2.set_xlim(0, 1.02); ax2.set_ylim(0, 1.02)

fig.suptitle("标准 PU vs feh-only 重要性采样", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
'''

THRESHOLD_FEHONLY = '''# 已知 CN 星标定阈值 + 候选体 [Fe/H] 分解

recall_quantile = 0.05

def feh_split(prob, thr):
    cand = (stars_clean["label"].values == -1) & (prob >= thr)
    f = feh_all[cand]
    return dict(n=int(cand.sum()), gt=int((f > -1.2).sum()), le=int((f <= -1.2).sum()),
                median=float(np.median(f)))

thr_base, ncand_base, _ = imp.threshold_candidates(stars_clean, p_base, recall_quantile)
thr_fm, ncand_fm, _ = imp.threshold_candidates(stars_clean, p_fm, recall_quantile)
thr_fk, ncand_fk, _ = imp.threshold_candidates(stars_clean, p_fk, recall_quantile)
fs_base, fs_fm, fs_fk = feh_split(p_base, thr_base), feh_split(p_fm, thr_fm), feh_split(p_fk, thr_fk)

print("候选体 [Fe/H] 分解:")
for nm, thr, nc, fs in [("标准 PU(等概率)", thr_base, ncand_base, fs_base),
                        ("feh 分箱匹配近邻", thr_fm, ncand_fm, fs_fm),
                        ("feh KDE 加权", thr_fk, ncand_fk, fs_fk)]:
    print(f"  {nm:18s} thr={thr:.4f}  n={nc:5d}  feh>-1.2: {fs['gt']:5d}  feh<=-1.2: {fs['le']:5d}  中位feh={fs['median']:.3f}")

outpath = str(_PROJECT_ROOT / "XGB" / "XGB_PU_feholly_matched_candidates_threshold.csv")
imp.export_candidates(stars_clean, p_fm, thr_fm, outpath)
print(f"\\nfeh 分箱匹配候选体已导出: {outpath}")
'''

LOAD_3D_COMPARE = '''# 五路对比：与 3D（teff/logg/feh）重要性采样合并

import json
fp = _PROJECT_ROOT / "XGB" / "importance_sampling_results.json"
if not fp.exists():
    print("未找到 importance_sampling_results.json（需先运行 3D 实验）。")
else:
    saved = json.loads(fp.read_text(encoding="utf-8"))["results"]
    def row(nm, r):
        return {"采样方式": nm, "ROC-AUC": f"{r['roc']:.4f}", "PR-AUC": f"{r['pr']:.4f}",
                "候选数": r["n_candidates"], "feh>-1.2": r["feh"]["n_feh_gt_-1.2"],
                "feh<=-1.2": r["feh"]["n_feh_le_-1.2"]}
    feh_fm = {"roc": res_fm["roc"], "pr": res_fm["pr"], "n_candidates": ncand_fm, "feh": fs_fm}
    feh_fk = {"roc": res_fk["roc"], "pr": res_fk["pr"], "n_candidates": ncand_fk, "feh": fs_fk}
    rows = [
        row("标准 PU (等概率)", saved["baseline"]),
        row("3D 分箱匹配近邻", saved["matched"]),
        row("feh 分箱匹配近邻", feh_fm),
        row("3D KDE 加权", saved["kde"]),
        row("feh KDE 加权", feh_fk),
    ]
    print(pd.DataFrame(rows).to_string(index=False))
'''


def build_feholly():
    n = nb()
    n.cells = [
        md("""# XGBoost PU Bagging — feh-only 重要性采样（消融：是否只是 [Fe/H] 的影响）

**实验目的：** 上一版 3D（teff/logg/feh）重要性采样显著拉低了测试排序性能、放大了候选数。本实验做**消融**——只把 **[Fe/H]** 一个变量用于匹配/加权，检验"是否只是 feh 的影响"：
- 若 feh-only 几乎复现 3D 的效果 → [Fe/H] 是主导混淆变量，teff/logg 贡献很小
- 若 feh-only 明显更弱 → teff/logg 也贡献了不可忽略的混淆

**方法（与 3D 版完全同构，只是维度从 3 降到 1）：**
1. **feh 分箱匹配近邻**：一维 [Fe/H] 分位分箱（5 箱），每个正样本在同箱内取 K=10 个最近 feh 邻居
2. **feh KDE 加权**：一维 [Fe/H] 高斯核密度比 w = f_pos / f_unl 加权采样

**对照：** 标准 PU（等概率采样）在同划分、同随机种子下重跑；并与已保存的 3D 结果合并成五路对比。"""),

        code(SHARED_IMP),

        md("""## 1. 构造 feh-only 采样器

只把 [Fe/H] 一个变量喂给匹配/KDE 构造，其余物理参数不参与。"""),

        code(PREP_FEHONLY),

        md("""## 2. 采样分布诊断（[Fe/H]）

观察 feh-only 匹配/KDE 是否把负样本的 [Fe/H] 分布拉向正样本。"""),

        code(DIAG_FEHONLY),

        md("""## 3. 运行 PU Bagging（T=500）

标准等概率 vs feh 匹配 vs feh KDE，三者同划分同种子。"""),

        code(RUN_FEHONLY),

        md("""## 4. 三路对比（指标 + PR/ROC 曲线）"""),

        code(COMPARE_FEHONLY),

        md("""## 5. 已知 CN 星标定阈值 + 候选体 [Fe/H] 分解"""),

        code(THRESHOLD_FEHONLY),

        md("""## 6. 五路对比（与 3D 重要性采样合并）"""),

        code(LOAD_3D_COMPARE),

        md("""## 7. 结论

**feh-only 重要性采样 vs 3D（teff/logg/feh）重要性采样：**

1. **是否只是 feh 的影响**：把 feh-only 匹配/KDE 与 3D 匹配/KDE 逐项对比（第 6 节）
   - feh-only ≈ 3D → [Fe/H] 是主导混淆变量
   - feh-only 明显更弱 → teff/logg 也贡献了混淆
2. **feh 分箱匹配 vs feh KDE 加权**：硬匹配通常比软加权对齐更激进，比较两者幅度即可判断"对齐强度"是否敏感
3. **与标准 PU 对比**：重要性采样（无论维度）都通过"抽掉物理捷径"暴露更弱的纯 CN 信号

> 判读要点：本实验不是要"改进"指标，而是要回答"基线高分里有多少来自 feh 这个混淆变量"。"""),
    ]
    return n


if __name__ == "__main__":
    BASE.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "ML_XGB_PU_matched.ipynb": build_matched(),
        "ML_XGB_PU_kde.ipynb": build_kde(),
        "ML_XGB_PU_feholly.ipynb": build_feholly(),
    }
    for rel_path, nb_obj in notebooks.items():
        out_path = BASE / rel_path
        with open(out_path, "w", encoding="utf-8") as f:
            nbf.write(nb_obj, f)
        print(f"Created: {out_path}")
    print(f"\nAll {len(notebooks)} importance-sampling notebooks built successfully!")
