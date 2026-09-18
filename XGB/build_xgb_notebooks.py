"""Build the XGB experiment notebooks.

Generated notebooks:
    - ML_XGB_PU_threshold.ipynb   # baseline (copied from PhaseSummary, paths fixed)
    - ML_XGB_PU_keybands.ipynb    # experiment 1: key-band input (CN3839/CN4142/CH4300)
    - ML_XGB_PU_twostage.ipynb    # experiment 2: two-stage PU

Run: D:/Anaconda/envs/myenv/python.exe XGB/build_xgb_notebooks.py
"""

import nbformat as nbf
from pathlib import Path

BASE = Path(__file__).resolve().parent          # XGB/
PROJECT_ROOT = BASE.parent                       # Lamost/


def nb():
    return nbf.v4.new_notebook(metadata={
        "kernelspec": {
            "display_name": "myenv",
            "language": "python",
            "name": "myenv",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0",
        },
    })


def md(src):
    return nbf.v4.new_markdown_cell(src)


def code(src):
    return nbf.v4.new_code_cell(src)


# ═══════════════════════════════════════════════════════════════════════
# Shared data loading cell (used by both experiments)
# ═══════════════════════════════════════════════════════════════════════

SHARED_LOAD = '''# 共享数据加载与基础库导入

import sys, os, time
from pathlib import Path

# 智能定位项目根目录：向上查找直到同时存在 ML/ 与 Data/
_PROJECT_ROOT = Path(os.getcwd())
for _ in range(5):
    if (_PROJECT_ROOT / "ML").exists() and (_PROJECT_ROOT / "Data").exists():
        break
    _PROJECT_ROOT = _PROJECT_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 10})
plt.rcParams.update({'axes.labelsize': 'large'})
import warnings
warnings.filterwarnings('ignore')

from PhaseSummary.shared.data_loader import BAND_DEFS, MOLECULAR_BAND_RANGES, CN_BAND_NAMES
from build_dr13_all_cache import load_dr13_all_cache

t0 = time.time()
data = load_dr13_all_cache()
X_clean = data['X_clean']
stars_clean = data['stars_clean']
feature_df = data['feature_df']
common_wave = data['common_wave']

print(f"数据加载完成 ({time.time()-t0:.0f}s):")
print(f"  归一化光谱 X_clean: {X_clean.shape}")
print(f"  恒星数量:           {len(stars_clean)}")
print(f"  已知CN星:          {(stars_clean['label']==1).sum()}")
print(f"  波长范围:           {common_wave[0]:.0f}-{common_wave[-1]:.0f} Å")
'''


# ═══════════════════════════════════════════════════════════════════════
# 1) Baseline: copy ML_XGB_PU_threshold, strip outputs, fix save paths
# ═══════════════════════════════════════════════════════════════════════

def build_baseline():
    src = PROJECT_ROOT / "PhaseSummary" / "03_ML_XGB" / "ML_XGB_PU_threshold.ipynb"
    nb_obj = nbf.read(str(src), as_version=4)
    for cell in nb_obj.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
        # 把输出路径从 PhaseSummary/03_ML_XGB 重定向到 XGB
        cell.source = cell.source.replace("PhaseSummary/03_ML_XGB", "XGB")
    return nb_obj


# ═══════════════════════════════════════════════════════════════════════
# 2) Experiment 1: key-band input
# ═══════════════════════════════════════════════════════════════════════

def build_keybands():
    nb1 = nb()
    nb1.cells = [
        md("""# XGBoost PU Bagging — 关键波段输入（CN3839 / CN4142 / CH4300）

**实验目的：** 与全波段基线 `ML_XGB_PU_threshold` 对比，只把三个 CN/CH 分子带窗口作为模型输入，检验"仅凭关键波段"能否达到（或接近）全波段 700 维的性能，进而判断 CN 判别信息是否主要集中在这三个分子带上。

**核心改动（相对基线）：**
1. 输入从全波段 **700 维** → 三个关键分子带窗口拼接，共 **182 维**
2. 分子带窗口：CN3839 (3830–3883Å)、CN4142 (4120–4216Å)、CH4300 (4285–4315Å)
3. 其余流程（PU Bagging T=500、已知 CN 星标定阈值、候选体导出）与基线完全一致，保证可比性

**预期：** 若关键波段性能与全波段相当，说明分子带承载了绝大部分判别信息；若明显下降，说明连续谱形态（温度/金属丰度背景）也提供了重要线索。"""),

        code(SHARED_LOAD),

        md("""## 1. 关键波段提取

从连续谱归一化光谱中裁剪出三个 CN/CH 分子带窗口并拼接。分子带是 CN 吸收最强、最具有判别力的区域；这里**仅输入分子带本身**（不含两侧连续谱），考察"关键波段"是否足够支撑识别。"""),

        code("""# 提取三个关键分子带波段（CN3839 / CN4142 / CH4300），拼接成 (N, 182) 输入
# MOLECULAR_BAND_RANGES = [(3830, 3883), (4120, 4216), (4285, 4315)] —— 只取分子带窗口

band_ranges = MOLECULAR_BAND_RANGES  # 三个关键波段
band_masks = [(common_wave >= l1) & (common_wave <= l2) for (l1, l2) in band_ranges]

X_bands = np.concatenate([X_clean[:, m] for m in band_masks], axis=1).astype(np.float32)
band_wave = np.concatenate([common_wave[m] for m in band_masks])
n_px_per_band = [int(m.sum()) for m in band_masks]

print("关键波段输入:")
for (l1, l2), npx in zip(band_ranges, n_px_per_band):
    print(f"  {l1}-{l2} Å : {npx} 像素")
print(f"  总输入维度: {X_bands.shape[1]}  (全波段为 {X_clean.shape[1]})")
print(f"  维度压缩: {X_clean.shape[1]} -> {X_bands.shape[1]}  ({X_bands.shape[1]/X_clean.shape[1]*100:.1f}%)")
"""),

        md("""## 2. 标准化与数据划分

与基线完全一致的标准化 + 分层划分（seed=42, test=15%, val=18%），保证两者在完全相同的数据划分上比较。"""),

        code("""# 标准化 + 分层划分（与基线一致）

import time, random
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

random_seed = 42

y_all = stars_clean['label'].map({1: 1, -1: 0}).values.astype(int)
X_spectra = X_bands.astype(np.float32)   # 关键波段输入 (N, 182)

ss = StandardScaler()
X_spec_scaled = ss.fit_transform(X_spectra).astype(np.float32)

all_idx = np.arange(len(y_all))
tv_idx, test_idx = train_test_split(
    all_idx, test_size=0.15, stratify=y_all, random_state=random_seed)
tr_idx, val_idx = train_test_split(
    tv_idx, test_size=0.18, stratify=y_all[tv_idx], random_state=random_seed)

n_pos = int(y_all.sum())
print(f"总样本: {len(y_all):,}  正样本(P): {n_pos}  负/未标注(U): {len(y_all)-n_pos:,}")
print(f"训练集: {len(tr_idx):,}  测试集: {len(test_idx):,}  输入维度: {X_spec_scaled.shape[1]}")
"""),

        md("""## 3. 运行 PU Bagging（T=500）

在 182 维关键波段输入上运行 PU Bagging，流程与基线完全相同。"""),

        code("""# 运行 PU Bagging (T=500) — 关键波段输入

T = 500
print(f"运行 PU Bagging (T={T}) — 输入维度 {X_spec_scaled.shape[1]}...")

X_tr = X_spec_scaled[tr_idx]
X_te = X_spec_scaled[test_idx]
y_tr = y_all[tr_idx]
y_te = y_all[test_idx]

pos_tr_idx = np.where(y_tr == 1)[0]
n_pos_tr = len(pos_tr_idx)
unl_tr_idx = np.where(y_tr == 0)[0]
X_pos = X_tr[pos_tr_idx]

rng = random.Random(random_seed)

te_prob_sum = np.zeros(len(test_idx), dtype=np.float64)
te_prob_sq = np.zeros(len(test_idx), dtype=np.float64)
all_prob_sum = np.zeros(len(y_all), dtype=np.float64)
all_prob_sq = np.zeros(len(y_all), dtype=np.float64)

xgb_params = {
    "max_depth": 3, "learning_rate": 0.1,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 1, "gamma": 0.0,
    "reg_alpha": 0.0, "reg_lambda": 1.0,
    "seed": random_seed, "verbosity": 0, "n_jobs": 1,
}

t0 = time.time()
for t in range(1, T + 1):
    neg_sample = rng.sample(list(unl_tr_idx), n_pos_tr)
    X_neg = X_tr[neg_sample]
    X_bal = np.vstack([X_pos, X_neg])
    y_bal = np.hstack([np.ones(n_pos_tr), np.zeros(n_pos_tr)])

    dtrain = xgb.DMatrix(X_bal, label=y_bal)
    model = xgb.train(xgb_params, dtrain, num_boost_round=50, verbose_eval=False)

    p_te = model.predict(xgb.DMatrix(X_te))
    p_all = model.predict(xgb.DMatrix(X_spec_scaled))

    te_prob_sum += p_te
    te_prob_sq += p_te ** 2
    all_prob_sum += p_all
    all_prob_sq += p_all ** 2

    if t % 100 == 0:
        p_m = te_prob_sum / t
        pr = average_precision_score(y_te, p_m)
        roc = roc_auc_score(y_te, p_m)
        print(f"  [{t:4d}/{T}]  ROC={roc:.4f}  PR={pr:.4f}  ({time.time()-t0:.0f}s)")

p_te_mean = te_prob_sum / T
p_all_mean = all_prob_sum / T
p_all_std = np.sqrt(np.maximum(all_prob_sq / T - p_all_mean**2, 0))

elapsed = time.time() - t0
roc = roc_auc_score(y_te, p_te_mean)
pr = average_precision_score(y_te, p_te_mean)

order = np.argsort(p_te_mean)[::-1]
p50 = y_te[order[:min(50, len(order))]].mean()
p100 = y_te[order[:min(100, len(order))]].mean()

print(f"PU Bagging 完成 ({elapsed:.0f}s)")
print(f"  测试集 ROC-AUC: {roc:.4f}")
print(f"  测试集 PR-AUC: {pr:.4f}")
print(f"  Precision@50: {p50:.4f}")
print(f"  Precision@100: {p100:.4f}")
print(f"  平均概率: {p_all_mean.mean():.4f} ± {p_all_mean.std():.4f}")

# 保存结果
stars_clean['xgb_pu_prob'] = p_all_mean
stars_clean['xgb_pu_std'] = p_all_std
"""),

        md("""## 4. 性能曲线（PR / ROC）"""),

        code("""# 绘制 PR 曲线与 ROC 曲线
from sklearn.metrics import precision_recall_curve, roc_curve, auc

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

precision, recall, _ = precision_recall_curve(y_te, p_te_mean)
pr_auc = average_precision_score(y_te, p_te_mean)
baseline = y_te.sum() / len(y_te)

ax1 = axes[0]
ax1.plot(recall, precision, 'b-', linewidth=2, label=f'XGB PU key-bands (AP={pr_auc:.3f})')
ax1.axhline(baseline, color='gray', linestyle='--', linewidth=1.0,
            label=f'Random baseline ({baseline:.3f})')
ax1.fill_between(recall, precision, baseline, alpha=0.08, color='blue')
ax1.set_xlabel('Recall'); ax1.set_ylabel('Precision')
ax1.set_title('Precision-Recall Curve')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(alpha=0.2); ax1.set_xlim(0, 1.02); ax1.set_ylim(0, 1.02)

fpr, tpr, _ = roc_curve(y_te, p_te_mean)
roc_auc = auc(fpr, tpr)
ax2 = axes[1]
ax2.plot(fpr, tpr, 'darkred', linewidth=2, label=f'XGB PU key-bands (AUC={roc_auc:.3f})')
ax2.plot([0, 1], [0, 1], 'gray', linestyle='--', linewidth=1.0, label='Random (AUC=0.500)')
ax2.fill_between(fpr, tpr, 0, alpha=0.08, color='darkred')
ax2.set_xlabel('False Positive Rate'); ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')
ax2.legend(fontsize=9, loc='lower right')
ax2.grid(alpha=0.2); ax2.set_xlim(0, 1.02); ax2.set_ylim(0, 1.02)

fig.suptitle('XGBoost PU Bagging (Key Bands) — Performance on Test Set', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

print(f'PR-AUC: {pr_auc:.4f}  |  ROC-AUC: {roc_auc:.4f}  |  N_test: {len(y_te)}')
"""),

        md("""## 5. 已知 CN 星标定阈值

用已知 CN 星概率的低分位标定候选阈值（保留约 95% 已知 CN 星），与基线做法一致。"""),

        code("""# 已知CN星标定候选阈值
from sklearn.neighbors import NearestNeighbors

known_mask = stars_clean['label'] == 1
known_scores = stars_clean.loc[known_mask, 'xgb_pu_prob'].values
recall_quantile = 0.05   # 保留约 95% 已知CN星

prob_threshold = float(np.nanquantile(known_scores, recall_quantile))

cand_mask = (stars_clean['label'] == -1) & (stars_clean['xgb_pu_prob'] >= prob_threshold)
candidates = stars_clean.loc[cand_mask].sort_values('xgb_pu_prob', ascending=False).copy()

n_known_above = int((known_scores >= prob_threshold).sum())

print(f"已知CN星概率: min={known_scores.min():.3f}  "
      f"q05={np.nanquantile(known_scores, 0.05):.3f}  "
      f"median={np.nanmedian(known_scores):.3f}  max={known_scores.max():.3f}")
print(f"标定阈值 (保留~{(1-recall_quantile)*100:.0f}%已知CN星): prob_threshold = {prob_threshold:.4f}")
print(f"已知CN星中 >= 阈值: {n_known_above}/{len(known_scores)}")
print(f"未标注候选体数量: {len(candidates)}  ({len(candidates)/len(stars_clean)*100:.2f}%)")

# 参数近邻平均光谱辅助函数（用于可视化对比，与基线一致）
_param_cols = ('teff', 'logg', 'feh')
n_param_nn = 50

def build_param_nn(stars, param_cols=_param_cols, n_neighbors=n_param_nn):
    P = stars[list(param_cols)].to_numpy(dtype=float)
    mu = np.nanmean(P, axis=0)
    sd = np.nanstd(P, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Ps = (P - mu) / sd
    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(P)), metric='euclidean')
    nn.fit(Ps)
    return nn, Ps

def param_cluster_mean_spec(idx, stars, Ps, X, n_neighbors=n_param_nn):
    cid_arr = stars['masked_cluster_id'].values
    cid = cid_arr[idx]
    cluster_pos = np.where(cid_arr == cid)[0]
    cluster_pos = cluster_pos[cluster_pos != idx]
    if len(cluster_pos) == 0:
        return None
    d = np.linalg.norm(Ps[cluster_pos] - Ps[idx], axis=1)
    order = np.argsort(d)
    sel = cluster_pos[order[:min(n_neighbors, len(cluster_pos))]]
    return np.nanmedian(X[sel], axis=0)

param_nn, Ps_all = build_param_nn(stars_clean, n_neighbors=n_param_nn)
print(f"参数近邻模型已构建 (n_neighbors={n_param_nn})")
"""),

        md("""## 6. 候选体可视化：光谱

展示候选体的**完整光谱**（归一化），并用色块标出作为模型输入的三个关键波段，便于直观判断 CN 增峰是否落在输入波段内。"""),

        code("""# 候选体光谱可视化（完整光谱 + 关键波段高亮）

top_n = 3
cand_vis = candidates.head(top_n)
cand_indices = cand_vis.index.values

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes = axes.flatten()

for i, idx in enumerate(cand_indices):
    ax = axes[i]
    flux = X_clean[idx]   # 完整 700 维光谱

    # 对比线：同一掩盖聚类内、三参数最接近的恒星平均光谱
    param_mean = param_cluster_mean_spec(idx, stars_clean, Ps_all, X_clean)
    ax.plot(common_wave, param_mean, color='seagreen',
            linewidth=1.0, linestyle='--', alpha=0.85,
            label='Cluster+Param-NN mean')
    ax.plot(common_wave, flux, color='navy', linewidth=0.8, label='Candidate')

    # 高亮作为模型输入的关键波段
    for l1, l2, c in [(3830, 3883, 'blue'), (4120, 4216, 'green'), (4285, 4315, 'red')]:
        ax.axvspan(l1, l2, alpha=0.12, color=c, zorder=0)

    prob = cand_vis.iloc[i]['xgb_pu_prob']
    teff = cand_vis.iloc[i]['teff']
    ax.set_title(f'#{i+1} | Teff={teff:.0f}K | prob={prob:.3f}', fontsize=8)
    ax.set_xlim(3800, 4500)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=6, loc='upper right')

for j in range(top_n, len(axes)):
    axes[j].axis('off')

fig.suptitle('XGBoost PU Bagging (Key Bands) — Top Candidates Spectra (Input bands shaded)', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
"""),

        md("""## 7. 导出高置信度候选体（已知 CN 星标定阈值）"""),

        code("""# 导出结果
out_cols = ['uid', 'ra', 'dec', 'teff', 'logg', 'feh', 'label', 'snru',
            'xgb_pu_prob', 'xgb_pu_std', 'masked_cluster_id']
out_cols = [c for c in out_cols if c in candidates.columns]

out_df = candidates[out_cols].copy()
outpath = str(_PROJECT_ROOT / 'XGB/XGB_PU_keybands_candidates_threshold.csv')
out_df.to_csv(outpath, index=False)

print(f"候选体已导出: {outpath}")
print(f"筛选阈值 prob_threshold = {prob_threshold:.4f}  (保留~{(1-recall_quantile)*100:.0f}%已知CN星)")
print(f"候选体统计 (共 {len(out_df)} 颗):")
print(f"  prob > 0.5: {(out_df['xgb_pu_prob'] > 0.5).sum()}")
print(f"  prob > 0.7: {(out_df['xgb_pu_prob'] > 0.7).sum()}")
"""),

        md("""## 8. 结论（与全波段对比）

**关键波段（182 维） vs 全波段（700 维）对比要点：**

1. **判别信息集中度**：若关键波段的 ROC-AUC / PR-AUC 接近全波段，说明 CN 判别信息高度集中在三个分子带；若明显下降，说明连续谱背景（温度、金属丰度、光度）也贡献了不可忽略的信息
2. **过拟合与稳定性**：更低的输入维度通常带来更快的训练、更低的过拟合风险，但也可能丢失连续谱的物理上下文
3. **物理可解释性**：关键波段输入天然聚焦在 CN/CH 吸收特征上，候选体的可解释性更强
4. **与全波段候选体对比**：可进一步比较两份候选体 CSV 的重叠率，判断两者是否捕获同一批目标

> 将本 notebook 得到的 ROC-AUC / PR-AUC / P@50 / P@100 与基线 `ML_XGB_PU_threshold` 逐项对比即可得出结论。"""),
    ]
    return nb1


# ═══════════════════════════════════════════════════════════════════════
# 3) Experiment 2: two-stage PU
# ═══════════════════════════════════════════════════════════════════════

def build_twostage():
    nb2 = nb()
    nb2.cells = [
        md("""# XGBoost PU Bagging — 两阶段 PU（净化负采样池）

**实验目的：** 标准 PU Bagging 每次从**全部**未标注集 U 中等量采样"负样本"。但 U 中很可能混有未被标记的真实 CN 星（潜在正样本），把它们当作负样本喂给模型会引入**标签噪声**，压制其真实概率。本实验引入两阶段策略：

1. **阶段一（100 轮）**：先跑一轮初始 Bagging，得到每个 U 样本的初步概率
2. **净化负采样池**：把 U 中"概率持续极高"的样本（大概率是潜在正样本）**暂时移出负采样候选池**
3. **阶段二（400 轮）**：在净化后的剩余样本上继续训练

**与标准 PU 的差异：** 标准 PU 全程 500 轮都在完整 U 上采样；两阶段 PU 用前 100 轮"发现"并剔除疑似正样本，后 400 轮避免用它们当负样本。"""),

        code(SHARED_LOAD),

        md("""## 1. 两阶段 PU 原理

**为什么需要两阶段？**

PU Learning 的经典假设是"负采样池中不含正样本"。但在 CN 星识别里，U = 全部未标注恒星，其中**必然**混有大量真实但未被文献收录的 CN 星。若每次都从完整 U 采样负样本，等于定期把"潜在正样本"当负样本训练，模型会：
- 对这批样本输出被压低的概率（因为它们反复被标注为负）
- 丢失对"更难分辨的 CN 星"的判别能力

**两阶段思路（本实验）：**

- 阶段一用标准 PU 跑 100 轮，得到的概率虽然粗糙，但足以暴露"哪些 U 样本始终被模型判为很像 CN"
- 把"均值极高 + 各轮持续高"的样本从负采样池移除（注意：**不是**把它们改成正样本，只是不再当负样本）
- 阶段二在净化后的池上继续 400 轮，负样本更"干净"，模型对真实负样本的边界更清晰"""),

        md("""## 2. 数据准备与划分

与基线完全一致：标准化 + 分层划分（seed=42）。两阶段与标准 PU 在**完全相同**的数据划分上比较。"""),

        code("""# 标准化 + 分层划分（与基线一致）

import time, random
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

random_seed = 42

y_all = stars_clean['label'].map({1: 1, -1: 0}).values.astype(int)
X_spectra = X_clean.astype(np.float32)   # 全波段 700 维

ss = StandardScaler()
X_spec_scaled = ss.fit_transform(X_spectra).astype(np.float32)

all_idx = np.arange(len(y_all))
tv_idx, test_idx = train_test_split(
    all_idx, test_size=0.15, stratify=y_all, random_state=random_seed)
tr_idx, val_idx = train_test_split(
    tv_idx, test_size=0.18, stratify=y_all[tv_idx], random_state=random_seed)

X_tr = X_spec_scaled[tr_idx]
X_te = X_spec_scaled[test_idx]
y_tr = y_all[tr_idx]
y_te = y_all[test_idx]

pos_tr_idx = np.where(y_tr == 1)[0]
n_pos_tr = len(pos_tr_idx)
unl_tr_idx = np.where(y_tr == 0)[0]   # 训练集内 U 的"位置"（索引 X_tr 的行）
X_pos = X_tr[pos_tr_idx]

n_pos = int(y_all.sum())
print(f"总样本: {len(y_all):,}  正样本(P): {n_pos}  负/未标注(U): {len(y_all)-n_pos:,}")
print(f"训练集: {len(tr_idx):,}  其中正样本: {n_pos_tr}  训练U: {len(unl_tr_idx):,}")
print(f"测试集: {len(test_idx):,}")
"""),

        md("""## 3. 通用 PU Bagging 训练函数

封装单次 PU Bagging，返回测试集均值概率、全样本均值概率、全样本标准差，以及**每样本在多少轮中被判为 >= 0.5**（用于阶段一的"持续性"统计）。"""),

        code("""# 通用 PU Bagging 训练函数

xgb_params = {
    "max_depth": 3, "learning_rate": 0.1,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 1, "gamma": 0.0,
    "reg_alpha": 0.0, "reg_lambda": 1.0,
    "seed": random_seed, "verbosity": 0, "n_jobs": 1,
}

def run_pu_bagging(neg_pool, T, rng, label="", report_every=100):
    \"\"\"在给定负采样池 neg_pool（训练集内位置）上跑 T 轮 PU Bagging。

    返回 (te_mean, all_mean, all_std, high_cnt)：
      te_mean  : 测试集平均概率
      all_mean : 全样本平均概率
      all_std  : 全样本概率标准差
      high_cnt : 全样本在多少轮中被判为 >= 0.5（用于"持续性"判断）
    \"\"\"
    te_sum = np.zeros(len(test_idx), dtype=np.float64)
    te_sq = np.zeros(len(test_idx), dtype=np.float64)
    all_sum = np.zeros(len(y_all), dtype=np.float64)
    all_sq = np.zeros(len(y_all), dtype=np.float64)
    high_cnt = np.zeros(len(y_all), dtype=np.int64)

    t0 = time.time()
    for t in range(1, T + 1):
        neg = rng.sample(list(neg_pool), n_pos_tr)
        X_neg = X_tr[neg]
        X_bal = np.vstack([X_pos, X_neg])
        y_bal = np.hstack([np.ones(n_pos_tr), np.zeros(n_pos_tr)])

        dtrain = xgb.DMatrix(X_bal, label=y_bal)
        model = xgb.train(xgb_params, dtrain, num_boost_round=50, verbose_eval=False)

        p_te = model.predict(xgb.DMatrix(X_te))
        p_all = model.predict(xgb.DMatrix(X_spec_scaled))

        te_sum += p_te
        te_sq += p_te ** 2
        all_sum += p_all
        all_sq += p_all ** 2
        high_cnt += (p_all >= 0.5).astype(np.int64)

        if t % report_every == 0:
            pm = te_sum / t
            print(f"  [{label} {t:4d}/{T}]  ROC={roc_auc_score(y_te, pm):.4f}  "
                  f"PR={average_precision_score(y_te, pm):.4f}  ({time.time()-t0:.0f}s)")

    te_mean = te_sum / T
    all_mean = all_sum / T
    all_std = np.sqrt(np.maximum(all_sq / T - all_mean ** 2, 0))
    return te_mean, all_mean, all_std, high_cnt


def report(name, te_mean):
    roc = roc_auc_score(y_te, te_mean)
    pr = average_precision_score(y_te, te_mean)
    order = np.argsort(te_mean)[::-1]
    p50 = y_te[order[:min(50, len(order))]].mean()
    p100 = y_te[order[:min(100, len(order))]].mean()
    print(f"{name:28s}  ROC={roc:.4f}  PR={pr:.4f}  P@50={p50:.4f}  P@100={p100:.4f}")
    return dict(roc=roc, pr=pr, p50=p50, p100=p100)
"""),

        md("""## 4. 阶段一：100 轮初始 Bagging + 识别潜在正样本

用标准 PU 跑 100 轮，然后按"均值极高 **且** 各轮持续高"的标准，把 U 中的疑似正样本识别出来。"""),

        code("""# ── 阶段一：100 轮初始 Bagging ──
T1 = 100
print(f"阶段一：初始 Bagging (T1={T1})...")
rng2 = random.Random(random_seed)   # 两阶段共用同一条随机流

te1, p1_mean, p1_std, p1_high = run_pu_bagging(
    unl_tr_idx, T1, rng2, label="stage1", report_every=50)

# ── 识别"持续极高"概率的 U 样本 ──
# 注意索引：unl_tr_idx 是训练集内的"位置"，需映射回全样本索引 tr_idx[unl_tr_idx]
unl_full_idx = tr_idx[unl_tr_idx]

HIGH_THRESH = 0.8     # 概率均值阈值（"极高"）
MIN_PERSIST = 0.8     # 持续性阈值：至少 80% 的 bag 中 prob >= 0.5

p1_u = p1_mean[unl_full_idx]
persist_u = p1_high[unl_full_idx] / T1

relabel_mask = (p1_u >= HIGH_THRESH) & (persist_u >= MIN_PERSIST)
removed_full = unl_full_idx[relabel_mask]
clean_pool = unl_tr_idx[~relabel_mask]   # 净化后的负采样池（训练集内位置）

print()
print(f"训练 U 总数: {len(unl_tr_idx):,}")
for thr in (0.5, 0.7, 0.8, 0.9):
    print(f"  p1_mean >= {thr}: {(p1_u >= thr).sum():,}")
print(f"被移出负采样池（均值>={HIGH_THRESH} 且 持续>={MIN_PERSIST}）: {len(removed_full):,} "
      f"({len(removed_full)/len(unl_tr_idx)*100:.2f}%)")
print(f"净化后负采样池大小: {len(clean_pool):,}")

# 阶段一初步评估
print()
print("阶段一（100 轮）测试集表现:")
report("stage1 (100 rounds)", te1)
"""),

        md("""## 5. 阶段二：400 轮在净化后的负采样池上训练

负采样只从 `clean_pool`（已移除疑似正样本）中进行。正样本仍为全部已知 CN 星。"""),

        code("""# ── 阶段二：400 轮在净化池上训练 ──
T2 = 400
print(f"阶段二：净化池继续训练 (T2={T2})...")

te2, p2_mean, p2_std, _ = run_pu_bagging(
    clean_pool, T2, rng2, label="stage2", report_every=100)

print()
print("两阶段最终（阶段二 400 轮）测试集表现:")
res_two = report("two-stage (100+400)", te2)
"""),

        md("""## 6. 对比基线：标准 PU Bagging（T=500）

在相同数据划分、相同随机种子下跑一次标准 PU（500 轮都在完整 U 上采样），作为两阶段的直接对照。"""),

        code("""# ── 标准 PU Bagging (T=500) 基线 ──
T_base = 500
print(f"标准 PU Bagging 基线 (T={T_base})...")
rng_base = random.Random(random_seed)

te_base, p_base_mean, p_base_std, _ = run_pu_bagging(
    unl_tr_idx, T_base, rng_base, label="base", report_every=100)

print()
print("标准 PU（500 轮）测试集表现:")
res_base = report("standard (500 rounds)", te_base)
"""),

        md("""## 7. 结果对比

并排对比标准 PU 与两阶段 PU，并叠加 PR / ROC 曲线。"""),

        code("""# ── 指标对比表 ──
import pandas as pd

comp = pd.DataFrame({
    '方法': ['标准 PU (500轮)', '两阶段 PU (100+400轮)'],
    'ROC-AUC': [res_base['roc'], res_two['roc']],
    'PR-AUC': [res_base['pr'], res_two['pr']],
    'P@50': [res_base['p50'], res_two['p50']],
    'P@100': [res_base['p100'], res_two['p100']],
})
print(comp.to_string(index=False))

# ── PR / ROC 曲线叠加 ──
from sklearn.metrics import precision_recall_curve, roc_curve, auc

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
base_rate = y_te.sum() / len(y_te)

# PR
pr_base, rc_base, _ = precision_recall_curve(y_te, te_base)
pr_two, rc_two, _ = precision_recall_curve(y_te, te2)
ax1 = axes[0]
ax1.plot(rc_base, pr_base, 'b-', linewidth=2, label=f'Standard PU (AP={res_base["pr"]:.3f})')
ax1.plot(rc_two, pr_two, 'darkorange', linewidth=2, label=f'Two-stage PU (AP={res_two["pr"]:.3f})')
ax1.axhline(base_rate, color='gray', linestyle='--', linewidth=1.0, label=f'Random ({base_rate:.3f})')
ax1.set_xlabel('Recall'); ax1.set_ylabel('Precision')
ax1.set_title('Precision-Recall Curve')
ax1.legend(fontsize=9); ax1.grid(alpha=0.2)
ax1.set_xlim(0, 1.02); ax1.set_ylim(0, 1.02)

# ROC
fpr_base, tpr_base, _ = roc_curve(y_te, te_base)
fpr_two, tpr_two, _ = roc_curve(y_te, te2)
ax2 = axes[1]
ax2.plot(fpr_base, tpr_base, 'darkred', linewidth=2, label=f'Standard PU (AUC={auc(fpr_base, tpr_base):.3f})')
ax2.plot(fpr_two, tpr_two, 'darkorange', linewidth=2, label=f'Two-stage PU (AUC={auc(fpr_two, tpr_two):.3f})')
ax2.plot([0, 1], [0, 1], 'gray', linestyle='--', linewidth=1.0)
ax2.set_xlabel('False Positive Rate'); ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')
ax2.legend(fontsize=9); ax2.grid(alpha=0.2)
ax2.set_xlim(0, 1.02); ax2.set_ylim(0, 1.02)

fig.suptitle('Standard PU vs Two-stage PU', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
"""),

        md("""## 8. 被移出负采样池样本的形态检查

观察阶段一识别出的"疑似潜在正样本"——它们的最终概率（阶段二）是否依然很高（印证它们确实像 CN，而非噪声）。"""),

        code("""# 被移出负采样池样本的最终概率分布
if len(removed_full) > 0:
    removed_final_prob = p2_mean[removed_full]
    kept_final_prob = p2_mean[tr_idx[clean_pool]]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(kept_final_prob, bins=60, alpha=0.5, color='steelblue',
            label=f'保留在负采样池的 U (n={len(clean_pool):,})')
    ax.hist(removed_final_prob, bins=60, alpha=0.7, color='crimson',
            label=f'被移出的疑似正样本 (n={len(removed_full):,})')
    ax.set_xlabel('Stage-2 final probability')
    ax.set_ylabel('Count')
    ax.set_title('Final Probability: Removed vs Kept U Samples')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    print(f"被移出样本的阶段二平均概率: {removed_final_prob.mean():.3f}")
    print(f"保留样本的阶段二平均概率:   {kept_final_prob.mean():.3f}")

    # 展示概率最高的几个被移出样本的光谱
    top_removed = np.argsort(removed_final_prob)[::-1][:3]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes = axes.flatten()
    for i, k in enumerate(top_removed):
        idx = removed_full[k]
        ax = axes[i]
        ax.plot(common_wave, X_clean[idx], color='navy', linewidth=0.8, label='Spectrum')
        for l1, l2, c in [(3830, 3883, 'blue'), (4120, 4216, 'green'), (4285, 4315, 'red')]:
            ax.axvspan(l1, l2, alpha=0.12, color=c, zorder=0)
        teff = stars_clean.iloc[idx]['teff']
        prob = removed_final_prob[k]
        ax.set_title(f'Removed #{k+1} | Teff={teff:.0f}K | prob={prob:.3f}', fontsize=8)
        ax.set_xlim(3800, 4500)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=6, loc='upper right')
    for j in range(len(top_removed), len(axes)):
        axes[j].axis('off')
    fig.suptitle('Top Removed U Samples (likely hidden positives)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()
else:
    print("没有样本被移出（阈值可能过高，可降低 HIGH_THRESH / MIN_PERSIST）。")
"""),

        md("""## 9. 已知 CN 星标定阈值 + 导出候选体

用阶段二的最终概率（`p2_mean`）按已知 CN 星低分位标定阈值并导出候选体，与基线做法一致。"""),

        code("""# 已知CN星标定候选阈值（基于阶段二最终概率）
known_mask = stars_clean['label'] == 1
known_scores = p2_mean[known_mask.values]
recall_quantile = 0.05

prob_threshold = float(np.nanquantile(known_scores, recall_quantile))

# 先记录阶段二最终概率
stars_clean['xgb_pu_prob'] = p2_mean
stars_clean['xgb_pu_std'] = p2_std

cand_mask = (stars_clean['label'] == -1) & (stars_clean['xgb_pu_prob'] >= prob_threshold)
candidates = stars_clean.loc[cand_mask].sort_values('xgb_pu_prob', ascending=False).copy()

n_known_above = int((known_scores >= prob_threshold).sum())
print(f"标定阈值 (保留~{(1-recall_quantile)*100:.0f}%已知CN星): prob_threshold = {prob_threshold:.4f}")
print(f"已知CN星中 >= 阈值: {n_known_above}/{len(known_scores)}")
print(f"未标注候选体数量: {len(candidates)}  ({len(candidates)/len(stars_clean)*100:.2f}%)")

# 导出
out_cols = ['uid', 'ra', 'dec', 'teff', 'logg', 'feh', 'label', 'snru',
            'xgb_pu_prob', 'xgb_pu_std', 'masked_cluster_id']
out_cols = [c for c in out_cols if c in candidates.columns]
out_df = candidates[out_cols].copy()
outpath = str(_PROJECT_ROOT / 'XGB/XGB_PU_twostage_candidates_threshold.csv')
out_df.to_csv(outpath, index=False)

print(f"候选体已导出: {outpath}")
print(f"候选体统计 (共 {len(out_df)} 颗):")
print(f"  prob > 0.5: {(out_df['xgb_pu_prob'] > 0.5).sum()}")
print(f"  prob > 0.7: {(out_df['xgb_pu_prob'] > 0.7).sum()}")
"""),

        md("""## 10. 结论

**两阶段 PU 的核心逻辑：**

1. **标签噪声是标准 PU 的主要隐患**：U 中混有大量真实但未收录的 CN 星，反复被当作负样本会压制其概率、模糊判别边界
2. **阶段一"发现"疑似正样本**：100 轮初始 Bagging 足以把"始终像 CN"的 U 样本暴露出来
3. **阶段二"净化"训练**：把这些样本从负采样池移出（不改标签），400 轮训练得到更干净的负边界
4. **如何判断是否有效**：对比 `两阶段 PU (100+400)` 与 `标准 PU (500)` 的 ROC-AUC / PR-AUC / P@50 / P@100

**调参建议：**

- 若阶段一移除的样本过多（>10% U），说明 `HIGH_THRESH` 或 `MIN_PERSIST` 过松，可能误删真实负样本
- 若移除过少（接近 0），可降低 `HIGH_THRESH`（如 0.6）或 `MIN_PERSIST`（如 0.5）
- 二者可在阶段一诊断输出中直观观察分布后微调"""),
    ]
    return nb2


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    BASE.mkdir(parents=True, exist_ok=True)

    notebooks = {
        "ML_XGB_PU_threshold.ipynb": build_baseline(),
        "ML_XGB_PU_keybands.ipynb": build_keybands(),
        "ML_XGB_PU_twostage.ipynb": build_twostage(),
    }

    for rel_path, nb_obj in notebooks.items():
        out_path = BASE / rel_path
        with open(out_path, "w", encoding="utf-8") as f:
            nbf.write(nb_obj, f)
        print(f"Created: {out_path}")

    print(f"\nAll {len(notebooks)} XGB notebooks built successfully!")
