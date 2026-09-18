"""XGBoost PU Bagging 的物理参数重要性采样模块。

背景
----
标准 PU Bagging 每次从**全部**未标注集 U 中**等概率**采样负样本。若正样本（已知 CN 星）
与未标注集在物理参数（teff / logg / feh）上的分布存在系统性差异，模型可能学会"用物理
参数区分正负"这条捷径，而不是真正依赖 CN 分子带形态。观测结果也印证了这一点：
[Fe/H] > -1.2 的候选星中有相当一部分 N 丰度较低的"误判"，而 [Fe/H] 更贫的部分识别更好。

本模块提供两种重要性采样方案，让负样本在物理参数空间尽量"对齐"正样本分布，迫使模型
关注 CN 分子带本身：

1. **三维分箱分层匹配近邻采样** (`build_match_candidates` + `MatchSampler`)
   - 在 (teff, logg, feh) 标准化空间做 3D 分位分箱（分层）
   - 每个正样本只在其"同箱"内取 K 个**最近邻** U 作为候选（分箱分层 + 近邻匹配），
     箱内 U 不足 K 个时用全局 KDTree 最近邻补齐
   - 每轮 bag 每个正样本从自己的候选池采 1 个负样本 → 1:1 物理参数对齐

2. **KDE 核密度加权抽样** (`build_kde_weights` + `WeightedSampler`)
   - 对正样本与 U 池分别拟合 3D 高斯核密度 f_pos / f_unl
   - 以重要性权重 w = f_pos / f_unl 对 U 加权采样，抬高"正样本密度相对更高"区域的采样概率

其余（XGB 参数、T 轮数、划分、阈值标定）与基线 `ML_XGB_PU_threshold` 完全一致，保证可比。
"""

import sys
import time
import random
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Lamost/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

from build_dr13_all_cache import load_dr13_all_cache

random_seed = 42
PARAM_COLS = ["teff", "logg", "feh"]  # 三维分箱 / KDE 使用的物理参数

xgb_params = {
    "max_depth": 3, "learning_rate": 0.1,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 1, "gamma": 0.0,
    "reg_alpha": 0.0, "reg_lambda": 1.0,
    "seed": random_seed, "verbosity": 0, "n_jobs": 1,
}


# ──────────────────────────────────────────────────────────────────────
# 数据加载与划分（与基线完全一致）
# ──────────────────────────────────────────────────────────────────────

def load_and_split():
    """加载缓存并做与基线一致的分层划分，返回一个 dict。"""
    data = load_dr13_all_cache()
    X_clean = data["X_clean"]
    stars_clean = data["stars_clean"]
    common_wave = data["common_wave"]

    y_all = stars_clean["label"].map({1: 1, -1: 0}).values.astype(int)
    all_idx = np.arange(len(y_all))
    tv_idx, test_idx = train_test_split(
        all_idx, test_size=0.15, stratify=y_all, random_state=random_seed)
    tr_idx, val_idx = train_test_split(
        tv_idx, test_size=0.18, stratify=y_all[tv_idx], random_state=random_seed)

    ss = StandardScaler()
    X_spec_scaled = ss.fit_transform(X_clean.astype(np.float32)).astype(np.float32)

    X_tr = X_spec_scaled[tr_idx]
    X_te = X_spec_scaled[test_idx]
    y_tr = y_all[tr_idx]
    y_te = y_all[test_idx]

    pos_tr_idx = np.where(y_tr == 1)[0]   # 训练矩阵内的行号
    n_pos_tr = int(len(pos_tr_idx))
    unl_tr_idx = np.where(y_tr == 0)[0]   # 训练矩阵内的行号（负采样池）

    return {
        "data": data,
        "X_clean": X_clean,
        "stars_clean": stars_clean,
        "common_wave": common_wave,
        "y_all": y_all,
        "tr_idx": tr_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "X_spec_scaled": X_spec_scaled,
        "X_tr": X_tr,
        "X_te": X_te,
        "y_tr": y_tr,
        "y_te": y_te,
        "pos_tr_idx": pos_tr_idx,
        "n_pos_tr": n_pos_tr,
        "unl_tr_idx": unl_tr_idx,
    }


def standardize_physics(stars, cols=PARAM_COLS):
    """把 (teff, logg, feh) 三列 z-score 标准化，返回 (Ps, mu, sd)。"""
    P = stars[list(cols)].to_numpy(dtype=np.float64)
    mu = np.nanmean(P, axis=0)
    sd = np.nanstd(P, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Ps = (P - mu) / sd
    return Ps, mu, sd


# ──────────────────────────────────────────────────────────────────────
# 采样器（统一 .sample(n) 接口，返回训练矩阵内的行号数组）
# ──────────────────────────────────────────────────────────────────────

class UniformSampler:
    """基线：从负采样池等概率无放回采样。"""

    def __init__(self, unl_tr_idx, seed=random_seed):
        self.unl = list(unl_tr_idx)
        self.rng = random.Random(seed)

    def sample(self, n):
        return np.array(self.rng.sample(self.unl, n), dtype=int)


class MatchSampler:
    """三维分箱分层匹配近邻采样：每个正样本从自己的候选池采 1 个负样本。

    candidates[i] 是第 i 个正样本可匹配的负样本（训练矩阵内行号）。
    """

    def __init__(self, candidates, seed=random_seed):
        self.candidates = candidates
        self.rng = random.Random(seed)
        self.n_pos = len(candidates)

    def sample(self, n):
        idx = np.empty(self.n_pos, dtype=int)
        for i, c in enumerate(self.candidates):
            idx[i] = self.rng.choice(c)
        return idx


class WeightedSampler:
    """KDE 重要性加权抽样：按权重 w 从负采样池无放回采样。"""

    def __init__(self, unl_tr_idx, weights, seed=random_seed):
        self.unl = np.asarray(unl_tr_idx, dtype=int)
        self.w = np.asarray(weights, dtype=np.float64)
        self.w = self.w / self.w.sum()
        self.rng = np.random.default_rng(seed)

    def sample(self, n):
        sel = self.rng.choice(len(self.unl), size=n, replace=False, p=self.w)
        return self.unl[sel]


# ──────────────────────────────────────────────────────────────────────
# 重要性采样的预处理构造
# ──────────────────────────────────────────────────────────────────────

def build_match_candidates(phys_pos, phys_unl, unl_idx, B=5, K=5):
    """三维分箱分层匹配近邻采样的候选池构造。

    参数
    ----
    phys_pos : (n_pos, D) 标准化物理参数（训练正样本）
    phys_unl : (n_unl, D) 标准化物理参数（训练 U）
    unl_idx  : (n_unl,)   U 在训练矩阵内的行号
    B        : 每维分位箱数（D 维共 B^D 个箱）
    K        : 每个正样本在同箱内取最近邻的个数（箱内不足 K 个时全局 KDTree 补齐）

    返回
    ----
    candidates : list[n_pos]，每个元素是该正样本可匹配的负样本行号数组
    n_cell     : 有多少个正样本"仅靠同箱"即获得足够近邻（分层命中数）
    """
    n_pos = len(phys_pos)
    D = phys_pos.shape[1]

    # 用 U 池的分位边界定义 D 维箱（正负共享同一套边界）
    edges = [np.quantile(phys_unl[:, d], np.linspace(0, 1, B + 1)) for d in range(D)]
    for e in edges:
        e[0] = -np.inf
        e[-1] = np.inf

    def cellid(P):
        cid = np.zeros(len(P), dtype=int)
        for d in range(D):
            cid = cid * B + np.digitize(P[:, d], edges[d][1:-1])
        return cid

    cell_unl = cellid(phys_unl)
    cell_pos = cellid(phys_pos)

    tree = cKDTree(phys_unl)
    candidates = []
    n_cell = 0
    for i in range(n_pos):
        cid = cell_pos[i]
        in_cell = np.where(cell_unl == cid)[0]
        if len(in_cell) >= K:
            # 同箱内取 K 个最近邻 U（分箱分层 + 近邻匹配）
            d = np.linalg.norm(phys_unl[in_cell] - phys_pos[i], axis=1)
            sel = in_cell[np.argsort(d)[:K]]
            cand = unl_idx[sel]
            n_cell += 1
        else:
            # 同箱内 U 不足 K 个 → 用全局最近邻补齐（分层 + 近邻兜底）
            _, nn = tree.query(phys_pos[i], k=K)
            nn = np.atleast_1d(nn)
            cand = unl_idx[np.unique(np.concatenate([in_cell, nn]))]
        candidates.append(np.asarray(cand, dtype=int))

    return candidates, n_cell


def build_kde_weights(phys_pos, phys_unl, unl_idx):
    """KDE 核密度估计重要性权重。

    对正样本与 U 池分别拟合 3D 高斯核密度 f_pos / f_unl，权重 w = f_pos / f_unl，
    再归一化。返回 (weights, ess)，ess 为有效样本量 1/Σw²（越小说明权重越集中）。
    """
    n_pos = len(phys_pos)
    kde_pos = gaussian_kde(phys_pos.T)
    kde_unl = gaussian_kde(phys_unl.T)
    f_pos = np.clip(kde_pos(phys_unl.T), 1e-12, None)
    f_unl = np.clip(kde_unl(phys_unl.T), 1e-12, None)
    w = f_pos / f_unl
    w = w / w.sum()
    ess = 1.0 / float(np.sum(w ** 2))
    return w, ess


# ──────────────────────────────────────────────────────────────────────
# PU Bagging 训练 + 评估
# ──────────────────────────────────────────────────────────────────────

def run_pu_bagging(X_tr, X_te, X_all, y_tr, y_te, n_pos_tr, sampler, T,
                   label="", report_every=100):
    """在给定采样器下跑 T 轮 PU Bagging，返回 (te_mean, all_mean)。"""
    pos_tr_idx = np.where(y_tr == 1)[0]
    X_pos = X_tr[pos_tr_idx]
    te_sum = np.zeros(len(y_te), dtype=np.float64)
    all_sum = np.zeros(len(X_all), dtype=np.float64)

    t0 = time.time()
    for t in range(1, T + 1):
        neg = sampler.sample(n_pos_tr)
        X_bal = np.vstack([X_pos, X_tr[neg]])
        y_bal = np.hstack([np.ones(n_pos_tr), np.zeros(n_pos_tr)])

        dtrain = xgb.DMatrix(X_bal, label=y_bal)
        model = xgb.train(xgb_params, dtrain, num_boost_round=50, verbose_eval=False)

        te_sum += model.predict(xgb.DMatrix(X_te))
        all_sum += model.predict(xgb.DMatrix(X_all))

        if t % report_every == 0:
            pm = te_sum / t
            print(f"    [{label} {t:4d}/{T}]  ROC={roc_auc_score(y_te, pm):.4f}  "
                  f"PR={average_precision_score(y_te, pm):.4f}  ({time.time()-t0:.0f}s)")

    return te_sum / T, all_sum / T


def metrics(y_te, te_mean):
    """测试集 ROC-AUC / PR-AUC / P@50 / P@100。"""
    roc = roc_auc_score(y_te, te_mean)
    pr = average_precision_score(y_te, te_mean)
    order = np.argsort(te_mean)[::-1]
    p50 = y_te[order[:min(50, len(order))]].mean()
    p100 = y_te[order[:min(100, len(order))]].mean()
    return dict(roc=float(roc), pr=float(pr), p50=float(p50), p100=float(p100))


def threshold_candidates(stars_clean, prob, recall_quantile=0.05):
    """已知 CN 星低分位标定阈值，返回 (thr, n_candidates, n_known_above)。"""
    known_mask = stars_clean["label"].values == 1
    known_scores = prob[known_mask]
    thr = float(np.nanquantile(known_scores, recall_quantile))
    cand = (stars_clean["label"].values == -1) & (prob >= thr)
    return thr, int(cand.sum()), int((known_scores >= thr).sum())


def export_candidates(stars_clean, prob, thr, outpath, prob_name="xgb_pu_prob"):
    """按阈值导出未标注候选体 CSV。"""
    out_cols = ["uid", "ra", "dec", "teff", "logg", "feh", "label", "snru",
                "masked_cluster_id"]
    out_cols = [c for c in out_cols if c in stars_clean.columns]
    stars_clean = stars_clean.copy()
    stars_clean["_p"] = prob
    df = stars_clean[(stars_clean["label"] == -1) & (stars_clean["_p"] >= thr)].copy()
    df = df.sort_values("_p", ascending=False)
    df = df[[c for c in (out_cols + ["_p"]) if c in df.columns]]
    df = df.rename(columns={"_p": prob_name})
    df.to_csv(outpath, index=False)
    return len(df)
