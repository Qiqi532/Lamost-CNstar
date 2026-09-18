"""Run the three XGB PU experiments and print a comparison table.

- baseline  : full-band 700-D, standard PU Bagging T=500
- keybands  : key-band 182-D (CN3839/CN4142/CH4300), standard PU Bagging T=500
- twostage  : two-stage PU (100 rounds discovery + 400 rounds on cleaned pool)

All three share the SAME train/test split (seed=42), so the comparison is
apples-to-apples. Results are saved to XGB/comparison_results.json and the
candidate CSVs are exported to XGB/.

Run: D:/Anaconda/envs/myenv/python.exe XGB/run_comparison.py
"""

import sys
import json
import time
import random
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

from PhaseSummary.shared.data_loader import MOLECULAR_BAND_RANGES
from build_dr13_all_cache import load_dr13_all_cache

random_seed = 42

xgb_params = {
    "max_depth": 3, "learning_rate": 0.1,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 1, "gamma": 0.0,
    "reg_alpha": 0.0, "reg_lambda": 1.0,
    "seed": random_seed, "verbosity": 0, "n_jobs": 1,
}


def extract_key_bands(X, common_wave):
    band_masks = [(common_wave >= l1) & (common_wave <= l2) for (l1, l2) in MOLECULAR_BAND_RANGES]
    return np.concatenate([X[:, m] for m in band_masks], axis=1).astype(np.float32)


def metrics(y_te, te_mean):
    roc = roc_auc_score(y_te, te_mean)
    pr = average_precision_score(y_te, te_mean)
    order = np.argsort(te_mean)[::-1]
    p50 = y_te[order[:min(50, len(order))]].mean()
    p100 = y_te[order[:min(100, len(order))]].mean()
    return dict(roc=float(roc), pr=float(pr), p50=float(p50), p100=float(p100))


def run_pu_bagging(X_tr, X_te, X_all, y_tr, y_te, n_pos_tr, unl_tr_idx, T, rng, label, report_every=100):
    X_pos = X_tr[np.where(y_tr == 1)[0]]
    te_sum = np.zeros(len(y_te), dtype=np.float64)
    all_sum = np.zeros(len(y_all), dtype=np.float64)
    high_cnt = np.zeros(len(y_all), dtype=np.int64)

    t0 = time.time()
    for t in range(1, T + 1):
        neg = rng.sample(list(unl_tr_idx), n_pos_tr)
        X_bal = np.vstack([X_pos, X_tr[neg]])
        y_bal = np.hstack([np.ones(n_pos_tr), np.zeros(n_pos_tr)])
        dtrain = xgb.DMatrix(X_bal, label=y_bal)
        model = xgb.train(xgb_params, dtrain, num_boost_round=50, verbose_eval=False)
        p_te = model.predict(xgb.DMatrix(X_te))
        p_all = model.predict(xgb.DMatrix(X_all))
        te_sum += p_te
        all_sum += p_all
        high_cnt += (p_all >= 0.5).astype(np.int64)
        if t % report_every == 0:
            pm = te_sum / t
            print(f"    [{label} {t:4d}/{T}]  ROC={roc_auc_score(y_te, pm):.4f}  "
                  f"PR={average_precision_score(y_te, pm):.4f}  ({time.time()-t0:.0f}s)")
    te_mean = te_sum / T
    all_mean = all_sum / T
    all_std = np.sqrt(np.maximum(all_sum / T - all_mean ** 2, 0))
    return te_mean, all_mean, all_std, high_cnt


def threshold_candidates(stars_clean, prob, recall_quantile=0.05):
    known_mask = stars_clean["label"].values == 1
    known_scores = prob[known_mask]
    thr = float(np.nanquantile(known_scores, recall_quantile))
    cand = (stars_clean["label"].values == -1) & (prob >= thr)
    return thr, int(cand.sum()), int((known_scores >= thr).sum())


# ── Load data once ────────────────────────────────────────────────────
print("=" * 72)
print("加载数据 ...")
t0 = time.time()
data = load_dr13_all_cache()
X_clean = data["X_clean"]
stars_clean = data["stars_clean"]
common_wave = data["common_wave"]
print(f"数据加载完成 ({time.time()-t0:.0f}s)")

y_all = stars_clean["label"].map({1: 1, -1: 0}).values.astype(int)
all_idx = np.arange(len(y_all))
tv_idx, test_idx = train_test_split(
    all_idx, test_size=0.15, stratify=y_all, random_state=random_seed)
tr_idx, val_idx = train_test_split(
    tv_idx, test_size=0.18, stratify=y_all[tv_idx], random_state=random_seed)

y_tr = y_all[tr_idx]
y_te = y_all[test_idx]
n_pos_tr = int((y_tr == 1).sum())
unl_tr_idx = np.where(y_tr == 0)[0]

print(f"总样本 {len(y_all):,} | 训练 {len(tr_idx):,} (P={n_pos_tr}, U={len(unl_tr_idx):,}) | 测试 {len(test_idx):,}")
print("=" * 72)

results = {}

# ── 1) Baseline: full-band 700-D, T=500 ───────────────────────────────
print("\n[1/3] 基线 baseline (全波段 700-D, T=500)")
ss = StandardScaler()
X_all_b = ss.fit_transform(X_clean.astype(np.float32)).astype(np.float32)
rng = random.Random(random_seed)
te_b, p_b, std_b, _ = run_pu_bagging(
    X_all_b[tr_idx], X_all_b[test_idx], X_all_b, y_tr, y_te, n_pos_tr, unl_tr_idx,
    500, rng, "baseline", report_every=100)
results["baseline"] = metrics(y_te, te_b)
thr_b, ncand_b, nknown_b = threshold_candidates(stars_clean, p_b)
results["baseline"]["threshold"] = thr_b
results["baseline"]["n_candidates"] = ncand_b
results["baseline"]["n_known_above"] = nknown_b

# ── 2) Keybands: 182-D, T=500 ─────────────────────────────────────────
print("\n[2/3] 关键波段 keybands (CN3839/CN4142/CH4300, 182-D, T=500)")
X_bands = extract_key_bands(X_clean, common_wave)
print(f"    关键波段输入维度: {X_bands.shape[1]}")
ss2 = StandardScaler()
X_all_k = ss2.fit_transform(X_bands).astype(np.float32)
rng = random.Random(random_seed)
te_k, p_k, std_k, _ = run_pu_bagging(
    X_all_k[tr_idx], X_all_k[test_idx], X_all_k, y_tr, y_te, n_pos_tr, unl_tr_idx,
    500, rng, "keybands", report_every=100)
results["keybands"] = metrics(y_te, te_k)
thr_k, ncand_k, nknown_k = threshold_candidates(stars_clean, p_k)
results["keybands"]["threshold"] = thr_k
results["keybands"]["n_candidates"] = ncand_k
results["keybands"]["n_known_above"] = nknown_k

# ── 3) Two-stage: T1=100 + T2=400 ─────────────────────────────────────
print("\n[3/3] 两阶段 two-stage (T1=100 发现 + T2=400 净化后训练)")
T1, T2 = 100, 400
HIGH_THRESH, MIN_PERSIST = 0.8, 0.8

ss3 = StandardScaler()
X_all_t = ss3.fit_transform(X_clean.astype(np.float32)).astype(np.float32)
X_tr_t = X_all_t[tr_idx]
X_te_t = X_all_t[test_idx]

rng2 = random.Random(random_seed)
te1, p1_mean, p1_std, p1_high = run_pu_bagging(
    X_tr_t, X_te_t, X_all_t, y_tr, y_te, n_pos_tr, unl_tr_idx,
    T1, rng2, "stage1", report_every=50)

unl_full_idx = tr_idx[unl_tr_idx]
p1_u = p1_mean[unl_full_idx]
persist_u = p1_high[unl_full_idx] / T1
relabel = (p1_u >= HIGH_THRESH) & (persist_u >= MIN_PERSIST)
removed_full = unl_full_idx[relabel]
clean_pool = unl_tr_idx[~relabel]
print(f"    阶段一移除疑似正样本: {int(relabel.sum()):,} / {len(unl_tr_idx):,} "
      f"({int(relabel.sum())/len(unl_tr_idx)*100:.2f}%)")

te2, p2_mean, p2_std, _ = run_pu_bagging(
    X_tr_t, X_te_t, X_all_t, y_tr, y_te, n_pos_tr, clean_pool,
    T2, rng2, "stage2", report_every=100)

results["twostage"] = metrics(y_te, te2)
results["twostage"]["n_removed"] = int(relabel.sum())
thr_t, ncand_t, nknown_t = threshold_candidates(stars_clean, p2_mean)
results["twostage"]["threshold"] = thr_t
results["twostage"]["n_candidates"] = ncand_t
results["twostage"]["n_known_above"] = nknown_t

# ── Save candidate CSVs ───────────────────────────────────────────────
out_cols = ["uid", "ra", "dec", "teff", "logg", "feh", "label", "snru",
            "masked_cluster_id"]
out_cols = [c for c in out_cols if c in stars_clean.columns]

def export(prob, thr, fname):
    stars_clean["_p"] = prob
    df = stars_clean[(stars_clean["label"] == -1) & (stars_clean["_p"] >= thr)].copy()
    df = df.sort_values("_p", ascending=False)
    df = df[[c for c in (out_cols + ["_p"]) if c in df.columns]]
    df = df.rename(columns={"_p": "xgb_pu_prob"})
    df.to_csv(_PROJECT_ROOT / "XGB" / fname, index=False)
    return len(df)

n1 = export(p_b, thr_b, "XGB_PU_candidates_threshold.csv")
n2 = export(p_k, thr_k, "XGB_PU_keybands_candidates_threshold.csv")
n3 = export(p2_mean, thr_t, "XGB_PU_twostage_candidates_threshold.csv")

# ── Final comparison table ────────────────────────────────────────────
print("\n" + "=" * 72)
print("结果对比")
print("=" * 72)
rows = []
for name, label in [("baseline", "标准 PU (全波段 700-D)"),
                    ("keybands", "关键波段 (182-D)"),
                    ("twostage", "两阶段 PU (100+400)")]:
    r = results[name]
    rows.append({
        "方法": label,
        "ROC-AUC": f"{r['roc']:.4f}",
        "PR-AUC": f"{r['pr']:.4f}",
        "P@50": f"{r['p50']:.4f}",
        "P@100": f"{r['p100']:.4f}",
        "阈值(q05)": f"{r['threshold']:.4f}",
        "候选数": r["n_candidates"],
    })
comp = pd.DataFrame(rows)
print(comp.to_string(index=False))

print()
print(f"候选体已导出到 XGB/:")
print(f"  基线    XGB_PU_candidates_threshold.csv          -> {n1} 颗")
print(f"  关键波段 XGB_PU_keybands_candidates_threshold.csv -> {n2} 颗")
print(f"  两阶段  XGB_PU_twostage_candidates_threshold.csv  -> {n3} 颗")

with open(_PROJECT_ROOT / "XGB" / "comparison_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n指标已保存: XGB/comparison_results.json")

print("\n完成。")
