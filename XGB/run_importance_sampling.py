"""运行三种采样方式下的 XGBoost PU Bagging 并输出对比。

- baseline : 标准 PU（U 池等概率采样），T=500
- matched  : 三维分箱分层匹配近邻采样，T=500
- kde      : KDE 核密度加权抽样，T=500

三者共享同一数据划分（seed=42），只改变"负样本采样方式"，其余完全一致。
结果保存到 XGB/importance_sampling_results.json，候选体 CSV 导出到 XGB/。

Run: D:/Anaconda/envs/myenv/python.exe XGB/run_importance_sampling.py
"""

import sys
import json
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

import importance_sampling as imp

random_seed = imp.random_seed
T = 500
B, K = 5, 10  # 三维分箱每维箱数 / 同箱内最近邻个数


def feh_breakdown(stars_clean, prob, thr):
    """候选体的 [Fe/H] 分布分解（用于直接检验 feh > -1.2 误判区）。"""
    cand = (stars_clean["label"].values == -1) & (prob >= thr)
    feh = stars_clean["feh"].values[cand]
    return {
        "n": int(cand.sum()),
        "n_feh_gt_-1.2": int((feh > -1.2).sum()),
        "n_feh_le_-1.2": int((feh <= -1.2).sum()),
        "median_feh": float(np.median(feh)) if len(feh) else None,
    }


def main():
    t0 = time.time()
    print("=" * 72)
    print("加载数据 + 划分 ...")
    S = imp.load_and_split()
    X_clean = S["X_clean"]
    stars_clean = S["stars_clean"]
    X_all = S["X_spec_scaled"]
    X_tr = S["X_tr"]
    X_te = S["X_te"]
    y_tr = S["y_tr"]
    y_te = S["y_te"]
    n_pos_tr = S["n_pos_tr"]
    unl_tr_idx = S["unl_tr_idx"]
    tr_idx = S["tr_idx"]
    print(f"总样本 {len(S['y_all']):,} | 训练 {len(tr_idx):,} (P={n_pos_tr}, U={len(unl_tr_idx):,}) "
          f"| 测试 {len(S['test_idx']):,}")
    print(f"数据准备完成 ({time.time()-t0:.0f}s)")

    # 物理参数标准化（teff/logg/feh，全样本统计量）
    phys_all, phys_mu, phys_sd = imp.standardize_physics(stars_clean)
    phys_tr = phys_all[tr_idx]
    phys_pos = phys_tr[np.where(y_tr == 1)[0]]   # 训练正样本
    phys_unl = phys_tr[unl_tr_idx]               # 训练 U
    print(f"标准化物理参数: 正样本 {phys_pos.shape}, U {phys_unl.shape}")
    print(f"  物理参数均值(teff/logg/feh): {phys_mu.round(1)}  标准差: {phys_sd.round(2)}")

    samplers = {}
    # ── baseline: 等概率 ──
    samplers["baseline"] = (imp.UniformSampler(unl_tr_idx, random_seed), "标准 PU (等概率采样)")

    # ── matched: 三维分箱分层匹配近邻 ──
    cands, n_cell = imp.build_match_candidates(phys_pos, phys_unl, unl_tr_idx, B=B, K=K)
    sizes = np.array([len(c) for c in cands])
    print(f"\n[匹配采样] 分箱 B={B}, 近邻兜底 K={K}")
    print(f"  仅靠同箱即满足的分层命中: {n_cell}/{len(cands)}")
    print(f"  每正样本候选池大小: min={sizes.min()}  med={np.median(sizes):.0f}  max={sizes.max()}")
    samplers["matched"] = (imp.MatchSampler(cands, random_seed), "三维分箱分层匹配近邻采样")

    # ── kde: KDE 加权 ──
    w, ess = imp.build_kde_weights(phys_pos, phys_unl, unl_tr_idx)
    print(f"\n[KDE 加权] 权重 max/median = {np.max(w)/np.median(w):.1f}, "
          f"有效样本量 ESS = {ess:.0f} / {len(unl_tr_idx):,}")
    samplers["kde"] = (imp.WeightedSampler(unl_tr_idx, w, random_seed), "KDE 核密度加权抽样")

    # ── 运行 ──
    results = {}
    probs = {}
    for name, (sampler, label) in samplers.items():
        print(f"\n{'='*72}")
        print(f"[{name}] {label}  (T={T})")
        te_mean, all_mean = imp.run_pu_bagging(
            X_tr, X_te, X_all, y_tr, y_te, n_pos_tr, sampler, T, label=name, report_every=100)
        m = imp.metrics(y_te, te_mean)
        thr, ncand, nknown = imp.threshold_candidates(stars_clean, all_mean)
        m["threshold"] = thr
        m["n_candidates"] = ncand
        m["n_known_above"] = nknown
        m["feh"] = feh_breakdown(stars_clean, all_mean, thr)
        results[name] = m
        probs[name] = all_mean
        print(f"    ROC={m['roc']:.4f}  PR={m['pr']:.4f}  P@50={m['p50']:.4f}  "
              f"P@100={m['p100']:.4f}  thr={thr:.4f}  cand={ncand}")
        print(f"    候选 feh 分解: >-1.2 = {m['feh']['n_feh_gt_-1.2']}, "
              f"<=-1.2 = {m['feh']['n_feh_le_-1.2']}, 中位 feh = {m['feh']['median_feh']}")

    # ── 导出候选体 CSV ──
    outdir = _PROJECT_ROOT / "XGB"
    export_name = {
        "baseline": "XGB_PU_importance_baseline_candidates_threshold.csv",
        "matched": "XGB_PU_matched_candidates_threshold.csv",
        "kde": "XGB_PU_kde_candidates_threshold.csv",
    }
    for name in samplers:
        thr = results[name]["threshold"]
        n = imp.export_candidates(stars_clean, probs[name], thr,
                                  str(outdir / export_name[name]))
        print(f"\n导出: {export_name[name]} -> {n} 颗")

    # ── 对比表 ──
    print("\n" + "=" * 72)
    print("采样方式对比")
    print("=" * 72)
    rows = []
    label_map = {"baseline": "标准 PU (等概率)", "matched": "三维分箱分层匹配近邻",
                 "kde": "KDE 核密度加权"}
    for name in ["baseline", "matched", "kde"]:
        r = results[name]
        rows.append({
            "采样方式": label_map[name],
            "ROC-AUC": f"{r['roc']:.4f}",
            "PR-AUC": f"{r['pr']:.4f}",
            "P@50": f"{r['p50']:.4f}",
            "P@100": f"{r['p100']:.4f}",
            "阈值(q05)": f"{r['threshold']:.4f}",
            "候选数": r["n_candidates"],
            "feh>-1.2": r["feh"]["n_feh_gt_-1.2"],
            "feh<=-1.2": r["feh"]["n_feh_le_-1.2"],
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ── 候选体重叠 ──
    print("\n候选体重叠（Jaccard 语义：交集大小）")
    def cand_set(prob, thr):
        return set(np.where((stars_clean["label"].values == -1) & (prob >= thr))[0].tolist())
    sets = {name: cand_set(probs[name], results[name]["threshold"]) for name in samplers}
    for a, b in [("baseline", "matched"), ("baseline", "kde"), ("matched", "kde")]:
        inter = len(sets[a] & sets[b])
        print(f"  {a} ∩ {b}: {inter}  (占 {a} {inter/len(sets[a])*100:.1f}% / "
              f"占 {b} {inter/len(sets[b])*100:.1f}%)")
    inter3 = len(sets["baseline"] & sets["matched"] & sets["kde"])
    print(f"  三者共同: {inter3}")

    # ── 保存 JSON ──
    out = {"results": results, "params": {"T": T, "B": B, "K": K, "seed": random_seed},
           "phys_mu": phys_mu.tolist(), "phys_sd": phys_sd.tolist(),
           "overlaps": {
               "baseline_matched": len(sets["baseline"] & sets["matched"]),
               "baseline_kde": len(sets["baseline"] & sets["kde"]),
               "matched_kde": len(sets["matched"] & sets["kde"]),
               "all_three": inter3,
           }}
    with open(outdir / "importance_sampling_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n指标已保存: XGB/importance_sampling_results.json")
    print(f"总耗时 {time.time()-t0:.0f}s。完成。")


if __name__ == "__main__":
    main()
