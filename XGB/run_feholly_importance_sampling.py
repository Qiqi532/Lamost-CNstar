"""feh-only 重要性采样消融实验：检验"是否只是 feh 的影响"。

在 3D（teff/logg/feh）重要性采样的基础上，额外只把 feh 作为匹配/加权变量，
重复两种采样方案：
    - feh_matched : 一维 feh 分箱分层匹配近邻采样
    - feh_kde     : 一维 feh KDE 密度比加权抽样

并把结果与已保存的 baseline / 3D matched / 3D kde（importance_sampling_results.json）
合并成一张五路对比表。若 feh-only 几乎复现 3D 的效果，说明 [Fe/H] 是主导混淆变量；
若明显更弱，说明 teff/logg 也贡献了不可忽略的混淆。

Run: D:/Anaconda/envs/myenv/python.exe XGB/run_feholly_importance_sampling.py
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
B, K = 5, 10


def feh_breakdown(stars_clean, prob, thr):
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
    print(f"总样本 {len(S['y_all']):,} | 训练 {len(tr_idx):,} (P={n_pos_tr}, U={len(unl_tr_idx):,})")

    # ── feh-only 标准化 ──
    phys1, mu1, sd1 = imp.standardize_physics(stars_clean, cols=["feh"])
    phys1_tr = phys1[tr_idx]
    phys1_pos = phys1_tr[np.where(y_tr == 1)[0]]
    phys1_unl = phys1_tr[unl_tr_idx]
    print(f"feh-only 标准化: mu={mu1[0]:.3f}, sd={sd1[0]:.3f}  (正样本 {phys1_pos.shape}, U {phys1_unl.shape})")

    samplers = {}
    # feh-only matched
    cands_f, n_cell_f = imp.build_match_candidates(phys1_pos, phys1_unl, unl_tr_idx, B=B, K=K)
    print(f"\n[feh_matched] 一维 feh 分箱 B={B}, 近邻 K={K}: 分层命中 {n_cell_f}/{len(cands_f)}, "
          f"候选池大小 {len(np.unique(np.concatenate([np.atleast_1d(c) for c in cands_f]))):,}")
    samplers["feh_matched"] = (imp.MatchSampler(cands_f, random_seed), "feh-only 分箱匹配近邻")

    # feh-only kde
    w_f, ess_f = imp.build_kde_weights(phys1_pos, phys1_unl, unl_tr_idx)
    print(f"[feh_kde] 一维 feh KDE: max/median={np.max(w_f)/np.median(w_f):.1f}, ESS={ess_f:.0f}/{len(unl_tr_idx):,}")
    samplers["feh_kde"] = (imp.WeightedSampler(unl_tr_idx, w_f, random_seed), "feh-only KDE 加权")

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

    # ── 导出 feh-only 候选体 ──
    outdir = _PROJECT_ROOT / "XGB"
    export_name = {
        "feh_matched": "XGB_PU_feholly_matched_candidates_threshold.csv",
        "feh_kde": "XGB_PU_feholly_kde_candidates_threshold.csv",
    }
    for name in samplers:
        thr = results[name]["threshold"]
        n = imp.export_candidates(stars_clean, probs[name], thr,
                                  str(outdir / export_name[name]))
        print(f"导出: {export_name[name]} -> {n} 颗")

    # ── 合并已保存的 baseline / 3D 结果 ──
    saved = json.loads((outdir / "importance_sampling_results.json").read_text(encoding="utf-8"))
    merged = dict(saved["results"])
    merged["feh_matched"] = results["feh_matched"]
    merged["feh_kde"] = results["feh_kde"]

    print("\n" + "=" * 72)
    print("五路对比（等概率 / 3D / feh-only）")
    print("=" * 72)
    label_map = {
        "baseline": "标准 PU (等概率)",
        "matched": "3D 分箱匹配近邻",
        "kde": "3D KDE 加权",
        "feh_matched": "feh-only 分箱匹配近邻",
        "feh_kde": "feh-only KDE 加权",
    }
    rows = []
    for name in ["baseline", "matched", "feh_matched", "kde", "feh_kde"]:
        r = merged[name]
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
            "中位feh": f"{r['feh']['median_feh']:.3f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ── 候选体重叠（uid 级）：feh-only vs 3D 是否一致 ──
    print("\n候选体重叠（uid 级）:")
    def uid_set(csv):
        p = outdir / csv
        if not p.exists():
            return set()
        return set(pd.read_csv(p)["uid"].astype(str).tolist())

    def cur_uid(prob, thr):
        cand = (stars_clean["label"].values == -1) & (prob >= thr)
        return set(stars_clean["uid"].values[cand].astype(str).tolist())

    s_fehm = cur_uid(probs["feh_matched"], results["feh_matched"]["threshold"])
    s_fehk = cur_uid(probs["feh_kde"], results["feh_kde"]["threshold"])
    s_3dm = uid_set("XGB_PU_matched_candidates_threshold.csv")
    s_3dk = uid_set("XGB_PU_kde_candidates_threshold.csv")
    s_base = uid_set("XGB_PU_importance_baseline_candidates_threshold.csv")

    def report_overlap(a, b, na, nb):
        inter = len(a & b)
        print(f"  {na} ∩ {nb}: {inter}  (占 {na} {inter/len(a)*100:.1f}% / 占 {nb} {inter/len(b)*100:.1f}%)")

    report_overlap(s_fehm, s_3dm, "feh_matched", "3D_matched")
    report_overlap(s_fehk, s_3dk, "feh_kde", "3D_kde")
    report_overlap(s_fehm, s_fehk, "feh_matched", "feh_kde")
    report_overlap(s_fehm, s_base, "feh_matched", "baseline")

    # ── 保存 ──
    out = {
        "fehonly_results": {"feh_matched": results["feh_matched"], "feh_kde": results["feh_kde"]},
        "params": {"T": T, "B": B, "K": K, "seed": random_seed},
        "feh_mu": float(mu1[0]), "feh_sd": float(sd1[0]),
        "overlaps": {
            "feh_matched_vs_3d_matched": len(s_fehm & s_3dm),
            "feh_kde_vs_3d_kde": len(s_fehk & s_3dk),
            "feh_matched_vs_feh_kde": len(s_fehm & s_fehk),
            "feh_matched_vs_baseline": len(s_fehm & s_base),
        },
    }
    with open(outdir / "importance_sampling_fehonly_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n指标已保存: XGB/importance_sampling_fehonly_results.json")
    print(f"总耗时 {time.time()-t0:.0f}s。完成。")


if __name__ == "__main__":
    main()
