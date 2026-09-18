"""Run LabelSpreading analysis end-to-end."""
import sys, os, time, warnings
sys.path.insert(0, 'd:/资料/2026大创/Lamost')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

from LabelSpreading.graph_utils import (
    build_knn_graph, build_graph_from_cache, run_label_spreading,
    evaluate_strategy, run_all_strategies, compute_cluster_zscore, export_candidates,
)
from PhaseSummary.shared.data_loader import ensure_cache
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJ = 'd:/资料/2026大创/Lamost'
OUT = f'{PROJ}/LabelSpreading'

print("="*60)
print("Label Spreading — Full Pipeline")
print("="*60)

# 1. Load data
data = ensure_cache()
X_clean = data['X_clean']
stars_clean = data['stars_clean']
feature_df = data['feature_df']
common_wave = data['common_wave']

known_mask = stars_clean['label'].values == 1
unl_mask = stars_clean['label'].values == -1
n_known = known_mask.sum()
n_unl = unl_mask.sum()

print(f"Data: {X_clean.shape[0]:,} stars, {n_known} known CN, {n_unl:,} unlabeled")

# 2. Clustering validation
print("\n--- Clustering Validation ---")
y_labeled = stars_clean['label'].map({1: 1, -1: -1}).values.astype(int)
W_quick = build_knn_graph(X_clean, n_neighbors=50, metric='cosine', standardize=False)
prob_quick, _ = run_label_spreading(W_quick, y_labeled, alpha=0.5)

z_masked = compute_cluster_zscore(prob_quick, stars_clean['masked_cluster_id'].values)

X_param = StandardScaler().fit_transform(stars_clean[['teff', 'logg', 'feh']].values)
param_clusters = KMeans(n_clusters=45, random_state=42, n_init=10).fit_predict(X_param)
z_param = compute_cluster_zscore(prob_quick, param_clusters)

print(f"Masked-band: known CN mean z={z_masked[known_mask].mean():.2f}, {100*(z_masked[unl_mask]>2).sum()/n_unl:.2f}% unl >2")
print(f"Param-only:  known CN mean z={z_param[known_mask].mean():.2f}, {100*(z_param[unl_mask]>2).sum()/n_unl:.2f}% unl >2")

# Plot clustering validation
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
bins = np.linspace(-2, 10, 60)
ax.hist(z_masked[known_mask], bins=bins, alpha=0.6, color='#2ecc71', label=f'Masked-band (mean={z_masked[known_mask].mean():.2f})')
ax.hist(z_param[known_mask], bins=bins, alpha=0.6, color='#e74c3c', label=f'Param-only (mean={z_param[known_mask].mean():.2f})')
ax.axvline(2.0, color='gray', linestyle='--', alpha=0.7, label='z=2.0')
ax.set_xlabel('Cluster z-score'); ax.set_ylabel('Known CN count')
ax.set_title('Known CN Stars: z-score Under Two Clusterings')
ax.legend(fontsize=8); ax.grid(alpha=0.2)

ax = axes[1]
methods = ['Masked-band', 'Param-only']
high_z_pct = [100*(z_masked[unl_mask]>2).sum()/n_unl, 100*(z_param[unl_mask]>2).sum()/n_unl]
high_z_known = [100*(z_masked[known_mask]>2).sum()/n_known, 100*(z_param[known_mask]>2).sum()/n_known]
x = np.arange(len(methods)); w = 0.35
ax.bar(x-w/2, high_z_pct, w, label='Unlabeled (z>2.0 %)', color='steelblue', edgecolor='white')
ax.bar(x+w/2, high_z_known, w, label='Known CN (z>2.0 %)', color='#e74c3c', edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(methods)
ax.set_ylabel('% with z > 2.0'); ax.set_title('High-z Fraction by Clustering Method')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.2)
fig.suptitle('Clustering Validation: Masked-Band vs Param-Only', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/clustering_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ clustering_validation.png saved")

# 3. Run all strategies
print("\n--- Running All Strategies ---")
K_VALUES = [30, 50, 100]
t_start = time.time()
all_outputs = run_all_strategies(
    X_clean=X_clean,
    feature_df=feature_df,
    stars_clean=stars_clean,
    k_values=K_VALUES,
    alpha=0.5,
    verbose=True,
)
results = all_outputs['results']
probs = all_outputs['probs']
elapsed = time.time() - t_start
print(f"All strategies done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

# 4. Strategy comparison table
results_df = pd.DataFrame(results).sort_values('precision@100', ascending=False)
print(f"\nBest strategy: {results_df.iloc[0]['name']}")
print(f"  Precision@50:  {results_df.iloc[0]['precision@50']:.4f}")
print(f"  Precision@100: {results_df.iloc[0]['precision@100']:.4f}")
print(f"  Prob ratio:    {results_df.iloc[0]['prob_ratio']:.2f}")

# Plot strategy comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
strat_top = results_df.head(8)
x = np.arange(len(strat_top)); w = 0.25

ax = axes[0]
ax.bar(x-w, strat_top['precision@50'], w, label='P@50', color='#3498db', edgecolor='white')
ax.bar(x, strat_top['precision@100'], w, label='P@100', color='#2ecc71', edgecolor='white')
ax.bar(x+w, strat_top['precision@200'], w, label='P@200', color='#e74c3c', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels([s.replace('_','\n') for s in strat_top['name']], rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Precision'); ax.set_title('Precision@K by Strategy')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)

ax = axes[1]
colors = ['#3498db' if 'spectra' in n else '#2ecc71' if 'physics' in n else '#f39c12' if 'delta' in n else '#9b59b6' for n in strat_top['name']]
ax.barh(strat_top['name'].str.replace('_',' '), strat_top['prob_ratio'], color=colors, edgecolor='white')
ax.axvline(1.0, color='gray', linestyle='--')
ax.set_xlabel('Prob Ratio (known CN / all median)'); ax.set_title('Signal-to-Background')

ax = axes[2]
for _, row in strat_top.iterrows():
    c = '#3498db' if 'spectra' in row['name'] else '#2ecc71' if 'physics' in row['name'] else '#f39c12' if 'delta' in row['name'] else '#9b59b6'
    ax.scatter(row['pseudo_pr_auc'], row['precision@100'], s=100, c=c, edgecolors='black', linewidth=0.5, zorder=3)
    ax.annotate(row['name'].replace('_','\n'), (row['pseudo_pr_auc'], row['precision@100']),
                fontsize=5.5, ha='center', va='bottom', textcoords='offset points', xytext=(0, 6))
ax.set_xlabel('Pseudo PR-AUC'); ax.set_ylabel('Precision@100')
ax.set_title('PR-AUC vs Precision@100'); ax.grid(alpha=0.3)

fig.suptitle('Label Spreading — Strategy Comparison', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/strategy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ strategy_comparison.png saved")

# 5. Best strategy z-score
best_name = results_df.iloc[0]['name']
prob_best = probs[best_name]
cluster_ids = stars_clean['masked_cluster_id'].values
z_scores = compute_cluster_zscore(prob_best, cluster_ids)

print(f"\n--- Best Strategy: {best_name} ---")
print(f"  Median prob all: {np.median(prob_best):.5f}")
print(f"  Mean prob known CN: {prob_best[known_mask].mean():.4f}")
print(f"  Median prob known CN: {np.median(prob_best[known_mask]):.4f}")
print(f"  Unlabeled prob > 0.1: {(prob_best[unl_mask] > 0.1).sum():,}")
print(f"  z > 2.0: {(z_scores > 2.0).sum():,} ({(z_scores > 2.0).sum()/len(z_scores)*100:.2f}%)")
print(f"  Unlabeled z > 2.0: {(z_scores[unl_mask] > 2.0).sum():,}  ← CANDIDATES")
print(f"  Known CN z > 2.0: {(z_scores[known_mask] > 2.0).sum()}/{n_known}")

# 6. Unlabeled candidate spectra
unl_indices = np.where(unl_mask)[0]
unl_z = z_scores[unl_mask]
unl_order = np.argsort(unl_z)[::-1]
top_unl = unl_indices[unl_order[:20]]
top_unl = top_unl[z_scores[top_unl] > 1.5]
n_show = min(16, len(top_unl))
top_unl = top_unl[:n_show]

# Cluster medians
cluster_medians = {}
unique_cids = stars_clean.iloc[top_unl]['masked_cluster_id'].unique()
for cid in unique_cids:
    cmask = stars_clean['masked_cluster_id'] == cid
    cluster_medians[cid] = np.nanmedian(X_clean[stars_clean.index[cmask]], axis=0)

known_cn_median = np.nanmedian(X_clean[known_mask], axis=0)

n_cols = 4; n_rows = (n_show + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2*n_cols, 3.2*n_rows))
axes_flat = axes.flatten() if n_rows > 1 else [axes]

for i in range(n_show):
    ax = axes_flat[i]
    idx = top_unl[i]
    star = stars_clean.iloc[idx]
    cid = star['masked_cluster_id']

    if cid in cluster_medians:
        ax.plot(common_wave, cluster_medians[cid], color='darkorange', linewidth=1.0, linestyle='--', alpha=0.7)
    ax.plot(common_wave, X_clean[idx], color='navy', linewidth=0.7)
    for l1, l2, c in [(3830, 3883, 'blue'), (4120, 4216, 'green'), (4285, 4315, 'red')]:
        ax.axvspan(l1, l2, alpha=0.10, color=c, zorder=0)
    ax.set_title(f'#{i+1} | Teff={star["teff"]:.0f}K | prob={prob_best[idx]:.3f} | z={z_scores[idx]:.2f}', fontsize=7.5)
    ax.set_xlim(3800, 4500); ax.tick_params(labelsize=6.5); ax.grid(alpha=0.15)

for j in range(n_show, len(axes_flat)): axes_flat[j].axis('off')
fig.suptitle(f'Label Spreading — Top {n_show} UNLABELED Candidates ({best_name})', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/unlabeled_candidate_spectra.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ unlabeled_candidate_spectra.png saved")

# 7. Known vs Candidate comparison
n_compare = 5
known_sample = np.random.RandomState(42).choice(np.where(known_mask)[0], min(n_compare, n_known), replace=False)
unl_sample = top_unl[:n_compare]

fig, axes = plt.subplots(2, n_compare, figsize=(3.2*n_compare, 7))
for col in range(n_compare):
    ax = axes[0, col]
    ax.plot(common_wave, known_cn_median, color='darkorange', linewidth=1.0, linestyle='--', alpha=0.7)
    ax.plot(common_wave, X_clean[known_sample[col]], color='#e74c3c', linewidth=0.7)
    for l1, l2, c in [(3830, 3883, 'blue'), (4120, 4216, 'green'), (4285, 4315, 'red')]:
        ax.axvspan(l1, l2, alpha=0.10, color=c, zorder=0)
    ax.set_title(f'Known #{col+1}\nTeff={stars_clean.iloc[known_sample[col]]["teff"]:.0f}K', fontsize=8, color='#c0392b')
    ax.set_xlim(3800, 4500); ax.tick_params(labelsize=6.5); ax.grid(alpha=0.15)

    ax = axes[1, col]
    ax.plot(common_wave, known_cn_median, color='darkorange', linewidth=1.0, linestyle='--', alpha=0.7)
    ax.plot(common_wave, X_clean[unl_sample[col]], color='navy', linewidth=0.7)
    for l1, l2, c in [(3830, 3883, 'blue'), (4120, 4216, 'green'), (4285, 4315, 'red')]:
        ax.axvspan(l1, l2, alpha=0.10, color=c, zorder=0)
    ax.set_title(f'Candidate #{col+1}\nTeff={stars_clean.iloc[unl_sample[col]]["teff"]:.0f}K z={z_scores[unl_sample[col]]:.2f}', fontsize=8, color='navy')
    ax.set_xlim(3800, 4500); ax.tick_params(labelsize=6.5); ax.grid(alpha=0.15)

fig.suptitle('Known CN Stars vs Unlabeled Label-Spreading Candidates', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/known_vs_candidate_spectra.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ known_vs_candidate_spectra.png saved")

# 8. Unlabeled param space
teff_all = stars_clean['teff'].values
logg_all = stars_clean['logg'].values

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
rng = np.random.RandomState(42)
bg_sample = rng.choice(np.where(unl_mask)[0], min(4000, n_unl), replace=False)

ax = axes[0]
ax.scatter(teff_all[bg_sample], logg_all[bg_sample], s=2, c='#e0e0e0', alpha=0.3, edgecolors='none', rasterized=True)
unl_high_z = unl_mask & (z_scores > 2.0)
sc = ax.scatter(teff_all[unl_high_z], logg_all[unl_high_z],
                s=20, c=z_scores[unl_high_z], cmap='YlOrRd',
                edgecolors='black', linewidth=0.3, vmin=2.0, vmax=max(5, z_scores.max()))
ax.scatter(teff_all[known_mask], logg_all[known_mask], s=80, c='blue', marker='*',
           edgecolors='white', linewidth=0.8, label=f'Known CN ({n_known})', zorder=4)
plt.colorbar(sc, ax=ax, label='z-score', shrink=0.8)
ax.set_xlabel('Teff (K)'); ax.set_ylabel('log g'); ax.invert_xaxis(); ax.invert_yaxis()
ax.set_title(f'Unlabeled High-Confidence Candidates (z > 2.0, n={unl_high_z.sum()})')
ax.legend(fontsize=7); ax.grid(alpha=0.2)

ax = axes[1]
ax.scatter(teff_all[bg_sample], logg_all[bg_sample], s=2, c='#e0e0e0', alpha=0.2, edgecolors='none', rasterized=True)
unl_very_high = unl_mask & (z_scores > 3.0)
sc2 = ax.scatter(teff_all[unl_very_high], logg_all[unl_very_high],
                 s=25, c=z_scores[unl_very_high], cmap='YlOrRd',
                 edgecolors='black', linewidth=0.5, vmin=3.0, vmax=max(6, z_scores.max()))
top5_idx = np.where(unl_very_high)[0][np.argsort(z_scores[unl_very_high])[::-1][:5]]
for i, idx in enumerate(top5_idx):
    ax.annotate(f'#{i+1}', (teff_all[idx], logg_all[idx]),
                fontsize=7, ha='right', va='bottom',
                textcoords='offset points', xytext=(-4, 4), color='darkred')
ax.scatter(teff_all[known_mask], logg_all[known_mask], s=80, c='blue', marker='*',
           edgecolors='white', linewidth=0.8, zorder=4)
plt.colorbar(sc2, ax=ax, label='z-score', shrink=0.8)
ax.set_xlabel('Teff (K)'); ax.set_ylabel('log g'); ax.invert_xaxis(); ax.invert_yaxis()
ax.set_title(f'Very High Confidence Candidates (z > 3.0, n={unl_very_high.sum()})')
ax.legend(fontsize=7); ax.grid(alpha=0.2)

fig.suptitle('Label Spreading — Unlabeled Candidate Distribution in Stellar Parameter Space', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/unlabeled_param_space.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ unlabeled_param_space.png saved")

# 9. K sensitivity
spectra_results = results_df[results_df['name'].str.contains('spectra')]
physics_results = results_df[results_df['name'].str.contains('physics_euclidean')]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
if len(spectra_results) > 0:
    ax.plot(spectra_results['K'], spectra_results['precision@100'], 'o-', color='#3498db', linewidth=2, markersize=8, label='Spectra Cosine')
if len(physics_results) > 0:
    ax.plot(physics_results['K'], physics_results['precision@100'], 's-', color='#2ecc71', linewidth=2, markersize=8, label='Physics Euclidean')
ax.set_xlabel('K (neighbors)'); ax.set_ylabel('Precision@100')
ax.set_title('Precision@100 vs K'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
if len(spectra_results) > 0:
    ax.plot(spectra_results['K'], spectra_results['prob_ratio'], 'o-', color='#3498db', linewidth=2, markersize=8, label='Spectra Cosine')
if len(physics_results) > 0:
    ax.plot(physics_results['K'], physics_results['prob_ratio'], 's-', color='#2ecc71', linewidth=2, markersize=8, label='Physics Euclidean')
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('K (neighbors)'); ax.set_ylabel('Prob Ratio')
ax.set_title('Signal-to-Background vs K'); ax.legend(); ax.grid(alpha=0.3)
fig.suptitle('Label Spreading — K Sensitivity', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/k_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ k_sensitivity.png saved")

# 10. Export candidates
stars_clean['ls_prob'] = prob_best
stars_clean['ls_zscore'] = z_scores

out_cols = ['uid', 'ra', 'dec', 'teff', 'logg', 'feh', 'label', 'snru',
            'mag_ps_g', 'ls_prob', 'ls_zscore', 'masked_cluster_id']
out_cols = [c for c in out_cols if c in stars_clean.columns]
stars_export = stars_clean.sort_values('ls_zscore', ascending=False)
export_df = stars_export[(stars_export['label'] == -1) & (stars_export['ls_zscore'] > 2.0)][out_cols].head(200).copy()
export_df.to_csv(f'{OUT}/LS_candidates.csv', index=False)
print(f"\n→ LS_candidates.csv: {len(export_df)} candidates")

full_out = stars_clean[['uid', 'ra', 'dec', 'teff', 'logg', 'feh', 'label',
                         'ls_prob', 'ls_zscore', 'masked_cluster_id']].copy()
full_out.to_csv(f'{OUT}/LS_all_probs.csv', index=False)
print(f"→ LS_all_probs.csv: {len(full_out):,} stars")

print(f"\n=== Label Spreading Done! ===")
print(f"Unlabeled high-confidence candidates (z>2.0): {unl_high_z.sum()} stars")
print(f"Exported to {OUT}/LS_candidates.csv")
