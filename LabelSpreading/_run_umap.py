"""Run UMAP analysis end-to-end."""
import sys, os, time, warnings
sys.path.insert(0, 'd:/资料/2026大创/Lamost')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

from LabelSpreading.umap_utils import run_umap_pipeline
from LabelSpreading.graph_utils import compute_cluster_zscore
from PhaseSummary.shared.data_loader import ensure_cache

PROJ = 'd:/资料/2026大创/Lamost'
OUT = f'{PROJ}/LabelSpreading'

print("="*60)
print("UMAP — Manifold Analysis & Density-Guided Candidate Discovery")
print("="*60)

# 1. Load data
data = ensure_cache()
X_clean = data['X_clean']
stars_clean = data['stars_clean']
common_wave = data['common_wave']
known_mask = stars_clean['label'].values == 1
unlabeled_mask = stars_clean['label'].values == -1
n_known = known_mask.sum()
n_unl = unlabeled_mask.sum()
print(f"Data: {X_clean.shape[0]:,} stars, {n_known} known CN, {n_unl:,} unlabeled")

# 2. Run UMAP pipeline
print("\nRunning UMAP pipeline (n_neighbors=30, min_dist=0.1, cosine)...")
result = run_umap_pipeline(
    X_clean, stars_clean,
    n_neighbors=30, min_dist=0.1, top_n=200, verbose=True,
)
embedding = result['embedding']
density = result['density']
candidates = result['candidate_indices']
print(f"Candidates selected: {len(candidates)}")

# 3. UMAP embedding visualization (3 views)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
rng = np.random.RandomState(42)
bg = rng.choice(np.where(unlabeled_mask)[0], 5000, replace=False)

ax = axes[0]
ax.scatter(embedding[bg, 0], embedding[bg, 1], s=1, c='#e0e0e0', alpha=0.3, edgecolors='none', rasterized=True)
ax.scatter(embedding[known_mask, 0], embedding[known_mask, 1],
           s=60, c='#e74c3c', marker='*', edgecolors='black', linewidth=0.5, label=f'Known CN ({n_known})', zorder=3)
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2'); ax.set_title('UMAP: Known CN Stars'); ax.legend(fontsize=8); ax.grid(alpha=0.2)

ax = axes[1]
sc = ax.scatter(embedding[bg, 0], embedding[bg, 1], s=1, c=density[bg], cmap='YlOrRd', alpha=0.4, vmin=0, edgecolors='none', rasterized=True)
ax.scatter(embedding[known_mask, 0], embedding[known_mask, 1],
           s=50, c='blue', marker='*', edgecolors='white', linewidth=0.5, zorder=3)
plt.colorbar(sc, ax=ax, label='CN Density (KDE)', shrink=0.8)
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2'); ax.set_title('UMAP: CN Density (KDE)'); ax.grid(alpha=0.2)

ax = axes[2]
ax.scatter(embedding[bg, 0], embedding[bg, 1], s=1, c='#e0e0e0', alpha=0.2, edgecolors='none', rasterized=True)
ax.scatter(embedding[known_mask, 0], embedding[known_mask, 1],
           s=50, c='blue', marker='*', edgecolors='white', linewidth=0.5, label=f'Known CN', zorder=3)
ax.scatter(embedding[candidates, 0], embedding[candidates, 1],
           s=30, c='#e74c3c', edgecolors='black', linewidth=0.3, label=f'UMAP Candidates ({len(candidates)})', zorder=2)
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2'); ax.set_title('UMAP: Selected Candidates'); ax.legend(fontsize=7); ax.grid(alpha=0.2)

fig.suptitle('UMAP Manifold Analysis — CN-Star Candidate Discovery', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/umap_embedding.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ umap_embedding.png saved")

# 4. Candidate spectra
cand_order = np.argsort(density[candidates])[::-1]
n_show = min(12, len(candidates))
show_indices = candidates[cand_order[:n_show]]
known_cn_median = np.nanmedian(X_clean[known_mask], axis=0)

n_cols = 4; n_rows = (n_show + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2*n_cols, 3.2*n_rows))
axes_flat = axes.flatten() if n_rows > 1 else [axes]

for i in range(n_show):
    ax = axes_flat[i]
    idx = show_indices[i]
    star = stars_clean.iloc[idx]
    ax.plot(common_wave, known_cn_median, color='darkorange', linewidth=1.0, linestyle='--', alpha=0.7, label='CN median')
    ax.plot(common_wave, X_clean[idx], color='navy', linewidth=0.7)
    for l1, l2, c in [(3830, 3883, 'blue'), (4120, 4216, 'green'), (4285, 4315, 'red')]:
        ax.axvspan(l1, l2, alpha=0.10, color=c, zorder=0)
    ax.set_title(f'#{i+1} Teff={star["teff"]:.0f}K\nd={density[idx]:.3f}', fontsize=7.5)
    ax.set_xlim(3800, 4500); ax.tick_params(labelsize=6.5); ax.grid(alpha=0.15)
    if i == 0: ax.legend(fontsize=5.5, loc='upper right')

for j in range(n_show, len(axes_flat)): axes_flat[j].axis('off')
fig.suptitle('UMAP — Top Candidates by CN Density', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/umap_candidate_spectra.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ umap_candidate_spectra.png saved")

# 5. Teff-logg distribution
teff_all = stars_clean['teff'].values
logg_all = stars_clean['logg'].values

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
bg_sample = rng.choice(np.where(unlabeled_mask)[0], 4000, replace=False)

ax = axes[0]
ax.scatter(teff_all[bg_sample], logg_all[bg_sample], s=2, c='#e0e0e0', alpha=0.2, edgecolors='none', rasterized=True)
sc = ax.scatter(teff_all[candidates], logg_all[candidates],
                s=25, c=density[candidates], cmap='YlOrRd', edgecolors='black', linewidth=0.3, zorder=2)
ax.scatter(teff_all[known_mask], logg_all[known_mask], s=80, c='blue', marker='*',
           edgecolors='white', linewidth=0.8, zorder=4, label=f'Known CN ({n_known})')
plt.colorbar(sc, ax=ax, label='CN Density', shrink=0.8)
ax.set_xlabel('Teff (K)'); ax.set_ylabel('log g'); ax.invert_xaxis(); ax.invert_yaxis()
ax.set_title(f'UMAP Candidates in Teff-logg ({len(candidates)} candidates)'); ax.legend(fontsize=7); ax.grid(alpha=0.2)

ax = axes[1]
bins = np.linspace(0, max(density.max(), density[known_mask].max()), 50)
ax.hist(density[unlabeled_mask], bins=bins, alpha=0.5, color='#b0b0b0', label=f'All unlabeled ({n_unl:,})')
ax.hist(density[known_mask], bins=bins, alpha=0.7, color='#e74c3c', label=f'Known CN ({n_known})')
ax.hist(density[candidates], bins=bins, alpha=0.7, color='#3498db', label=f'UMAP candidates ({len(candidates)})')
ax.axvline(np.median(density[known_mask]), color='#e74c3c', linestyle='--', label=f'Median known CN density', alpha=0.8)
ax.set_xlabel('CN Density (KDE)'); ax.set_ylabel('Count')
ax.set_title('Density Distribution'); ax.legend(fontsize=7); ax.grid(alpha=0.2)

fig.suptitle('UMAP — Candidate Distribution in Stellar Parameter Space', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/umap_param_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ umap_param_distribution.png saved")

# 6. UMAP vs LS overlap
ls_path = f'{OUT}/LS_all_probs.csv'
if os.path.exists(ls_path):
    ls_df = pd.read_csv(ls_path)
    ls_unl = ls_df[ls_df['label'] == -1].copy()
    ls_top200 = ls_unl.sort_values('ls_zscore', ascending=False).head(200)
    ls_uids = set(ls_top200['uid'].values)

    umap_uids = set(stars_clean.iloc[candidates]['uid'].values)
    overlap = ls_uids & umap_uids
    ls_only = ls_uids - umap_uids
    umap_only = umap_uids - ls_uids

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2); w = 0.4
    ax.bar(x[0], len(ls_only), w, color='#3498db', edgecolor='white', label='LS only')
    ax.bar(x[0], len(overlap), w, bottom=len(ls_only), color='#9b59b6', edgecolor='white', label='Both')
    ax.bar(x[1], len(umap_only), w, color='#e74c3c', edgecolor='white', label='UMAP only')
    ax.bar(x[1], len(overlap), w, bottom=len(umap_only), color='#9b59b6', edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(['LS top-200', 'UMAP top-200'])
    ax.set_ylabel('Number of Candidates')
    ax.set_title(f'UMAP vs Label Spreading Candidate Overlap\nOverlap: {len(overlap)} stars ({100*len(overlap)/200:.1f}%)')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/umap_vs_ls_overlap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("→ umap_vs_ls_overlap.png saved")
    print(f"\nOverlap: LS∩UMAP={len(overlap)} ({100*len(overlap)/200:.1f}%), LS only={len(ls_only)}, UMAP only={len(umap_only)}")
else:
    print("LS_all_probs.csv not found — skipping overlap analysis")

# 7. Export
out_cols = ['uid', 'ra', 'dec', 'teff', 'logg', 'feh', 'label', 'snru', 'mag_ps_g', 'masked_cluster_id']
stars_export = stars_clean.iloc[candidates].copy()
density_series = pd.Series(density, index=stars_clean.index)
stars_export['umap_density'] = density_series.iloc[candidates].values
stars_export = stars_export.sort_values('umap_density', ascending=False)
out_cols = [c for c in out_cols if c in stars_export.columns] + ['umap_density']
export_df = stars_export[out_cols].head(200).copy()
export_df.to_csv(f'{OUT}/UMAP_candidates.csv', index=False)
print(f"→ UMAP_candidates.csv: {len(export_df)} candidates")

print(f"\n=== UMAP Done! ===")
print(f"UMAP candidates: {len(candidates)}")
print(f"  Teff range: {stars_clean.iloc[candidates]['teff'].min():.0f} - {stars_clean.iloc[candidates]['teff'].max():.0f} K")
print(f"  logg range: {stars_clean.iloc[candidates]['logg'].min():.2f} - {stars_clean.iloc[candidates]['logg'].max():.2f}")
