"""Run SpectralClustering analysis end-to-end."""
import sys, os, time, warnings
sys.path.insert(0, 'd:/资料/2026大创/Lamost')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

from LabelSpreading.graph_utils import build_knn_graph, compute_cluster_zscore, export_candidates
from LabelSpreading.spectral_utils import (
    build_normalized_laplacian, spectral_decomposition,
    cluster_on_eigenvectors, evaluate_cluster_enrichment, run_spectral_pipeline,
)
from PhaseSummary.shared.data_loader import ensure_cache

PROJ = 'd:/资料/2026大创/Lamost'
OUT = f'{PROJ}/LabelSpreading'

print("="*60)
print("Spectral Clustering — Full Pipeline")
print("="*60)

data = ensure_cache()
X_clean = data['X_clean']
stars_clean = data['stars_clean']
common_wave = data['common_wave']
known_mask = stars_clean['label'].values == 1
n_known = known_mask.sum()

print(f"Data: {X_clean.shape[0]:,} stars, {n_known} known CN")

# 1. Build KNN graph + run spectral decomposition
K = 50
print(f"\nBuilding KNN graph (K={K}, cosine)...")
t0 = time.time()
W = build_knn_graph(X_clean, n_neighbors=K, metric='cosine', standardize=False)
print(f"Graph: {W.shape}, edges: {W.nnz:,} ({time.time()-t0:.1f}s)")

print("\nRunning spectral pipeline...")
t0 = time.time()
result = run_spectral_pipeline(
    W, stars_clean,
    n_eigenvectors=20,
    n_clusters_list=[5, 10, 15, 20, 30],
)
print(f"Spectral pipeline done in {time.time()-t0:.1f}s")

# 2. Eigenvalue analysis
eigenvalues = result['eigenvalues']
gaps = np.diff(eigenvalues)
best_k = np.argmax(gaps[:15]) + 2

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.plot(range(1, len(eigenvalues)+1), eigenvalues, 'o-', color='#3498db', markersize=6)
ax.set_xlabel('Eigenvalue index'); ax.set_ylabel('Eigenvalue')
ax.set_title('Normalized Laplacian Eigenvalues'); ax.axhline(0, color='gray', linestyle='--', alpha=0.3); ax.grid(alpha=0.2)

ax = axes[1]
ax.plot(range(2, len(eigenvalues)+1), gaps, 's-', color='#e74c3c', markersize=6)
ax.set_xlabel('Eigenvalue index'); ax.set_ylabel('Eigengap')
ax.set_title('Eigengaps (larger gap = natural cluster boundary)')
ax.axvline(best_k, color='green', linestyle='--', alpha=0.6, label=f'Max gap at k={best_k}')
ax.legend(); ax.grid(alpha=0.2)
fig.suptitle('Spectral Clustering — Eigenvalue Analysis', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/eigenvalues.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"→ eigenvalues.png saved (max eigengap at k={best_k})")

# 3. Spectral embedding visualization
eigenvectors = result['eigenvectors']
best_labels = result['best_cluster_labels']
n_clusters = result['best_n_clusters']

fig = plt.figure(figsize=(14, 5.5))
ax = fig.add_subplot(1, 2, 1)
for cid in range(n_clusters):
    cmask = best_labels == cid
    ax.scatter(eigenvectors[cmask, 0], eigenvectors[cmask, 1],
              s=1, alpha=0.5, rasterized=True)
ax.scatter(eigenvectors[known_mask, 0], eigenvectors[known_mask, 1],
          s=60, c='red', marker='*', edgecolors='black', linewidth=0.5,
          label=f'Known CN ({n_known})', zorder=5)
ax.set_xlabel('Eigenvector 1'); ax.set_ylabel('Eigenvector 2')
ax.set_title(f'Spectral Embedding ({n_clusters} clusters)'); ax.legend(fontsize=5); ax.grid(alpha=0.2)

ax = fig.add_subplot(1, 2, 2)
sc = ax.scatter(eigenvectors[:, 0], eigenvectors[:, 1],
               s=1, c=eigenvectors[:, 2], cmap='RdYlBu_r', alpha=0.5, rasterized=True)
ax.scatter(eigenvectors[known_mask, 0], eigenvectors[known_mask, 1],
          s=60, c='green', marker='*', edgecolors='black', linewidth=0.5, zorder=5)
plt.colorbar(sc, ax=ax, label='Eigenvector 3')
ax.set_xlabel('Eigenvector 1'); ax.set_ylabel('Eigenvector 2')
ax.set_title('Spectral Embedding (colored by EV3)'); ax.grid(alpha=0.2)

fig.suptitle(f'Spectral Clustering — {n_clusters} Clusters in Eigenvector Space', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/spectral_embedding.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ spectral_embedding.png saved")

# 4. Enrichment analysis
enrich_df = result['best_enrichment']
global_rate = known_mask.sum() / len(stars_clean)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
colors = ['#e74c3c' if e > 3 else '#2ecc71' if e > 1 else '#b0b0b0' for e in enrich_df['enrichment']]
ax.barh(enrich_df['cluster'].astype(str), enrich_df['enrichment'], color=colors, edgecolor='white')
ax.axvline(1.0, color='gray', linestyle='--', label=f'Global rate ({global_rate:.4f})')
ax.axvline(3.0, color='red', linestyle='--', alpha=0.5, label='3x enrichment')
ax.set_xlabel('Enrichment Ratio'); ax.set_ylabel('Cluster')
ax.set_title(f'CN-Star Enrichment by Spectral Cluster'); ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.2)

ax = axes[1]
top5 = enrich_df.head(5)
ax.bar(top5['cluster'].astype(str), top5['n_total'], color='steelblue', alpha=0.4, label='Total stars')
ax.bar(top5['cluster'].astype(str), top5['n_cn'], color='#e74c3c', alpha=0.8, label='Known CN')
ax.set_xlabel('Cluster'); ax.set_ylabel('Count')
ax.set_title('Top-5 Enriched Clusters'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.2)

fig.suptitle('Spectral Clustering — CN Enrichment Analysis', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/cluster_enrichment.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ cluster_enrichment.png saved")
print("\nTop enriched clusters:")
print(enrich_df.head(10).to_string(index=False))

# 5. Enriched cluster spectra
top_cluster = enrich_df.iloc[0]['cluster']
cmask = best_labels == top_cluster
unl_in_cluster = cmask & (stars_clean['label'].values == -1)
known_in_cluster = cmask & known_mask

print(f"\nTop enriched cluster #{top_cluster}: Total={cmask.sum()}, Known CN={known_in_cluster.sum()}, Unlabeled={unl_in_cluster.sum()}")

n_show = min(12, unl_in_cluster.sum())
unl_indices_in_cluster = np.where(unl_in_cluster)[0]
rng = np.random.RandomState(42)
show_indices = rng.choice(unl_indices_in_cluster, n_show, replace=False)

cluster_median = np.nanmedian(X_clean[cmask], axis=0)
known_cn_median = np.nanmedian(X_clean[known_mask], axis=0)

n_cols = 4; n_rows = (n_show + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
axes_flat = axes.flatten() if n_rows > 1 else [axes]

for i in range(n_show):
    ax = axes_flat[i]
    idx = show_indices[i]
    ax.plot(common_wave, cluster_median, color='darkorange', linewidth=1.0, linestyle='--', alpha=0.7, label='Cluster median')
    ax.plot(common_wave, known_cn_median, color='#e74c3c', linewidth=0.8, linestyle=':', alpha=0.6, label='CN median')
    ax.plot(common_wave, X_clean[idx], color='navy', linewidth=0.7)
    for l1, l2, c in [(3830, 3883, 'blue'), (4120, 4216, 'green'), (4285, 4315, 'red')]:
        ax.axvspan(l1, l2, alpha=0.10, color=c, zorder=0)
    ax.set_title(f'#{i+1} Teff={stars_clean.iloc[idx]["teff"]:.0f}K', fontsize=7)
    ax.set_xlim(3800, 4500); ax.tick_params(labelsize=6); ax.grid(alpha=0.15)
    if i == 0: ax.legend(fontsize=5, loc='upper right')

for j in range(n_show, len(axes_flat)): axes_flat[j].axis('off')
fig.suptitle(f'Spectral Clustering — Unlabeled Stars in Top Enriched Cluster #{top_cluster}', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/enriched_cluster_spectra.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ enriched_cluster_spectra.png saved")

print(f"\n=== Spectral Clustering Done! ===")
print(f"Best n_clusters: {n_clusters}")
print(f"Top enriched cluster: #{top_cluster} ({enrich_df.iloc[0]['enrichment']:.1f}x enrichment)")
print(f"  {known_in_cluster.sum()}/{cmask.sum()} known CN in cluster")
print(f"  {unl_in_cluster.sum()} unlabeled stars to investigate")
