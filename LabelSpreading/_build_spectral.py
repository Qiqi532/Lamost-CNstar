"""Build spectral_clustering.ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "myenv", "language": "python", "name": "myenv"},
    "language_info": {"name": "python", "version": "3.10.0"},
}

cells = []

def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# Spectral Clustering — 图拉普拉斯特征分解发现 CN 星子群

**核心思路：** 谱聚类的特征向量对应图的低频振动模式，能揭示数据的内在聚类结构。CN增强星如果在光谱流形上形成独立子群，会在某个特征向量维度上显现。

**与 Label Spreading 的关系：**
- Label Spreading = 在图结构上做**标签平滑传播**
- Spectral Clustering = 在图拉普拉斯矩阵上做**特征分解聚类**
- 两者共享同一个 KNN 图，从不同数学角度探索流形结构
""")

md("""## 1. 环境配置与数据加载""")

code(r"""import sys, os, time, warnings
from pathlib import Path
_PROJECT_ROOT = Path(os.getcwd())
for _ in range(5):
    if (_PROJECT_ROOT / "stars.csv").exists(): break
    _PROJECT_ROOT = _PROJECT_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.rcParams.update({'font.size': 10})
import seaborn as sns; sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

from LabelSpreading.graph_utils import build_knn_graph, compute_cluster_zscore, export_candidates
from LabelSpreading.spectral_utils import (
    build_normalized_laplacian, spectral_decomposition,
    cluster_on_eigenvectors, evaluate_cluster_enrichment, run_spectral_pipeline,
)
from PhaseSummary.shared.data_loader import ensure_cache

data = ensure_cache()
X_clean = data['X_clean']
stars_clean = data['stars_clean']
feature_df = data['feature_df']
common_wave = data['common_wave']

known_mask = stars_clean['label'].values == 1
print(f"数据加载: {X_clean.shape[0]:,} stars, {known_mask.sum()} known CN")
""")

md("""## 2. 构建 KNN 图 + 谱分解""")

code(r"""K = 50
print(f"Building KNN graph (K={K}, cosine on 700-D spectra)...")
W = build_knn_graph(X_clean, n_neighbors=K, metric='cosine', standardize=False)
print(f"Graph: {W.shape}, nonzeros={W.nnz:,}")

print(f"\nRunning spectral decomposition (20 eigenvectors)...")
result = run_spectral_pipeline(
    W, stars_clean,
    n_eigenvectors=20,
    n_clusters_list=[5, 10, 15, 20, 30],
)

print("\nDone.")
""")

md("""## 3. 特征值分布""")

code(r"""eigenvalues = result['eigenvalues']

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.plot(range(1, len(eigenvalues)+1), eigenvalues, 'o-', color='#3498db', markersize=6)
ax.set_xlabel('Eigenvalue index'); ax.set_ylabel('Eigenvalue')
ax.set_title('Normalized Laplacian Eigenvalues (smallest)')
ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
ax.grid(alpha=0.2)

# Eigengap analysis
ax = axes[1]
gaps = np.diff(eigenvalues)
ax.plot(range(2, len(eigenvalues)+1), gaps, 's-', color='#e74c3c', markersize=6)
ax.set_xlabel('Eigenvalue index'); ax.set_ylabel('Eigengap')
ax.set_title('Eigengaps (larger gap = natural cluster boundary)')
best_k = np.argmax(gaps[:15]) + 2
ax.axvline(best_k, color='green', linestyle='--', alpha=0.6, label=f'Max gap at k={best_k}')
ax.legend(); ax.grid(alpha=0.2)

fig.suptitle('Spectral Clustering — Eigenvalue Analysis', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(str(_PROJECT_ROOT / 'LabelSpreading/eigenvalues.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"Max eigengap suggests natural cluster count: k={best_k}")
""")

md("""## 4. 特征向量空间可视化（前 3 维）""")

code(r"""eigenvectors = result['eigenvectors']
best_labels = result['best_cluster_labels']
n_clusters = result['best_n_clusters']

fig = plt.figure(figsize=(14, 5.5))

# (a) Colored by spectral cluster
ax = fig.add_subplot(1, 2, 1)
for cid in range(n_clusters):
    cmask = best_labels == cid
    ax.scatter(eigenvectors[cmask, 0], eigenvectors[cmask, 1],
              s=1, alpha=0.5, label=f'C{cid}' if cmask.sum() > 500 else '', rasterized=True)
ax.scatter(eigenvectors[known_mask, 0], eigenvectors[known_mask, 1],
          s=60, c='red', marker='*', edgecolors='black', linewidth=0.5,
          label=f'Known CN ({known_mask.sum()})', zorder=5)
ax.set_xlabel('Eigenvector 1'); ax.set_ylabel('Eigenvector 2')
ax.set_title(f'Spectral Embedding (colored by {n_clusters}-cluster)')
ax.legend(fontsize=5, loc='upper right', ncol=2); ax.grid(alpha=0.2)

# (b) Colored by eigenvector 3
ax = fig.add_subplot(1, 2, 2)
sc = ax.scatter(eigenvectors[:, 0], eigenvectors[:, 1],
               s=1, c=eigenvectors[:, 2], cmap='RdYlBu_r', alpha=0.5, rasterized=True)
ax.scatter(eigenvectors[known_mask, 0], eigenvectors[known_mask, 1],
          s=60, c='green', marker='*', edgecolors='black', linewidth=0.5, zorder=5)
cbar = plt.colorbar(sc, ax=ax, label='Eigenvector 3')
ax.set_xlabel('Eigenvector 1'); ax.set_ylabel('Eigenvector 2')
ax.set_title('Spectral Embedding (colored by EV3)')
ax.grid(alpha=0.2)

fig.suptitle(f'Spectral Clustering — {n_clusters} Clusters in Eigenvector Space', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(str(_PROJECT_ROOT / 'LabelSpreading/spectral_embedding.png'), dpi=150, bbox_inches='tight')
plt.show()
""")

md("""## 5. 各簇 CN 星富集度""")

code(r"""enrich_df = result['best_enrichment']
global_rate = known_mask.sum() / len(stars_clean)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) Enrichment bar chart
ax = axes[0]
colors = ['#e74c3c' if e > 3 else '#2ecc71' if e > 1 else '#b0b0b0' for e in enrich_df['enrichment']]
ax.barh(enrich_df['cluster'].astype(str), enrich_df['enrichment'], color=colors, edgecolor='white')
ax.axvline(1.0, color='gray', linestyle='--', label=f'Global rate ({global_rate:.4f})')
ax.axvline(3.0, color='red', linestyle='--', alpha=0.5, label='3x enrichment')
ax.set_xlabel('Enrichment Ratio'); ax.set_ylabel('Cluster')
ax.set_title(f'CN-Star Enrichment by Spectral Cluster ({n_clusters} clusters)')
ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.2)

# (b) Top enriched clusters — CN count vs total
ax = axes[1]
top5 = enrich_df.head(5)
ax.bar(top5['cluster'].astype(str), top5['n_total'], color='steelblue', alpha=0.4, label='Total stars')
ax.bar(top5['cluster'].astype(str), top5['n_cn'], color='#e74c3c', alpha=0.8, label='Known CN')
ax.set_xlabel('Cluster'); ax.set_ylabel('Count')
ax.set_title('Top-5 Enriched Clusters: CN vs Total')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.2)

fig.suptitle('Spectral Clustering — CN Enrichment Analysis', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(str(_PROJECT_ROOT / 'LabelSpreading/cluster_enrichment.png'), dpi=150, bbox_inches='tight')
plt.show()

print("Top enriched clusters:")
print(enrich_df.head(10).to_string(index=False))
""")

md("""## 6. 高富集簇候选体光谱""")

code(r"""# Find the most enriched cluster
top_cluster = enrich_df.iloc[0]['cluster']
cmask = best_labels == top_cluster
unl_in_cluster = cmask & (stars_clean['label'].values == -1)
known_in_cluster = cmask & known_mask

print(f"Top enriched cluster #{top_cluster}:")
print(f"  Total stars: {cmask.sum()}")
print(f"  Known CN: {known_in_cluster.sum()}")
print(f"  Unlabeled: {unl_in_cluster.sum()}")

# Plot spectra of unlabeled stars in the enriched cluster vs cluster median
n_show = min(12, unl_in_cluster.sum())
unl_indices_in_cluster = np.where(unl_in_cluster)[0]

# Randomly select from unlabeled
rng = np.random.RandomState(42)
show_indices = rng.choice(unl_indices_in_cluster, n_show, replace=False)

# Cluster median and known CN median
cluster_median = np.nanmedian(X_clean[cmask], axis=0)
known_cn_median = np.nanmedian(X_clean[known_mask], axis=0)

n_cols = 4; n_rows = (n_show + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
axes_flat = axes.flatten() if n_rows > 1 else [axes]

for i in range(n_show):
    ax = axes_flat[i]
    idx = show_indices[i]
    ax.plot(common_wave, cluster_median, color='darkorange', linewidth=1.0,
            linestyle='--', alpha=0.7, label='Cluster median')
    ax.plot(common_wave, known_cn_median, color='#e74c3c', linewidth=0.8,
            linestyle=':', alpha=0.6, label='CN median')
    ax.plot(common_wave, X_clean[idx], color='navy', linewidth=0.7)
    for l1, l2, c in [(3830, 3883, 'blue'), (4120, 4216, 'green'), (4285, 4315, 'red')]:
        ax.axvspan(l1, l2, alpha=0.10, color=c, zorder=0)
    ax.set_title(f'#{i+1} Teff={stars_clean.iloc[idx]["teff"]:.0f}K', fontsize=7)
    ax.set_xlim(3800, 4500); ax.tick_params(labelsize=6); ax.grid(alpha=0.15)
    if i == 0: ax.legend(fontsize=5, loc='upper right')

for j in range(n_show, len(axes_flat)): axes_flat[j].axis('off')

fig.suptitle(f'Spectral Clustering — Unlabeled Stars in Top Enriched Cluster #{top_cluster}',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(str(_PROJECT_ROOT / 'LabelSpreading/enriched_cluster_spectra.png'), dpi=150, bbox_inches='tight')
plt.show()
""")

md("""## 7. 总结

### 关键发现

1. **谱聚类找到 CN 富集簇** — 拉普拉斯特征向量空间中存在 CN 星聚集的子区域
2. **特征向量可视化** — 前 3 个特征向量可直观展示 CN 星在图流形上的分布
3. **与 Label Spreading 互补** — 谱聚类是"硬"聚类，Label Spreading 是"软"传播，两者结合提供更全面的流形图景

### 局限

- 特征分解在大图上较慢（33K×33K 约需 1-2 分钟）
- 簇数量选择影响结果（通过 eigengap 启发式选择）
- CN 星仅 73 颗，在 coarse 聚类中可能被淹没在少量簇中
""")

nb.cells = cells
with open("d:/资料/2026大创/Lamost/LabelSpreading/spectral_clustering.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Notebook written: {len(cells)} cells")
