"""Run GNN analysis end-to-end."""
import sys, os, time, warnings
sys.path.insert(0, 'd:/资料/2026大创/Lamost')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

import torch
from LabelSpreading.graph_utils import build_knn_graph, compute_cluster_zscore
from LabelSpreading.gnn_models import build_model
from LabelSpreading.gnn_train import prepare_gnn_data, train_gnn, predict_all, evaluate_gnn_topk
from PhaseSummary.shared.data_loader import ensure_cache
from sklearn.decomposition import PCA

PROJ = 'd:/资料/2026大创/Lamost'
OUT = f'{PROJ}/LabelSpreading'
device = torch.device('cpu')
print("="*60)
print(f"GNN — Node Classification on KNN Graph (CPU, {device})")
print("="*60)

# 1. Load data
data = ensure_cache()
X_clean = data['X_clean']
stars_clean = data['stars_clean']
common_wave = data['common_wave']
known_mask = stars_clean['label'].values == 1
n_known = known_mask.sum()
print(f"Data: {X_clean.shape[0]:,} stars, {n_known} known CN")

# 2. PCA features + KNN graph
pca = PCA(n_components=32, random_state=42)
X_pca = pca.fit_transform(X_clean)
print(f"PCA features: {X_pca.shape} (cumulative variance: {pca.explained_variance_ratio_.sum():.3f})")

K = 50
print(f"Building KNN graph (K={K}, cosine)...")
W = build_knn_graph(X_clean, n_neighbors=K, metric='cosine', standardize=False)
print(f"Graph: {W.shape}, edges: {W.nnz:,}")

# 3. Prepare labels and splits
y_labels = stars_clean['label'].map({1: 1, -1: -1}).values.astype(int)
pos_indices = np.where(y_labels == 1)[0]
unl_indices = np.where(y_labels == -1)[0]
n_pos = len(pos_indices)

rng = np.random.RandomState(42)
perm = rng.permutation(n_pos)
n_train = int(n_pos * 0.6)
n_val = int(n_pos * 0.2)
train_pos = pos_indices[perm[:n_train]]
val_pos = pos_indices[perm[n_train:n_train+n_val]]
test_pos = pos_indices[perm[n_train+n_val:]]

n_unl_train = 500
unl_train = rng.choice(unl_indices, n_unl_train, replace=False)

train_mask = np.zeros(len(stars_clean), dtype=bool)
train_mask[train_pos] = True
train_mask[unl_train] = True
y_labels_train = y_labels.copy()
y_labels_train[unl_train] = 0

val_mask = np.zeros(len(stars_clean), dtype=bool)
val_mask[val_pos] = True
unl_val = rng.choice(list(set(unl_indices) - set(unl_train)), 200, replace=False)
val_mask[unl_val] = True

print(f"Train nodes: {train_mask.sum()}, Val: {val_mask.sum()}")
print(f"  Pos: train={len(train_pos)}, val={len(val_pos)}, test={len(test_pos)}")

gnn_data = prepare_gnn_data(X_pca, W, y_labels_train, device)

# 4. Train GCN
print("\n--- Training GCN ---")
model_gcn = build_model('gcn', in_features=32, device=device)
print(f"GCN params: {sum(p.numel() for p in model_gcn.parameters()):,}")

result_gcn = train_gnn(
    model_gcn, gnn_data, train_mask, val_mask,
    epochs=300, lr=0.01, pos_weight=100.0, patience=40, verbose=True,
)
print(f"GCN Best val PR-AUC: {result_gcn['best_val_pr_auc']:.4f}")

prob_gcn = predict_all(model_gcn, gnn_data)
metrics_gcn = evaluate_gnn_topk(prob_gcn, stars_clean['label'].values)
print(f"GCN Precision@50:  {metrics_gcn['precision@50']:.4f}")
print(f"GCN Precision@100: {metrics_gcn['precision@100']:.4f}")

# 5. Train GAT
print("\n--- Training GAT ---")
gnn_data_gat = prepare_gnn_data(X_pca, W, y_labels_train, device)
model_gat = build_model('gat', in_features=32, device=device)
print(f"GAT params: {sum(p.numel() for p in model_gat.parameters()):,}")

result_gat = train_gnn(
    model_gat, gnn_data_gat, train_mask, val_mask,
    epochs=300, lr=0.005, pos_weight=100.0, patience=40, verbose=True,
)
print(f"GAT Best val PR-AUC: {result_gat['best_val_pr_auc']:.4f}")

prob_gat = predict_all(model_gat, gnn_data_gat)
metrics_gat = evaluate_gnn_topk(prob_gat, stars_clean['label'].values)
print(f"GAT Precision@50:  {metrics_gat['precision@50']:.4f}")
print(f"GAT Precision@100: {metrics_gat['precision@100']:.4f}")

# 6. Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.plot(result_gcn['history']['train_loss'], color='#3498db', alpha=0.7, linewidth=1, label='GCN')
ax.plot(result_gat['history']['train_loss'], color='#e74c3c', alpha=0.7, linewidth=1, label='GAT')
ax.set_xlabel('Epoch'); ax.set_ylabel('Train Loss'); ax.set_title('Training Loss')
ax.legend(); ax.grid(alpha=0.2)

ax = axes[1]
ax.plot(result_gcn['history']['val_pr_auc'], color='#3498db', linewidth=1.5, label='GCN')
ax.plot(result_gat['history']['val_pr_auc'], color='#e74c3c', linewidth=1.5, label='GAT')
ax.set_xlabel('Epoch'); ax.set_ylabel('Val PR-AUC'); ax.set_title('Validation PR-AUC')
ax.legend(); ax.grid(alpha=0.2)

fig.suptitle('GNN Training Curves: GCN vs GAT', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ training_curves.png saved")

# 7. Method comparison (GCN vs GAT vs LabelSpreading)
# Try to load LS results
try:
    ls_df = pd.read_csv(f'{OUT}/LS_all_probs.csv')
    ls_prob = ls_df['ls_prob'].values
    metrics_ls = evaluate_gnn_topk(ls_prob, stars_clean['label'].values)
except:
    from LabelSpreading.graph_utils import run_label_spreading
    y_labeled = stars_clean['label'].map({1: 1, -1: -1}).values.astype(int)
    ls_prob, _ = run_label_spreading(W, y_labeled, alpha=0.5)
    metrics_ls = evaluate_gnn_topk(ls_prob, stars_clean['label'].values)

comparison = pd.DataFrame({
    'Method': ['GCN', 'GAT', 'LabelSpreading'],
    'Precision@50': [metrics_gcn['precision@50'], metrics_gat['precision@50'], metrics_ls['precision@50']],
    'Precision@100': [metrics_gcn['precision@100'], metrics_gat['precision@100'], metrics_ls['precision@100']],
    'Median Prob (known CN)': [metrics_gcn['median_prob_known'], metrics_gat['median_prob_known'], metrics_ls['median_prob_known']],
})

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(3); w = 0.25
ax.bar(x-w, comparison['Precision@50'], w, label='P@50', color='#3498db', edgecolor='white')
ax.bar(x, comparison['Precision@100'], w, label='P@100', color='#2ecc71', edgecolor='white')
ax.bar(x+w, comparison['Median Prob (known CN)'], w, label='Median Prob CN', color='#f39c12', edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(comparison['Method'])
ax.set_title('GCN vs GAT vs Label Spreading'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig(f'{OUT}/method_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ method_comparison.png saved")
print(comparison.to_string(index=False))

# 8. GNN candidate visualization
prob_gnn = prob_gcn if metrics_gcn['precision@100'] >= metrics_gat.get('precision@100', 0) else prob_gat
best_model = 'GCN' if metrics_gcn['precision@100'] >= metrics_gat.get('precision@100', 0) else 'GAT'
print(f"\nBest GNN model: {best_model}")

cluster_ids = stars_clean['masked_cluster_id'].values
z_gnn = compute_cluster_zscore(prob_gnn, cluster_ids)

unl_mask_arr = stars_clean['label'].values == -1
unl_z = z_gnn[unl_mask_arr]
top_unl_gnn = np.where(unl_mask_arr)[0][np.argsort(unl_z)[::-1][:12]]

teff_all = stars_clean['teff'].values
logg_all = stars_clean['logg'].values

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
rng = np.random.RandomState(42)
bg = rng.choice(np.where(unl_mask_arr)[0], 3000, replace=False)
ax.scatter(teff_all[bg], logg_all[bg], s=2, c='#e0e0e0', alpha=0.2, edgecolors='none')
sc = ax.scatter(teff_all[top_unl_gnn], logg_all[top_unl_gnn],
               s=40, c=prob_gnn[top_unl_gnn], cmap='YlOrRd', edgecolors='black', linewidth=0.5)
ax.scatter(teff_all[known_mask], logg_all[known_mask], s=80, c='blue', marker='*',
          edgecolors='white', linewidth=0.8)
plt.colorbar(sc, ax=ax, label=f'{best_model} prob'); ax.invert_xaxis(); ax.invert_yaxis()
ax.set_xlabel('Teff (K)'); ax.set_ylabel('log g')
ax.set_title(f'{best_model} Top Unlabeled Candidates'); ax.grid(alpha=0.2)

ax = axes[1]
known_cn_med = np.nanmedian(X_clean[known_mask], axis=0)
for i, idx in enumerate(top_unl_gnn[:6]):
    flux = X_clean[idx]
    offset = i * 0.3
    ax.plot(common_wave, flux + offset, color='navy', linewidth=0.7)
    ax.text(4510, offset + 0.05, f'#{i+1} Teff={teff_all[idx]:.0f}K',
            fontsize=6, va='bottom', ha='left', color='navy')
for l1, l2, c in [(3830, 3883, 'blue'), (4120, 4216, 'green'), (4285, 4315, 'red')]:
    ax.axvspan(l1, l2, alpha=0.08, color=c, zorder=0)
ax.set_xlim(3800, 4550); ax.set_xlabel('Wavelength (A)'); ax.set_ylabel('Flux + offset')
ax.set_title(f'{best_model} Top 6 Unlabeled Candidates'); ax.grid(alpha=0.15)

fig.suptitle(f'Graph Neural Network — {best_model} Candidate Detection', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/gnn_candidates.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ gnn_candidates.png saved")
print(f"{best_model} unlabeled candidates (z>2): {(z_gnn[unl_mask_arr] > 2).sum()}")

print(f"\n=== GNN Done! ===")
print(f"GCN: P@50={metrics_gcn['precision@50']:.3f}, P@100={metrics_gcn['precision@100']:.3f}")
print(f"GAT: P@50={metrics_gat['precision@50']:.3f}, P@100={metrics_gat['precision@100']:.3f}")
