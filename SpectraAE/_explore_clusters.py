"""Explore cluster data for per-cluster AE feasibility."""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

stars = pd.read_pickle('ML/_cache/stars_clustered.pkl')
cluster_ids = stars['masked_cluster_id'].values
print(f'Total stars: {len(cluster_ids)}')

feature_df = pd.read_pickle('ML/_cache/feature_df.pkl')
y_all = (feature_df['label'] == 1).values
print(f'Known CN positives: {np.sum(y_all)}')

unique_clusters, counts = np.unique(cluster_ids, return_counts=True)
print(f'Total clusters: {len(unique_clusters)}')
print(f'Size: min={counts.min()}, max={counts.max()}, median={np.median(counts):.0f}, mean={counts.mean():.1f}, std={counts.std():.1f}')

print()
buckets = [(0,30),(30,100),(100,300),(300,1000),(1000,99999)]
for lo, hi in buckets:
    in_bucket = (counts >= lo) & (counts < hi)
    n_cl = in_bucket.sum()
    n_st = counts[in_bucket].sum()
    n_pos = int(sum(y_all[cluster_ids == c].sum() for c in unique_clusters[in_bucket]))
    print(f'  [{lo:4d}, {hi:4d}): {n_cl:4d} clusters, {n_st:6d} stars, {n_pos:4d} CN+')

print()
for threshold in [30, 50, 100, 200]:
    big = unique_clusters[counts >= threshold]
    mask = np.isin(cluster_ids, big)
    n_st = mask.sum()
    n_pos = int(y_all[mask].sum())
    print(f'Clusters >= {threshold:3d}: {len(big):4d} clusters, {n_st:6d} stars ({n_st/len(cluster_ids)*100:.1f}%), {n_pos:3d} CN+ ({n_pos/max(1,y_all.sum())*100:.1f}%)')

print()
top = sorted(zip(unique_clusters, counts), key=lambda x: x[1], reverse=True)[:20]
for cid, size in top:
    n_pos = int(y_all[cluster_ids == cid].sum())
    cluster_label_id = stars[stars['masked_cluster_id'] == cid]['cluster_label_id'].values[0] if 'cluster_label_id' in stars.columns else '?'
    print(f'  Cluster {cid:3d} (label={cluster_label_id}): size={size:4d}, CN+={n_pos}')

# Check cluster_label_id column
print(f'\nStars columns: {list(stars.columns)[:10]}')
print(f'Unique cluster_label_id values: {stars["cluster_label_id"].nunique() if "cluster_label_id" in stars.columns else "N/A"}')
