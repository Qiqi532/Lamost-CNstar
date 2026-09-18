#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""nnPU vs weighted_bce 对照实验（新缓存 dr13_all_cache）。

与 EndToEndPU_Method.ipynb 的数据加载 / 划分 / 超参完全一致，
仅对比 loss_mode = 'weighted_bce' 与 'nnpu' 两种损失函数。
"""

import sys, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _ in range(6):
    if (_PROJECT_ROOT / 'ML').exists() and (_PROJECT_ROOT / 'Data').exists():
        break
    _PROJECT_ROOT = _PROJECT_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from EndToEndPU.config import EndToEndPUConfig
from EndToEndPU.models.resnet_cn_attention import create_model
from EndToEndPU.trainer import EndToEndPUTrainer

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'PyTorch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}  |  device={DEVICE}')

# ── 数据加载（新缓存）──
from build_dr13_all_cache import load_dr13_all_cache
data = load_dr13_all_cache()
X_clean = data['X_clean'].astype(np.float32)
stars_clean = data['stars_clean']
common_wave = data['common_wave'].astype(np.float64)

y = stars_clean['label'].values.astype(int)   # 1 = CN, -1 = unlabeled
N = len(y)
n_pos = int((y == 1).sum())

# ── 划分（与 notebook 完全一致）──
rng = np.random.RandomState(SEED)
pos_idx = np.where(y == 1)[0]
unl_idx = np.where(y == -1)[0]
rng.shuffle(pos_idx)

n_test = max(1, int(len(pos_idx) * 0.15))
test_pos = pos_idx[:n_test]
rest_pos = pos_idx[n_test:]
n_val = max(1, int(len(rest_pos) * 0.15))
val_pos = rest_pos[:n_val]
train_pos = rest_pos[n_val:]

rng.shuffle(unl_idx)
n_test_unl = max(2000, int(len(unl_idx) * 0.15))
n_val_unl = max(1500, int(len(unl_idx) * 0.15))
test_unl = unl_idx[:n_test_unl]
val_unl = unl_idx[n_test_unl:n_test_unl + n_val_unl]
train_unl = unl_idx[n_test_unl + n_val_unl:]

val_idx = np.concatenate([val_pos, val_unl])
test_idx = np.concatenate([test_pos, test_unl])

X_train_pos = X_clean[train_pos]
X_train_unl = X_clean[train_unl]
X_val = X_clean[val_idx]
y_val_bin = (y[val_idx] == 1).astype(np.float32)
X_test = X_clean[test_idx]
y_test_bin = (y[test_idx] == 1).astype(int)

print(f"数据: {N:,} 星 / {n_pos} CN | "
      f"训练 {len(train_pos)} 正 / {len(train_unl):,} 未标注 | "
      f"验证 {len(val_pos)} 正 | 测试 {len(test_pos)} 正")


def make_config(loss_mode):
    return EndToEndPUConfig(
        use_cn_attention=True, cn_band_weight_init=3.0,
        base_ch=64, dropout=0.3, se_reduction=8, res_blocks=4,
        learning_rate=1e-3, weight_decay=1e-2, batch_size=64,
        n_epochs=200, early_stopping_patience=40,
        loss_mode=loss_mode, positive_weight=10.0, neg_ratio=1.0,
        mixup_alpha=0.2, pos_augment=True, grad_clip=1.0,
        hard_neg_frac=0.3, random_seed=SEED, device=DEVICE,
    )


@torch.no_grad()
def predict(model, X, batch_size=1024):
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).float().to(DEVICE)
        preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def evaluate(model):
    tp = predict(model, X_test)
    auroc = roc_auc_score(y_test_bin, tp)
    auprc = average_precision_score(y_test_bin, tp)
    order = np.argsort(tp)[::-1]
    def pk(k):
        kk = min(k, len(order)); return y_test_bin[order[:kk]].mean()
    ap = predict(model, X_clean)
    ks = ap[y == 1]
    thr = float(np.nanquantile(ks, 0.05))
    n_cand = int(((y == -1) & (ap >= thr)).sum())
    return dict(auroc=auroc, auprc=auprc, p50=pk(50), p100=pk(100),
                p200=pk(200), thr=thr, n_cand=n_cand,
                ks_min=float(ks.min()), ks_med=float(np.nanmedian(ks)))


results = {}
for mode in ["weighted_bce", "nnpu"]:
    print(f"\n{'=' * 60}\n  训练 {mode}\n{'=' * 60}", flush=True)
    cfg = make_config(mode)
    model = create_model(cfg)
    trainer = EndToEndPUTrainer(model, cfg)
    t0 = time.time()
    res = trainer.train(
        X_train_pos=X_train_pos, X_train_unl=X_train_unl,
        X_val=X_val, y_val_bin=y_val_bin,
        wave=common_wave if cfg.pos_augment else None,
        verbose=True,
    )
    model = res['model']
    r = evaluate(model)
    r['val_pr'] = res['best_val_pr']
    r['val_roc'] = res['best_val_roc']
    r['best_epoch'] = res['best_epoch']
    r['elapsed'] = time.time() - t0
    results[mode] = r
    print(f"\n[{mode}] 最优 epoch={r['best_epoch']}  val PR={r['val_pr']:.4f}  "
          f"ROC={r['val_roc']:.4f}  ({r['elapsed']:.0f}s)", flush=True)

print("\n\n" + "=" * 64)
print("对照结果 (测试集):")
print(f"  {'':14s} {'AUROC':>8s} {'AUPRC':>8s} {'P@50':>7s} {'P@100':>7s} "
      f"{'P@200':>7s} {'阈值':>7s} {'候选':>6s}")
for mode, r in results.items():
    print(f"  {mode:14s} {r['auroc']:>8.4f} {r['auprc']:>8.4f} {r['p50']:>7.4f} "
          f"{r['p100']:>7.4f} {r['p200']:>7.4f} {r['thr']:>7.4f} {r['n_cand']:>6d}")
print("=" * 64)
for mode, r in results.items():
    print(f"  [{mode}] 已知 CN 星概率 min={r['ks_min']:.3f}  median={r['ks_med']:.3f}")
