"""Experiment 1: AE Pretraining + Fine-tuning.

Compares:
  1. Frozen AE features + XGBoost
  2. Frozen AE features + MLP
  3. AE fine-tuning (end-to-end)
  4. From-scratch baseline (same architecture, random init)
  5. Existing ResNet+CN attention baseline

Usage:
    python -m EndToEndPU.experiments.exp1_ae_pretrain.run_exp1
    python -m EndToEndPU.experiments.exp1_ae_pretrain.run_exp1 --quick  # 50-epoch quick test
"""

import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

warnings.filterwarnings("ignore")

_CURRENT_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _CURRENT_DIR.parent
_PROJECT_ROOT = _EXPERIMENTS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from EndToEndPU.data_loader import load_data, split_data, get_train_pos_unl, create_val_labeled_set
from EndToEndPU.evaluate import compute_metrics, evaluate_model
from EndToEndPU.config import EndToEndPUConfig
from EndToEndPU.models.resnet_cn_attention import create_model as create_resnet_model
from EndToEndPU.trainer import EndToEndPUTrainer

from EndToEndPU.experiments.configs import AEPretrainConfig
from EndToEndPU.experiments.exp1_ae_pretrain.ae_classifier import (
    build_ae_classifier, AEClassifier, extract_features_from_ae,
)
from EndToEndPU.experiments.exp1_ae_pretrain.train_ae_finetune import train_ae_finetune, _predict

from sklearn.metrics import average_precision_score, roc_auc_score

from EndToEndPU.experiments.utils.multi_seed import run_multi_seed
from EndToEndPU.experiments.utils.result_io import save_experiment_results, build_comparison_table

RESULTS_DIR = _EXPERIMENTS_DIR / "results" / "exp1"

# ═══════════════════════════════════════════════════════════════════════
# AE checkpoint definitions
# ═══════════════════════════════════════════════════════════════════════

AE_CHECKPOINTS = {
    "ae256": {
        "path": str(_PROJECT_ROOT / "SpectraAE" / "checkpoints" / "ae256" / "ae_best.pt"),
        "latent_dim": 256,
        "base_ch": 64,
        "description": "Vanilla AE, 256d, epoch 297",
    },
    "cn_aware_128": {
        "path": str(_PROJECT_ROOT / "SpectraAE" / "checkpoints" / "cn_aware" / "lat128" / "ae_best.pt"),
        "latent_dim": 128,
        "base_ch": 32,
        "description": "CN-aware AE, 128d, epoch 98",
    },
    "cn_aware_64": {
        "path": str(_PROJECT_ROOT / "SpectraAE" / "checkpoints" / "cn_aware" / "ae_best.pt"),
        "latent_dim": 64,
        "base_ch": 32,
        "description": "CN-aware AE, 64d, epoch 195",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Sub-experiment 1a: Frozen AE features + XGBoost
# ═══════════════════════════════════════════════════════════════════════

def run_frozen_xgboost(config: AEPretrainConfig, seed: int) -> Dict:
    """Train XGBoost PU classifier on frozen AE features."""
    ckpt_info = _resolve_checkpoint(config)
    print(f"\n  Frozen AE + XGBoost: {ckpt_info['description']}")
    device = config.device

    # 1. Load AE and extract features for all data
    model = build_ae_classifier(
        checkpoint_path=ckpt_info["path"],
        latent_dim=ckpt_info["latent_dim"],
        base_ch=ckpt_info["base_ch"],
        freeze_encoder=True,
        hidden=128,
        dropout=0.3,
        device=device,
        verbose=True,
    )
    model.to(device)

    data = load_data()
    X_all = data["X_clean"]
    y_all = data["y"]
    wave = data["wave"]
    meta = data["meta"]

    print("  Extracting AE features...")
    features = extract_features_from_ae(model, X_all, batch_size=1024, device=device)
    print(f"  Features shape: {features.shape}")

    # 2. Split data
    split = split_data(
        X_all, y_all,
        test_split=0.15, val_split=0.15,
        random_seed=seed, return_test=True,
    )

    # Get indices for train/test
    train_mask = np.zeros(len(y_all), dtype=bool)
    train_unl_mask = np.zeros(len(y_all), dtype=bool)
    val_mask = np.zeros(len(y_all), dtype=bool)
    test_mask = np.zeros(len(y_all), dtype=bool)

    # Reconstruct masks from split
    X_train = split["X_train"]
    X_val = split["X_val"]
    X_test = split["X_test"]

    # Use feature-based train/val/test by matching spectra
    # Simpler: use the split indices directly
    n_total = len(y_all)
    all_indices = np.arange(n_total)
    rng = np.random.RandomState(seed)
    rng.shuffle(all_indices)

    # Re-split features the same way
    pos_idx = np.where(y_all == 1)[0]
    unl_idx = np.where(y_all == -1)[0]
    rng.shuffle(pos_idx)

    n_test_pos = max(1, int(len(pos_idx) * 0.15))
    test_pos = pos_idx[:n_test_pos]
    remaining_pos = pos_idx[n_test_pos:]
    n_val_pos = max(1, int(len(remaining_pos) * 0.15))
    val_pos = remaining_pos[:n_val_pos]
    train_pos = remaining_pos[n_val_pos:]

    rng.shuffle(unl_idx)
    unl_test_n = max(2000, int(len(unl_idx) * 0.15))
    unl_val_n = max(1500, int(len(unl_idx) * 0.15))
    test_unl = unl_idx[:unl_test_n]
    val_unl = unl_idx[unl_test_n:unl_test_n + unl_val_n]
    train_unl = unl_idx[unl_test_n + unl_val_n:]

    train_idx = np.concatenate([train_pos, train_unl])
    val_idx = np.concatenate([val_pos, val_unl])
    test_idx = np.concatenate([test_pos, test_unl])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    # 3. Train classifier on features (XGBoost or RandomForest fallback)
    X_train_feat = features[train_idx]
    y_train_feat = y_all[train_idx]
    # PU: positives=1, unlabeled=0
    y_train_bin = (y_train_feat == 1).astype(int)

    X_val_feat = features[val_idx]
    y_val_bin = (y_all[val_idx] == 1).astype(int)

    X_test_feat = features[test_idx]
    y_test_bin = (y_all[test_idx] == 1).astype(int)

    # XGBoost with scale_pos_weight for imbalance (or RandomForest fallback)
    try:
        import xgboost as xgb
        n_pos_train = y_train_bin.sum()
        n_neg_train = len(y_train_bin) - n_pos_train
        scale_pos_weight = n_neg_train / max(n_pos_train, 1)
        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, eval_metric="aucpr", early_stopping_rounds=30,
        )
        clf.fit(X_train_feat, y_train_bin, eval_set=[(X_val_feat, y_val_bin)], verbose=False)
        test_probs = clf.predict_proba(X_test_feat)[:, 1]
        best_iter = clf.best_iteration if clf.best_iteration else 200
        _clf_type = "XGBoost"
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        n_pos_train = y_train_bin.sum()
        n_neg_train = len(y_train_bin) - n_pos_train
        class_weight = {0: 1.0, 1: n_neg_train / max(n_pos_train, 1)}
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight=class_weight,
            random_state=seed, n_jobs=-1,
        )
        clf.fit(np.concatenate([X_train_feat, X_val_feat], axis=0),
                np.concatenate([y_train_bin, y_val_bin], axis=0))
        test_probs = clf.predict_proba(X_test_feat)[:, 1]
        best_iter = 200
        _clf_type = "RandomForest"

    # 4. Evaluate
    test_metrics = compute_metrics(test_probs, y_test_bin)

    # Score all unlabeled
    all_unl_mask = y_all == -1
    unl_probs = clf.predict_proba(features[all_unl_mask])[:, 1]
    unl_meta = meta.iloc[all_unl_mask].copy()
    unl_meta["cn_prob"] = unl_probs
    candidates_df = unl_meta.sort_values("cn_prob", ascending=False).reset_index(drop=True)

    eval_result = {
        "test": test_metrics,
        "summary": f"{_clf_type} on AE{ckpt_info['latent_dim']}d feat | "
                   f"AUROC={test_metrics.get('auroc', float('nan')):.4f} "
                   f"AUPRC={test_metrics.get('auprc', float('nan')):.4f}",
        "candidates_df": candidates_df,
    }

    train_result = {
        "best_epoch": best_iter,
        "best_val_pr": 0,
        "best_val_roc": 0,
        "history": {},
        "elapsed_seconds": 0,
    }

    return {
        "model": None,
        "train_result": train_result,
        "eval_result": eval_result,
        "config": config,
        "feature_dim": ckpt_info["latent_dim"],
    }


# ═══════════════════════════════════════════════════════════════════════
# Sub-experiment 1b: Frozen AE features + MLP
# ═══════════════════════════════════════════════════════════════════════

def run_frozen_mlp(config: AEPretrainConfig, seed: int) -> Dict:
    """Train a small PyTorch MLP on frozen AE features using weighted BCE."""
    ckpt_info = _resolve_checkpoint(config)
    device = config.device
    print(f"\n  Frozen AE + MLP: {ckpt_info['description']}")

    # 1. Extract features
    ae_model = build_ae_classifier(
        checkpoint_path=ckpt_info["path"],
        latent_dim=ckpt_info["latent_dim"],
        base_ch=ckpt_info["base_ch"],
        freeze_encoder=True,
        device=device,
        verbose=False,
    )
    ae_model.to(device)

    data = load_data()
    X_all = data["X_clean"]
    y_all = data["y"]
    meta = data["meta"]

    print("  Extracting AE features...")
    features = extract_features_from_ae(ae_model, X_all, batch_size=1024, device=device)

    # 2. Split (same as frozen_xgboost)
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_all == 1)[0]
    unl_idx = np.where(y_all == -1)[0]
    rng.shuffle(pos_idx)

    n_test_pos = max(1, int(len(pos_idx) * 0.15))
    test_pos = pos_idx[:n_test_pos]
    remaining_pos = pos_idx[n_test_pos:]
    n_val_pos = max(1, int(len(remaining_pos) * 0.15))
    val_pos = remaining_pos[:n_val_pos]
    train_pos = remaining_pos[n_val_pos:]

    rng.shuffle(unl_idx)
    unl_test_n = max(2000, int(len(unl_idx) * 0.15))
    unl_val_n = max(1500, int(len(unl_idx) * 0.15))
    test_unl = unl_idx[:unl_test_n]
    val_unl = unl_idx[unl_test_n:unl_test_n + unl_val_n]
    train_unl = unl_idx[unl_test_n + unl_val_n:]

    test_idx = np.concatenate([test_pos, test_unl])
    val_idx = np.concatenate([val_pos, val_unl])

    X_train_pos_feat = features[train_pos]
    X_train_unl_feat = features[train_unl]
    X_val_feat = features[val_idx]
    y_val_bin = (y_all[val_idx] == 1).astype(np.float32)
    X_test_feat = features[test_idx]
    y_test_bin = (y_all[test_idx] == 1).astype(int)

    # 3. Train MLP on features
    latent_dim = ckpt_info["latent_dim"]
    mlp = torch.nn.Sequential(
        torch.nn.Linear(latent_dim, 128, bias=False),
        torch.nn.LayerNorm(128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(64, 1),
    ).to(device)

    for m in mlp.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.01)

    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config.n_epochs, eta_min=1e-6
    )
    bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    n_pos = len(train_pos)
    n_unl = len(train_unl)
    best_val_pr = -np.inf
    best_state = None
    best_epoch = 0
    no_improve = 0
    train_rng = np.random.RandomState(seed)

    for epoch in range(1, config.n_epochs + 1):
        mlp.train()
        # Balanced sampling
        neg_n = max(n_pos, 64)
        neg_idx = train_rng.choice(n_unl, size=neg_n, replace=False)
        X_batch = np.concatenate([X_train_pos_feat, X_train_unl_feat[neg_idx]], axis=0)
        y_batch = np.concatenate([np.ones(n_pos), np.zeros(neg_n)]).astype(np.float32)

        X_t = torch.from_numpy(X_batch).float().to(device)
        y_t = torch.from_numpy(y_batch).float().to(device)

        opt.zero_grad()
        logits = mlp(X_t).squeeze(-1)
        losses = bce(logits, y_t)
        pos_mask = y_t == 1
        if pos_mask.sum() > 0:
            losses = losses.clone()
            losses[pos_mask] *= config.positive_weight
        loss = losses.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        opt.step()
        sched.step()

        # Validate
        mlp.eval()
        with torch.no_grad():
            val_logits = mlp(torch.from_numpy(X_val_feat).float().to(device)).squeeze(-1)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_pr = average_precision_score(y_val_bin, val_probs)

        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_state = copy.deepcopy(mlp.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= config.early_stopping_patience:
            break

    mlp.load_state_dict(best_state)
    mlp.eval()

    # Evaluate
    with torch.no_grad():
        test_logits = mlp(torch.from_numpy(X_test_feat).float().to(device)).squeeze(-1)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
    test_metrics = compute_metrics(test_probs, y_test_bin)

    all_unl_mask = y_all == -1
    with torch.no_grad():
        unl_logits = mlp(torch.from_numpy(features[all_unl_mask]).float().to(device)).squeeze(-1)
        unl_probs = torch.sigmoid(unl_logits).cpu().numpy()
    unl_meta = meta.iloc[all_unl_mask].copy()
    unl_meta["cn_prob"] = unl_probs
    candidates_df = unl_meta.sort_values("cn_prob", ascending=False).reset_index(drop=True)

    return {
        "model": mlp,
        "train_result": {"best_epoch": best_epoch, "best_val_pr": best_val_pr,
                         "best_val_roc": 0, "history": {}, "elapsed_seconds": 0},
        "eval_result": {"test": test_metrics, "summary": f"MLP on AE{latent_dim}d | "
                        f"AUROC={test_metrics.get('auroc', float('nan')):.4f}",
                        "candidates_df": candidates_df},
        "config": config,
    }


# ═══════════════════════════════════════════════════════════════════════
# Sub-experiment 1c: AE fine-tuning (end-to-end)
# ═══════════════════════════════════════════════════════════════════════

def run_ae_finetune(config: AEPretrainConfig, seed: int) -> Dict:
    """Fine-tune AEClassifier end-to-end on PU task."""
    ckpt_info = _resolve_checkpoint(config)
    device = config.device
    print(f"\n  AE Fine-tune: {ckpt_info['description']}")

    # 1. Build model with unfrozen encoder
    model = build_ae_classifier(
        checkpoint_path=ckpt_info["path"],
        latent_dim=ckpt_info["latent_dim"],
        base_ch=ckpt_info["base_ch"],
        freeze_encoder=False,  # fine-tune!
        hidden=config.classifier_hidden,
        dropout=config.classifier_dropout,
        device=device,
    )

    # 2. Load and split data
    data = load_data()
    X_all = data["X_clean"]
    y_all = data["y"]
    wave = data["wave"]
    meta = data["meta"]

    split = split_data(X_all, y_all, test_split=0.15, val_split=0.15,
                       random_seed=seed, return_test=True)

    X_train_pos, X_train_unl = get_train_pos_unl(split)
    X_val, y_val_bin = create_val_labeled_set(split)
    X_test = split["X_test"]
    y_test = split["y_test"]

    # 3. Fine-tune
    train_result = train_ae_finetune(
        model, config,
        X_train_pos=X_train_pos,
        X_train_unl=X_train_unl,
        X_val=X_val,
        y_val_bin=y_val_bin,
        wave=wave,
        verbose=True,
    )
    model = train_result["model"]

    # 4. Evaluate (reuse EndToEndPU evaluate)
    eval_result = evaluate_model(
        model, X_test=X_test, y_test=y_test,
        X_train_pos=X_train_pos, X_train_unl=X_train_unl,
        meta=meta, device=device,
    )

    print(eval_result.get("summary", ""))

    return {
        "model": model,
        "train_result": train_result,
        "eval_result": eval_result,
        "config": config,
    }


# ═══════════════════════════════════════════════════════════════════════
# Sub-experiment 1d: From-scratch AEClassifier (no pretraining)
# ═══════════════════════════════════════════════════════════════════════

def run_from_scratch(config: AEPretrainConfig, seed: int) -> Dict:
    """Train AEClassifier from random init (no pretrained weights)."""
    device = config.device
    ckpt_info = _resolve_checkpoint(config)
    print(f"\n  From-scratch AEClassifier (AE{ckpt_info['latent_dim']}d arch, random init)")

    # Build same architecture but random init
    from EndToEndPU.experiments.exp1_ae_pretrain.ae_classifier import Encoder as _EncoderDirect
    Encoder = _EncoderDirect
    encoder = Encoder(in_channels=1, base_ch=ckpt_info["base_ch"], latent_dim=ckpt_info["latent_dim"])
    model = AEClassifier(
        encoder=encoder,
        latent_dim=ckpt_info["latent_dim"],
        hidden=config.classifier_hidden,
        dropout=config.classifier_dropout,
    )

    # Load and split
    data = load_data()
    X_all = data["X_clean"]
    y_all = data["y"]
    wave = data["wave"]
    meta = data["meta"]

    split = split_data(X_all, y_all, test_split=0.15, val_split=0.15,
                       random_seed=seed, return_test=True)
    X_train_pos, X_train_unl = get_train_pos_unl(split)
    X_val, y_val_bin = create_val_labeled_set(split)
    X_test = split["X_test"]
    y_test = split["y_test"]

    train_result = train_ae_finetune(
        model, config,
        X_train_pos=X_train_pos,
        X_train_unl=X_train_unl,
        X_val=X_val,
        y_val_bin=y_val_bin,
        wave=wave,
        verbose=True,
    )
    model = train_result["model"]

    eval_result = evaluate_model(
        model, X_test=X_test, y_test=y_test,
        X_train_pos=X_train_pos, X_train_unl=X_train_unl,
        meta=meta, device=device,
    )
    print(eval_result.get("summary", ""))

    return {
        "model": model,
        "train_result": train_result,
        "eval_result": eval_result,
        "config": config,
    }


# ═══════════════════════════════════════════════════════════════════════
# Baseline: Existing ResNet + CN attention
# ═══════════════════════════════════════════════════════════════════════

def run_resnet_baseline(config: AEPretrainConfig, seed: int) -> Dict:
    """Run the existing ResNet + CN attention model as baseline."""
    print(f"\n  ResNet + CN Attention baseline (seed={seed})")

    resnet_config = EndToEndPUConfig(
        use_cn_attention=True,
        cn_band_weight_init=3.0,
        base_ch=64,
        dropout=0.3,
        se_reduction=8,
        res_blocks=4,
        learning_rate=1e-3,
        weight_decay=1e-2,
        batch_size=64,
        n_epochs=config.n_epochs,
        early_stopping_patience=config.early_stopping_patience,
        loss_mode="weighted_bce",
        positive_weight=10.0,
        neg_ratio=1.0,
        grad_clip=1.0,
        hard_neg_frac=0.3,
        mixup_alpha=0.2,
        pos_augment=True,
        random_seed=seed,
        device=config.device,
    )

    data = load_data()
    X_all = data["X_clean"]
    y_all = data["y"]
    wave = data["wave"]
    meta = data["meta"]

    split = split_data(X_all, y_all, test_split=0.15, val_split=0.15,
                       random_seed=seed, return_test=True)
    X_train_pos, X_train_unl = get_train_pos_unl(split)
    X_val, y_val_bin = create_val_labeled_set(split)
    X_test = split["X_test"]
    y_test = split["y_test"]

    model = create_resnet_model(resnet_config)

    trainer = EndToEndPUTrainer(model, resnet_config)
    train_result = trainer.train(
        X_train_pos=X_train_pos,
        X_train_unl=X_train_unl,
        X_val=X_val,
        y_val_bin=y_val_bin,
        wave=wave,
        verbose=False,
    )
    model = train_result["model"]

    eval_result = evaluate_model(
        model, X_test=X_test, y_test=y_test,
        X_train_pos=X_train_pos, X_train_unl=X_train_unl,
        meta=meta, device=config.device,
    )

    return {
        "model": model,
        "train_result": train_result,
        "eval_result": eval_result,
        "config": resnet_config,
    }


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _resolve_checkpoint(config: AEPretrainConfig) -> Dict:
    """Resolve checkpoint info from config, checking if file exists."""
    ckpt_path = Path(config.ae_checkpoint_path)
    if not ckpt_path.exists():
        ckpt_path = _PROJECT_ROOT / config.ae_checkpoint_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {config.ae_checkpoint_path}")

    return {
        "path": str(ckpt_path),
        "latent_dim": config.ae_latent_dim,
        "base_ch": config.ae_base_ch,
        "description": f"AE{config.ae_latent_dim}d (base_ch={config.ae_base_ch})",
    }


def main():
    parser = argparse.ArgumentParser(description="Exp 1: AE Pretraining + Fine-tuning")
    parser.add_argument("--quick", action="store_true", help="Quick test (50 epochs)")
    parser.add_argument("--ae-variant", type=str, default="ae256",
                        choices=["ae256", "cn_aware_128", "cn_aware_64", "all"],
                        help="Which AE checkpoint to test")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "frozen_xgb", "frozen_mlp", "finetune", "scratch", "resnet"],
                        help="Which sub-experiment to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")

    epochs = 50 if args.quick else 300
    patience = 15 if args.quick else 50
    seeds = args.seeds if not args.quick else [42]

    # Build config
    base_config = AEPretrainConfig(
        n_epochs=epochs,
        early_stopping_patience=patience,
        device=args.device,
    )

    t_total = time.time()
    all_results = {}

    ae_variants = (
        ["ae256", "cn_aware_128", "cn_aware_64"] if args.ae_variant == "all"
        else [args.ae_variant]
    )

    for ae_key in ae_variants:
        ckpt_info = AE_CHECKPOINTS[ae_key]
        config = AEPretrainConfig(
            ae_checkpoint_path=ckpt_info["path"],
            ae_latent_dim=ckpt_info["latent_dim"],
            ae_base_ch=ckpt_info["base_ch"],
            n_epochs=epochs,
            early_stopping_patience=patience,
            device=args.device,
        )

        if args.mode in ("all", "frozen_xgb"):
            name = f"AE-{ae_key}-frozen-XGB"
            print(f"\n{'═' * 64}\n  {name}\n{'═' * 64}")
            agg = run_multi_seed(run_frozen_xgboost, config, seeds, name)
            all_results[name] = agg
            _save_agg(agg, results_dir, name)

        if args.mode in ("all", "frozen_mlp"):
            name = f"AE-{ae_key}-frozen-MLP"
            print(f"\n{'═' * 64}\n  {name}\n{'═' * 64}")
            agg = run_multi_seed(run_frozen_mlp, config, seeds, name)
            all_results[name] = agg
            _save_agg(agg, results_dir, name)

        if args.mode in ("all", "finetune"):
            name = f"AE-{ae_key}-finetune"
            print(f"\n{'═' * 64}\n  {name}\n{'═' * 64}")
            agg = run_multi_seed(run_ae_finetune, config, seeds, name)
            all_results[name] = agg
            _save_agg(agg, results_dir, name)

        if args.mode in ("all", "scratch"):
            name = f"AE-{ae_key}-scratch"
            print(f"\n{'═' * 64}\n  {name}\n{'═' * 64}")
            agg = run_multi_seed(run_from_scratch, config, seeds, name)
            all_results[name] = agg
            _save_agg(agg, results_dir, name)

    if args.mode in ("all", "resnet"):
        name = "ResNet-CN-Attention"
        print(f"\n{'═' * 64}\n  {name} (baseline)\n{'═' * 64}")
        agg = run_multi_seed(run_resnet_baseline, base_config, seeds, name)
        all_results[name] = agg
        _save_agg(agg, results_dir, name)

    # ── Master comparison ──
    elapsed_total = time.time() - t_total
    print(f"\n{'═' * 64}")
    print(f"EXPERIMENT 1 — MASTER COMPARISON")
    print(f"{'═' * 64}")
    table = build_comparison_table(all_results)
    print(table.to_string(index=False))

    # Save comparison table
    table.to_csv(results_dir / "comparison_table.csv", index=False)
    with open(results_dir / "all_results.json", "w") as f:
        json.dump({
            name: {"mean_metrics": r.get("mean_metrics", {}),
                   "std_metrics": r.get("std_metrics", {}),
                   "n_seeds": len([s for s in r.get("per_seed", []) if s.get("success")])}
            for name, r in all_results.items()
        }, f, indent=2)

    print(f"\nTotal time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    print(f"Results saved to: {results_dir}")


def _save_agg(agg: Dict, save_dir: Path, name: str):
    """Save aggregated multi-seed results."""
    save_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace("-", "_").replace(" ", "_").lower()
    with open(save_dir / f"agg_{safe_name}.json", "w") as f:
        json.dump({
            "experiment_name": name,
            "mean_metrics": agg.get("mean_metrics", {}),
            "std_metrics": agg.get("std_metrics", {}),
            "n_seeds": len([s for s in agg.get("per_seed", []) if s.get("success")]),
            "elapsed_seconds": agg.get("elapsed_seconds", 0),
        }, f, indent=2)


if __name__ == "__main__":
    main()
