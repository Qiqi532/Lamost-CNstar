"""End-to-End PU Deep Network — 1D ResNet + CN Attention + nnPU Loss.

Main training script.  Trains and evaluates the model, comparing:
  1. Full model (CN attention + SE)
  2. Baseline (no CN attention)
  3. Ablation variants

Usage:
    python -m EndToEndPU.train                    # train with default config
    python -m EndToEndPU.train --no-cn-attention  # baseline without CN attention
    python -m EndToEndPU.train --cn-weight 5.0    # stronger CN focus
    python -m EndToEndPU.train --epochs 500       # longer training
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch

warnings.filterwarnings("ignore")

# ── Path setup ──
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from EndToEndPU.config import EndToEndPUConfig
from EndToEndPU.data_loader import load_data, split_data, get_train_pos_unl, create_val_labeled_set
from EndToEndPU.models.resnet_cn_attention import create_model
from EndToEndPU.trainer import EndToEndPUTrainer
from EndToEndPU.evaluate import evaluate_model, export_candidates, compute_metrics

RESULTS_DIR = _CURRENT_DIR / "results"


def setup_results_dir() -> Path:
    """Create timestamped results directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_experiment(
    config: EndToEndPUConfig,
    experiment_name: str = "default",
    save_dir: Path = None,
) -> Dict:
    """Run a single training experiment.

    Parameters
    ----------
    config : EndToEndPUConfig
        Training configuration.
    experiment_name : str
        Label for this experiment.
    save_dir : Path or None
        Directory to save results (model weights, metrics, etc.).

    Returns
    -------
    dict with keys: model, train_result, eval_result, config.
    """
    print(f"\n{'=' * 64}")
    print(f"Experiment: {experiment_name}")
    print(f"  CN attention: {config.use_cn_attention}")
    print(f"  CN weight init: {config.cn_band_weight_init}")
    print(f"  nnPU clamp: {config.nnpu_clamp}")
    print(f"  Mixup α: {config.mixup_alpha}")
    print(f"  Pos augment: {config.pos_augment}")
    print(f"  Device: {config.device}")
    print(f"{'=' * 64}")

    # ── 0. Reproducibility ──
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    # ── 1. Load data ──
    print("\n[1/5] Loading data...")
    data = load_data()
    X_all = data["X_clean"]
    y_all = data["y"]
    wave = data["wave"]
    meta = data["meta"]

    # ── 2. Split ──
    print("\n[2/5] Splitting data...")
    split = split_data(
        X_all, y_all,
        test_split=0.15,     # ~11 positives held out
        val_split=0.15,      # ~9 positives for validation
        random_seed=config.random_seed,
        return_test=True,
    )

    X_train_pos, X_train_unl = get_train_pos_unl(split)
    X_val, y_val_bin = create_val_labeled_set(split)
    X_test = split["X_test"]
    y_test = split["y_test"]

    # ── 3. Create model ──
    print("\n[3/5] Creating model...")
    model = create_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total  |  {n_trainable:,} trainable")

    # ── 4. Train ──
    print("\n[4/5] Training...")
    trainer = EndToEndPUTrainer(model, config)
    train_result = trainer.train(
        X_train_pos=X_train_pos,
        X_train_unl=X_train_unl,
        X_val=X_val,
        y_val_bin=y_val_bin,
        wave=wave if config.pos_augment else None,
        verbose=True,
    )
    model = train_result["model"]

    # ── 5. Evaluate ──
    print("\n[5/5] Evaluating...")
    eval_result = evaluate_model(
        model,
        X_test=X_test,
        y_test=y_test,
        X_train_pos=X_train_pos,
        X_train_unl=X_train_unl,
        meta=meta,
        device=config.device,
    )

    print(eval_result.get("summary", ""))

    # ── Save results ──
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = save_dir / f"model_{experiment_name}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "experiment_name": experiment_name,
            "best_epoch": train_result["best_epoch"],
            "best_val_pr": train_result["best_val_pr"],
            "best_val_roc": train_result["best_val_roc"],
        }, model_path)
        print(f"\nModel saved to: {model_path}")

        # Save metrics
        metrics = {
            "experiment_name": experiment_name,
            "config": {
                k: str(v) if isinstance(v, (list, type(None))) else v
                for k, v in config.__dict__.items()
            },
            "best_epoch": train_result["best_epoch"],
            "best_val_pr": float(train_result["best_val_pr"]),
            "best_val_roc": float(train_result["best_val_roc"]),
            "elapsed_seconds": train_result["elapsed_seconds"],
            "n_params": n_params,
        }
        if "test" in eval_result:
            t = eval_result["test"]
            metrics["test_auroc"] = float(t["auroc"]) if not np.isnan(t["auroc"]) else None
            metrics["test_auprc"] = float(t["auprc"]) if not np.isnan(t["auprc"]) else None
            for k in [50, 100, 200]:
                metrics[f"test_precision@{k}"] = float(t.get(f"precision@{k}", 0))
                metrics[f"test_recall@{k}"] = float(t.get(f"recall@{k}", 0))

        with open(save_dir / f"metrics_{experiment_name}.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Export candidates
        if "candidates_df" in eval_result:
            export_candidates(
                eval_result["candidates_df"],
                top_n=200,
                output_csv=str(save_dir / f"candidates_{experiment_name}.csv"),
            )

        # Save training history
        history = {
            k: [float(x) for x in v]
            for k, v in train_result["history"].items()
        }
        with open(save_dir / f"history_{experiment_name}.json", "w") as f:
            json.dump(history, f, indent=2)

    return {
        "model": model,
        "train_result": train_result,
        "eval_result": eval_result,
        "config": config,
    }


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End PU Deep Network: 1D ResNet + CN Attention + nnPU"
    )
    # Model
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--se-reduction", type=int, default=8)
    parser.add_argument("--no-cn-attention", action="store_true",
                        help="Disable CN attention (baseline)")
    parser.add_argument("--cn-weight", type=float, default=3.0,
                        help="Initial CN band attention weight")
    parser.add_argument("--freeze-cn", action="store_true",
                        help="Freeze CN attention weights")
    parser.add_argument("--res-blocks", type=int, default=4, choices=[3, 4])
    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    # Loss
    parser.add_argument("--loss-mode", type=str, default="weighted_bce",
                        choices=["weighted_bce", "nnpu", "upu"],
                        help="Loss function: weighted_bce (default), nnpu, upu")
    parser.add_argument("--pos-weight", type=float, default=10.0,
                        help="Positive class weight for weighted_bce")
    parser.add_argument("--neg-ratio", type=float, default=1.0,
                        help="Ratio of pseudo-negatives to positives per epoch (1.0 = balanced)")
    parser.add_argument("--pi-p", type=float, default=None,
                        help="Class prior (auto if not set)")
    parser.add_argument("--no-clamp", action="store_true",
                        help="Disable nnPU clamping (use uPU)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping norm (0=disable)")
    parser.add_argument("--hard-neg-frac", type=float, default=0.3,
                        help="Fraction of hard negatives per epoch")
    # Regularization
    parser.add_argument("--no-mixup", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--augment-factor", type=int, default=20)
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Run both full model and no-CN-attention baseline")
    args = parser.parse_args()

    # ── Build config ──
    config = EndToEndPUConfig(
        base_ch=args.base_ch,
        dropout=args.dropout,
        se_reduction=args.se_reduction,
        use_cn_attention=not args.no_cn_attention,
        cn_band_weight_init=args.cn_weight,
        freeze_cn_attention=args.freeze_cn,
        res_blocks=args.res_blocks,
        learning_rate=args.lr,
        weight_decay=args.wd,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        early_stopping_patience=args.patience,
        pi_p=args.pi_p,
        nnpu_clamp=not args.no_clamp,
        loss_mode=args.loss_mode,
        positive_weight=args.pos_weight,
        neg_ratio=args.neg_ratio,
        grad_clip=args.grad_clip,
        hard_neg_frac=args.hard_neg_frac,
        mixup_alpha=0.0 if args.no_mixup else 0.2,
        pos_augment=not args.no_augment,
        pos_augment_factor=args.augment_factor,
        random_seed=args.seed,
        device=args.device,
    )

    save_dir = setup_results_dir()
    print(f"Results directory: {save_dir}")

    t_total = time.time()

    if args.compare_baseline:
        # Run both experiments
        exp_name_full = args.name or "full_model"
        exp_name_base = "baseline_no_cn_attn"

        print("\n" + "=" * 64)
        print("RUN 1/2: Full model (CN attention)")
        print("=" * 64)
        result_full = run_experiment(config, exp_name_full, save_dir)

        config_baseline = EndToEndPUConfig(
            use_cn_attention=False,
            base_ch=args.base_ch,
            dropout=args.dropout,
            se_reduction=args.se_reduction,
            res_blocks=args.res_blocks,
            learning_rate=args.lr,
            weight_decay=args.wd,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            early_stopping_patience=args.patience,
            pi_p=args.pi_p,
            nnpu_clamp=not args.no_clamp,
            loss_mode=args.loss_mode,
            positive_weight=args.pos_weight,
            neg_ratio=args.neg_ratio,
            grad_clip=args.grad_clip,
            hard_neg_frac=args.hard_neg_frac,
            mixup_alpha=0.0 if args.no_mixup else 0.2,
            pos_augment=not args.no_augment,
            pos_augment_factor=args.augment_factor,
            random_seed=args.seed,
            device=args.device,
        )

        print("\n" + "=" * 64)
        print("RUN 2/2: Baseline (no CN attention)")
        print("=" * 64)
        result_baseline = run_experiment(config_baseline, exp_name_base, save_dir)

        # ── Comparison summary ──
        print("\n" + "=" * 64)
        print("COMPARISON SUMMARY")
        print("=" * 64)
        compare = [
            ("Full (CN attn)", result_full),
            ("Baseline (no CN)", result_baseline),
        ]
        for name, r in compare:
            tr = r["train_result"]
            ev = r["eval_result"]
            test = ev.get("test", {})
            print(f"\n  {name}:")
            print(f"    Val PR-AUC:    {tr['best_val_pr']:.4f}  (ep {tr['best_epoch']})")
            print(f"    Val ROC-AUC:   {tr['best_val_roc']:.4f}")
            if test:
                print(f"    Test PR-AUC:   {test.get('auprc', float('nan')):.4f}")
                print(f"    Test ROC-AUC:  {test.get('auroc', float('nan')):.4f}")
                print(f"    P@50/P@100:    {test.get('precision@50', 0):.4f} / "
                      f"{test.get('precision@100', 0):.4f}")

        # Save comparison
        comparison = {
            "full_model": {
                "val_pr_auc": float(result_full["train_result"]["best_val_pr"]),
                "val_roc_auc": float(result_full["train_result"]["best_val_roc"]),
            },
            "baseline": {
                "val_pr_auc": float(result_baseline["train_result"]["best_val_pr"]),
                "val_roc_auc": float(result_baseline["train_result"]["best_val_roc"]),
            },
        }
        if "test" in result_full["eval_result"]:
            comparison["full_model"]["test_pr_auc"] = float(
                result_full["eval_result"]["test"]["auprc"]
            )
            comparison["baseline"]["test_pr_auc"] = float(
                result_baseline["eval_result"]["test"]["auprc"]
            )
        with open(save_dir / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {save_dir / 'comparison.json'}")

    else:
        exp_name = args.name or ("cn_attention" if config.use_cn_attention else "baseline")
        result = run_experiment(config, exp_name, save_dir)
        train_result = result["train_result"]
        print(f"\nFinal: val PR={train_result['best_val_pr']:.4f}  "
              f"ROC={train_result['best_val_roc']:.4f}  "
              f"best epoch={train_result['best_epoch']}")

    elapsed_total = time.time() - t_total
    print(f"\nTotal time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
