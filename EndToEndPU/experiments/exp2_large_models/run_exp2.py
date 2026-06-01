"""Experiment 2: Larger Models + Stronger Regularization.

Grid sweep over model scale and regularization strength.
Compares wider/deeper ResNet variants with SWA, label smoothing, and dropout.

Usage:
    python -m EndToEndPU.experiments.exp2_large_models.run_exp2
    python -m EndToEndPU.experiments.exp2_large_models.run_exp2 --quick   # 50-epoch quick test
    python -m EndToEndPU.experiments.exp2_large_models.run_exp2 --variant wider_128d4
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import torch

warnings.filterwarnings("ignore")

_CURRENT_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _CURRENT_DIR.parent
_PROJECT_ROOT = _EXPERIMENTS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from EndToEndPU.data_loader import load_data, split_data, get_train_pos_unl, create_val_labeled_set
from EndToEndPU.evaluate import evaluate_model

from EndToEndPU.experiments.configs import LargeModelConfig
from EndToEndPU.experiments.exp2_large_models.extended_models import (
    create_model_extended, build_model_variants,
)
from EndToEndPU.experiments.exp2_large_models.extended_trainer import ExtendedPUTrainer

from EndToEndPU.experiments.utils.multi_seed import run_multi_seed
from EndToEndPU.experiments.utils.result_io import save_experiment_results, build_comparison_table

RESULTS_DIR = _EXPERIMENTS_DIR / "results" / "exp2"


# ═══════════════════════════════════════════════════════════════════════
# Single variant runner
# ═══════════════════════════════════════════════════════════════════════

def run_model_variant(config: LargeModelConfig, seed: int) -> Dict:
    """Train a single model variant with given config and seed.

    Parameters
    ----------
    config : LargeModelConfig
        Must have base_ch, res_blocks, dropout, use_swa, label_smoothing set.
    seed : int

    Returns
    -------
    dict with model, train_result, eval_result, config.
    """
    variant_label = (
        f"ch={config.base_ch} blk={config.res_blocks} drop={config.dropout}"
        f"{' SWA' if config.use_swa else ''}"
        f"{' LS=' + str(config.label_smoothing) if config.label_smoothing > 0 else ''}"
    )
    print(f"\n  Variant: {variant_label}  (seed={seed})")
    print(f"  Params: ", end="")

    device = config.device

    # Load data
    data = load_data()
    X_all = data["X_clean"]
    y_all = data["y"]
    wave = data["wave"]
    meta = data["meta"]

    # Split
    split = split_data(X_all, y_all, test_split=0.15, val_split=0.15,
                       random_seed=seed, return_test=True)
    X_train_pos, X_train_unl = get_train_pos_unl(split)
    X_val, y_val_bin = create_val_labeled_set(split)
    X_test = split["X_test"]
    y_test = split["y_test"]

    # Build model
    model = create_model_extended(
        base_ch=config.base_ch,
        res_blocks=config.res_blocks,
        dropout=config.dropout,
        se_reduction=config.se_reduction,
        use_cn_attention=True,  # always use CN attention
        cn_band_weight_init=3.0,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{n_params:,} total  |  {n_trainable:,} trainable")

    # Train
    trainer = ExtendedPUTrainer(model, config)
    train_result = trainer.train(
        X_train_pos=X_train_pos,
        X_train_unl=X_train_unl,
        X_val=X_val,
        y_val_bin=y_val_bin,
        wave=wave if config.pos_augment else None,
        verbose=True,
    )
    model = train_result["model"]

    # Evaluate
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
        "n_params": n_params,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp 2: Larger Models + Regularization")
    parser.add_argument("--quick", action="store_true", help="Quick test (50 epochs)")
    parser.add_argument("--variant", type=str, default="all",
                        help="Specific variant to run (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")

    all_variants = build_model_variants()

    # Determine which variants to run
    if args.variant == "all":
        variants_to_run = all_variants
    elif args.variant in all_variants:
        variants_to_run = {args.variant: all_variants[args.variant]}
    else:
        print(f"Unknown variant: {args.variant}")
        print(f"Available: {list(all_variants.keys())}")
        return

    # Determine seeds per variant
    # Heavy variants (256ch, 5 blocks) use fewer seeds
    heavy_variants = {"wider_256d4", "very_high_dropout_128d4_d07"}

    t_total = time.time()
    all_results = {}

    for variant_name, variant_kwargs in variants_to_run.items():
        n_seeds_variant = 1 if (args.quick or variant_name in heavy_variants) else len(args.seeds)
        seeds_variant = args.seeds[:n_seeds_variant]
        if args.quick:
            seeds_variant = [42]

        epochs = 50 if args.quick else 300
        patience = 15 if args.quick else 50

        config = LargeModelConfig(
            base_ch=variant_kwargs["base_ch"],
            res_blocks=variant_kwargs["res_blocks"],
            dropout=variant_kwargs["dropout"],
            use_swa=variant_kwargs["use_swa"],
            label_smoothing=variant_kwargs["label_smoothing"],
            n_epochs=epochs,
            early_stopping_patience=patience,
            batch_size=64 if variant_kwargs["base_ch"] <= 128 else 32,
            device=args.device,
        )

        print(f"\n{'═' * 64}")
        print(f"  {variant_name}  (seeds={seeds_variant}, epochs={epochs})")
        print(f"{'═' * 64}")

        agg = run_multi_seed(run_model_variant, config, seeds_variant, variant_name)
        all_results[variant_name] = agg

        # Save individual variant results
        safe_name = variant_name.replace("-", "_")
        _save_result(agg, config, results_dir, safe_name)

    # ── Master comparison ──
    elapsed_total = time.time() - t_total
    print(f"\n{'═' * 64}")
    print("EXPERIMENT 2 — MASTER COMPARISON")
    print(f"{'═' * 64}")
    table = build_comparison_table(all_results)
    print(table.to_string(index=False))

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


def _save_result(agg: Dict, config: LargeModelConfig, save_dir: Path, name: str):
    """Save aggregated result for a single variant."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save full aggregated result
    summary = {
        "variant": name,
        "config": {
            "base_ch": config.base_ch,
            "res_blocks": config.res_blocks,
            "dropout": config.dropout,
            "use_swa": config.use_swa,
            "label_smoothing": config.label_smoothing,
            "n_epochs": config.n_epochs,
            "weight_decay": config.weight_decay,
        },
        "mean_metrics": agg.get("mean_metrics", {}),
        "std_metrics": agg.get("std_metrics", {}),
        "n_seeds": len([s for s in agg.get("per_seed", []) if s.get("success")]),
    }
    with open(save_dir / f"summary_{name}.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
