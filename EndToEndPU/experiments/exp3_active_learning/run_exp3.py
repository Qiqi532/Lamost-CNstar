"""Experiment 3: Active Learning for PU Classification.

Compares 4 selection strategies:
  1. Confidence-based: select top-K highest predicted probability
  2. Uncertainty-based: select K with probability closest to 0.5
  3. Committee-based: select K with highest prediction variance (3 models)
  4. Random baseline: select K randomly

Also includes baselines:
  - All 54 positives (upper bound)
  - 34 initial only (lower bound)

Usage:
    python -m EndToEndPU.experiments.exp3_active_learning.run_exp3
    python -m EndToEndPU.experiments.exp3_active_learning.run_exp3 --quick  # 100-epoch quick test
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
from EndToEndPU.models.resnet_cn_attention import create_model as create_resnet_model
from EndToEndPU.trainer import EndToEndPUTrainer
from EndToEndPU.config import EndToEndPUConfig

from EndToEndPU.experiments.configs import ActiveLearningConfig
from EndToEndPU.experiments.exp3_active_learning.active_trainer import (
    active_learning_loop, evaluate_on_test,
)
from EndToEndPU.experiments.exp3_active_learning.active_selection import select_candidates

from EndToEndPU.experiments.utils.multi_seed import run_multi_seed
from EndToEndPU.experiments.utils.result_io import save_experiment_results, build_comparison_table

RESULTS_DIR = _EXPERIMENTS_DIR / "results" / "exp3"


# ═══════════════════════════════════════════════════════════════════════
# Model factory
# ═══════════════════════════════════════════════════════════════════════

def _model_factory(config, seed: int):
    """Create a fresh ResNet+CN attention model for each active learning round."""
    model_config = EndToEndPUConfig(
        base_ch=config.base_ch,
        dropout=config.dropout,
        se_reduction=config.se_reduction,
        use_cn_attention=config.use_cn_attention,
        cn_band_weight_init=config.cn_band_weight_init,
        res_blocks=config.res_blocks,
        device=config.device,
    )
    return create_resnet_model(model_config)


# ═══════════════════════════════════════════════════════════════════════
# Runner for a single strategy + seed
# ═══════════════════════════════════════════════════════════════════════

def run_active_strategy(config: ActiveLearningConfig, seed: int) -> Dict:
    """Run active learning with a specific strategy and seed.

    Returns dict with active_learning_result + eval_result.
    """
    strategy = config.strategy
    print(f"\n  Active Learning: strategy={strategy}, seed={seed}")

    # Load data
    data = load_data()
    X_all = data["X_clean"]
    y_all = data["y"]
    wave = data["wave"]
    meta = data["meta"]

    # Run active learning loop
    al_result = active_learning_loop(
        config=config,
        X_all=X_all,
        y_all=y_all,
        meta=meta,
        wave=wave,
        model_factory=lambda s: _model_factory(config, s),
        verbose=True,
    )

    # Extract final round metrics
    round_metrics = al_result["round_metrics"]
    if round_metrics:
        final_metrics = round_metrics[-1]
    else:
        final_metrics = {}

    eval_result = {
        "test": final_metrics,
        "summary": f"Active Learning ({strategy}): "
                   f"AUROC={final_metrics.get('auroc', float('nan')):.4f} "
                   f"AUPRC={final_metrics.get('auprc', float('nan')):.4f}",
        "round_metrics": round_metrics,
    }

    train_result = {
        "best_epoch": 0,
        "best_val_pr": final_metrics.get("auprc", 0),
        "best_val_roc": final_metrics.get("auroc", 0),
        "history": {},
        "elapsed_seconds": 0,
    }

    return {
        "model": al_result.get("final_model"),
        "train_result": train_result,
        "eval_result": eval_result,
        "config": config,
        "active_learning": al_result,
    }


# ═══════════════════════════════════════════════════════════════════════
# Baseline: All 54 positives (upper bound)
# ═══════════════════════════════════════════════════════════════════════

def run_all_pos_baseline(config: ActiveLearningConfig, seed: int) -> Dict:
    """Train with all available training positives (no active learning)."""
    print(f"\n  Baseline: All training positives (seed={seed})")

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
        n_epochs=config.n_epochs * (config.n_rounds + 1),  # match total training budget
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
        X_train_pos=X_train_pos, X_train_unl=X_train_unl,
        X_val=X_val, y_val_bin=y_val_bin, wave=wave, verbose=False,
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
        "config": config,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp 3: Active Learning")
    parser.add_argument("--quick", action="store_true", help="Quick test (100 epochs)")
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["all", "confidence", "uncertainty", "committee", "random",
                                 "all_pos", "initial_only"],
                        help="Which strategy to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")

    epochs = 100 if args.quick else 200
    patience = 20 if args.quick else 40
    seeds = args.seeds if not args.quick else [42]

    strategies = (
        ["confidence", "uncertainty", "committee", "random"]
        if args.strategy == "all"
        else [args.strategy]
    )

    t_total = time.time()
    all_results = {}

    for strategy in strategies:
        config = ActiveLearningConfig(
            strategy=strategy,
            n_epochs=epochs,
            early_stopping_patience=patience,
            device=args.device,
        )
        name = f"AL-{strategy}"
        print(f"\n{'═' * 64}\n  {name}\n{'═' * 64}")

        # Committee strategy takes longer (3x models per round)
        n_seeds_use = [seeds[0]] if (args.quick or strategy == "committee") else seeds

        agg = run_multi_seed(run_active_strategy, config, n_seeds_use, name)
        all_results[name] = agg
        _save_al_result(agg, results_dir, name)

    # Baselines
    base_config = ActiveLearningConfig(
        n_epochs=epochs,
        early_stopping_patience=patience,
        device=args.device,
    )

    for baseline_name, baseline_fn in [
        ("All-54-positives", run_all_pos_baseline),
    ]:
        if args.strategy in ("all", "all_pos"):
            print(f"\n{'═' * 64}\n  {baseline_name}\n{'═' * 64}")
            agg = run_multi_seed(baseline_fn, base_config, seeds, baseline_name)
            all_results[baseline_name] = agg
            _save_al_result(agg, results_dir, baseline_name)

    # ── Master comparison ──
    elapsed_total = time.time() - t_total
    print(f"\n{'═' * 64}")
    print("EXPERIMENT 3 — MASTER COMPARISON")
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

    # Also save per-round metrics for AL strategies
    for name, agg in all_results.items():
        if name.startswith("AL-"):
            _save_round_metrics(agg, results_dir, name)

    print(f"\nTotal time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    print(f"Results saved to: {results_dir}")


def _make_json_safe(obj):
    """Recursively convert numpy arrays/scalars to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _save_al_result(agg: Dict, save_dir: Path, name: str):
    """Save aggregated active learning result."""
    save_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace("-", "_").replace(" ", "_").lower()
    with open(save_dir / f"agg_{safe_name}.json", "w") as f:
        json.dump(_make_json_safe({
            "experiment_name": name,
            "mean_metrics": agg.get("mean_metrics", {}),
            "std_metrics": agg.get("std_metrics", {}),
            "n_seeds": len([s for s in agg.get("per_seed", []) if s.get("success")]),
            "elapsed_seconds": agg.get("elapsed_seconds", 0),
        }), f, indent=2)


def _save_round_metrics(agg: Dict, save_dir: Path, name: str):
    """Extract and save per-round metrics from active learning results."""
    safe_name = name.replace("-", "_").replace(" ", "_").lower()
    per_seed_rounds = []
    for sr in agg.get("per_seed", []):
        if not sr.get("success"):
            continue
        al = sr.get("active_learning", {})
        round_metrics = al.get("round_metrics", [])
        per_seed_rounds.append({
            "seed": sr.get("seed"),
            "round_metrics": round_metrics,
        })
    if per_seed_rounds:
        with open(save_dir / f"rounds_{safe_name}.json", "w") as f:
            json.dump(_make_json_safe(per_seed_rounds), f, indent=2)


if __name__ == "__main__":
    main()
