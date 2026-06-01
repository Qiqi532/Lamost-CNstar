"""Structured result save/load for experiments.

Follows the same pattern as EndToEndPU/train.py run_experiment().
"""

import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch


def save_experiment_results(
    result: Dict,
    save_dir: Path,
    experiment_name: str,
    save_model: bool = True,
):
    """Save experiment results (model, metrics, history, candidates).

    Parameters
    ----------
    result : dict
        Must contain: train_result, eval_result.
        May contain: model, config, candidates_df.
    save_dir : Path
        Directory to save results.
    experiment_name : str
        Unique name for this experiment run.
    save_model : bool
        Whether to save model checkpoint.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    train_result = result.get("train_result", {})
    eval_result = result.get("eval_result", {})
    config = result.get("config", None)

    # Save model
    if save_model and "model" in result:
        model_path = save_dir / f"model_{experiment_name}.pt"
        save_dict = {
            "experiment_name": experiment_name,
            "best_epoch": train_result.get("best_epoch", 0),
            "best_val_pr": float(train_result.get("best_val_pr", 0)),
            "best_val_roc": float(train_result.get("best_val_roc", 0)),
        }
        # Store config if available
        if config is not None:
            if hasattr(config, '__dict__'):
                save_dict["config"] = {
                    k: str(v) if isinstance(v, (list, type(None), Path)) else v
                    for k, v in config.__dict__.items()
                }
        try:
            save_dict["model_state_dict"] = result["model"].state_dict()
            torch.save(save_dict, model_path)
        except Exception as e:
            print(f"  Warning: Could not save model: {e}")

    # Save metrics
    metrics = {
        "experiment_name": experiment_name,
        "best_epoch": train_result.get("best_epoch", 0),
        "best_val_pr": float(train_result.get("best_val_pr", 0)),
        "best_val_roc": float(train_result.get("best_val_roc", 0)),
        "elapsed_seconds": train_result.get("elapsed_seconds", 0),
    }

    if "test" in eval_result:
        t = eval_result["test"]
        for k in ["auroc", "auprc"]:
            v = t.get(k, None)
            metrics[f"test_{k}"] = float(v) if v is not None and not np.isnan(v) else None
        for k in [50, 100, 200]:
            metrics[f"test_precision@{k}"] = float(t.get(f"precision@{k}", 0))
            metrics[f"test_recall@{k}"] = float(t.get(f"recall@{k}", 0))

    if "seed" in result:
        metrics["seed"] = result["seed"]

    with open(save_dir / f"metrics_{experiment_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save training history
    history = train_result.get("history", {})
    if history:
        history_clean = {
            k: [float(x) for x in v]
            for k, v in history.items()
        }
        with open(save_dir / f"history_{experiment_name}.json", "w") as f:
            json.dump(history_clean, f, indent=2)

    # Export candidates
    candidates_df = eval_result.get("candidates_df", None)
    if candidates_df is not None:
        top_n = min(200, len(candidates_df))
        candidates_df.head(top_n).to_csv(
            save_dir / f"candidates_{experiment_name}.csv", index=False
        )


def load_experiment_results(
    load_dir: Path,
    experiment_name: str,
) -> Dict:
    """Load saved experiment results.

    Returns dict with keys: metrics, history, (optionally model_state_dict).
    """
    result = {}

    metrics_path = load_dir / f"metrics_{experiment_name}.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            result["metrics"] = json.load(f)

    history_path = load_dir / f"history_{experiment_name}.json"
    if history_path.exists():
        with open(history_path) as f:
            result["history"] = json.load(f)

    model_path = load_dir / f"model_{experiment_name}.pt"
    if model_path.exists():
        result["model_path"] = str(model_path)

    return result


def build_comparison_table(
    all_results: Dict[str, Dict],
    metrics: list = None,
) -> "pd.DataFrame":
    """Build a comparison DataFrame from all experiment results.

    Parameters
    ----------
    all_results : dict
        Mapping of experiment_name -> aggregated result dict (from run_multi_seed).
    metrics : list of str
        Metric keys to include (default: standard set).

    Returns
    -------
    pd.DataFrame with columns: Experiment, AUROC, AUPRC, P@50, P@100, etc.
    """
    import pandas as pd

    if metrics is None:
        metrics = [
            "test_auroc", "test_auprc",
            "test_precision@50", "test_precision@100",
            "test_recall@50", "test_recall@100",
            "val_pr", "val_roc",
        ]

    rows = []
    for name, result in all_results.items():
        row = {"Experiment": name}
        mean = result.get("mean_metrics", {})
        std = result.get("std_metrics", {})
        for m in metrics:
            if m in mean:
                if m in std and std[m] > 0:
                    row[m] = f"{mean[m]:.4f} ± {std[m]:.4f}"
                else:
                    row[m] = f"{mean[m]:.4f}"
            else:
                row[m] = "—"
        rows.append(row)

    return pd.DataFrame(rows)
