"""Multi-seed experiment runner with aggregation.

Provides a generic `run_multi_seed()` function that runs any experiment
across multiple random seeds and returns aggregated metrics with mean/std.
"""

from typing import Any, Callable, Dict, List
import numpy as np
import time


def run_multi_seed(
    train_fn: Callable[[Any, int], Dict],
    config: Any,
    seeds: List[int],
    experiment_name: str = "",
    verbose: bool = True,
) -> Dict:
    """Run an experiment across multiple random seeds.

    Parameters
    ----------
    train_fn : callable
        Function that takes (config, seed) and returns a dict with at least
        keys: "eval_result", "train_result". The eval_result should contain
        a "test" dict with "auroc", "auprc", "precision@50", "precision@100".
    config : dataclass
        Experiment config (will be cloned per seed via config.__class__(**vars(config))).
    seeds : list of int
        List of random seeds to use.
    experiment_name : str
        Label for this experiment (used in print output).
    verbose : bool
        Print per-seed results.

    Returns
    -------
    dict with keys:
        per_seed : list of per-seed result dicts (with seed added)
        mean_metrics : dict of mean values across seeds
        std_metrics : dict of std values across seeds
        seeds : list of seeds tested
        experiment_name : str
    """
    from copy import deepcopy
    import dataclasses

    per_seed_results = []
    t0 = time.time()

    for i, seed in enumerate(seeds):
        # Clone config with new seed
        if dataclasses.is_dataclass(config):
            config_seed = config.__class__(**{
                k: v for k, v in config.__dict__.items()
            })
        else:
            config_seed = deepcopy(config)
        config_seed.random_seed = seed

        if verbose:
            print(f"\n{'─' * 48}")
            print(f"[{experiment_name}] Seed {seed} ({i + 1}/{len(seeds)})")
            print(f"{'─' * 48}")

        try:
            result = train_fn(config_seed, seed)
            result["seed"] = seed
            result["success"] = True
            per_seed_results.append(result)

            if verbose and "eval_result" in result:
                ev = result["eval_result"]
                test = ev.get("test", {})
                print(f"  Test AUROC: {test.get('auroc', float('nan')):.4f}  "
                      f"AUPRC: {test.get('auprc', float('nan')):.4f}  "
                      f"P@50: {test.get('precision@50', 0):.3f}")
        except Exception as e:
            print(f"  ERROR (seed={seed}): {e}")
            per_seed_results.append({"seed": seed, "success": False, "error": str(e)})

    elapsed = time.time() - t0

    # Aggregate numeric metrics
    success_results = [r for r in per_seed_results if r.get("success")]
    if not success_results:
        print(f"\n  ALL SEEDS FAILED for {experiment_name}")
        return {
            "per_seed": per_seed_results,
            "mean_metrics": {},
            "std_metrics": {},
            "seeds": seeds,
            "experiment_name": experiment_name,
            "elapsed_seconds": elapsed,
        }

    # Collect metrics from test results
    metric_keys = [
        "test_auroc", "test_auprc",
        "test_precision@50", "test_precision@100",
        "test_recall@50", "test_recall@100",
        "val_pr", "val_roc", "best_epoch",
    ]
    mean_metrics = {}
    std_metrics = {}

    for key in metric_keys:
        values = []
        for r in success_results:
            ev = r.get("eval_result", {})
            tr = r.get("train_result", {})
            test = ev.get("test", {})

            if key.startswith("test_"):
                test_key = key[5:]  # remove "test_" prefix
                v = test.get(test_key, None)
            elif key == "val_pr":
                v = tr.get("best_val_pr", None)
            elif key == "val_roc":
                v = tr.get("best_val_roc", None)
            elif key == "best_epoch":
                v = tr.get("best_epoch", None)
            else:
                v = None

            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                values.append(float(v))

        if values:
            mean_metrics[key] = float(np.mean(values))
            std_metrics[key] = float(np.std(values))

    n_success = len(success_results)
    if verbose:
        print(f"\n{'═' * 48}")
        print(f"[{experiment_name}] Summary ({n_success}/{len(seeds)} seeds, {elapsed:.0f}s)")
        for key in mean_metrics:
            print(f"  {key}: {mean_metrics[key]:.4f} ± {std_metrics.get(key, 0):.4f}")
        print(f"{'═' * 48}")

    return {
        "per_seed": per_seed_results,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "seeds": seeds,
        "experiment_name": experiment_name,
        "elapsed_seconds": elapsed,
    }
