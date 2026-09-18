"""XGBoost PU-Bagging on Autoencoder bottleneck features.

Runs PU-Bagging on AE-64d features and compares with 700px raw spectra baseline.

Usage:
    python SpectraAE/pu_bagging_ae.py                      # run comparison
    python SpectraAE/pu_bagging_ae.py --no-spectra          # skip 700px baseline (faster)

Or import:
    from SpectraAE.pu_bagging_ae import run_pu_bagging_ae, build_comparison
"""

import sys, warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ML.pu_bagging import load_pu_data, run_pu_bagging, build_comparison_df


def run_pu_bagging_ae(
    ae_features: np.ndarray,
    data: Dict,
    T: int = 500,
    verbose: bool = True,
) -> Dict:
    """Run XGBoost PU-Bagging on AE-extracted features.

    Parameters
    ----------
    ae_features : np.ndarray, shape (N, 64)
        Bottleneck features from trained autoencoder.
    data : dict
        From ML.pu_bagging.load_pu_data(); provides y_all, tr_idx, test_idx,
        cluster_ids, df_model.
    T : int
        Number of bagging iterations.

    Returns
    -------
    dict with same keys as run_pu_bagging() output.
    """
    # Standardize AE features
    X_ae = ae_features.copy().astype(np.float32)
    ss = StandardScaler()
    X_ae = ss.fit_transform(X_ae).astype(np.float32)

    return run_pu_bagging(
        X_ae, data["y_all"], data["tr_idx"], data["test_idx"],
        data["cluster_ids"], data["df_model"],
        T=T, name="AE-64d", verbose=verbose,
    )


def build_comparison(
    res_ae: Dict,
    res_spec: Dict,
    res_cn9: Optional[Dict] = None,
) -> pd.DataFrame:
    """Build comparison table: AE-64d vs Raw Spectra 700-D vs CN 9-D (optional).

    Uses ML.pu_bagging.build_comparison_df() for consistent formatting.
    """
    rows = [res_ae, res_spec]
    if res_cn9 is not None:
        rows.append(res_cn9)
    return build_comparison_df(*rows) if len(rows) == 2 else build_comparison_df(
        *rows[:2]
    ).append(pd.DataFrame([res_cn9]))


# ══════════════════════════════════════════════════════════════════
# Standalone runner
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PU-Bagging on AE features")
    parser.add_argument("--features", type=str,
                        default="SpectraAE/_cache/ae_features_64d.npy")
    parser.add_argument("--T", type=int, default=500, help="Bagging iterations")
    parser.add_argument("--no-spectra", action="store_true",
                        help="Skip 700px baseline (use cached if available)")
    parser.add_argument("--output", type=str,
                        default="SpectraAE/results/pu_bagging_ae_comparison.csv")
    args = parser.parse_args()

    # Load AE features
    feats_path = Path(args.features)
    if not feats_path.exists():
        raise FileNotFoundError(
            f"Features not found: {feats_path}. Run extract_features.py first."
        )
    ae_features = np.load(feats_path).astype(np.float32)
    print(f"Loaded AE features: {ae_features.shape}")

    # Load shared data (labels, splits, cluster info)
    data = load_pu_data()
    print(f"Loaded: {len(data['y_all'])} stars, {int(data['y_all'].sum())} positives")

    # ── Run PU-Bagging on AE features ─────────────────────────
    print(f"\n{'='*60}")
    print(f"PU-Bagging on AE-64d (T={args.T})")
    print(f"{'='*60}")
    res_ae = run_pu_bagging_ae(ae_features, data, T=args.T)

    # ── Run PU-Bagging on 700px spectra (baseline) ────────────
    print(f"\n{'='*60}")
    print(f"PU-Bagging on Raw Spectra 700-D (T={args.T})")
    print(f"{'='*60}")
    res_spec = run_pu_bagging(
        data["X_spec"], data["y_all"], data["tr_idx"], data["test_idx"],
        data["cluster_ids"], data["df_model"],
        T=args.T, name="Raw Spectra 700-D",
    )

    # ── Comparison ────────────────────────────────────────────
    comp_df = build_comparison_df(res_ae, res_spec)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(f"\n{'='*60}")
    print("AE-64d vs Raw Spectra 700-D — COMPARISON")
    print(f"{'='*60}")
    print(comp_df.to_string(index=False))

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(output_path, index=False)
    print(f"\nSaved comparison to {output_path}")

    # Save per-star AE probabilities
    out = data["df_model"][["teff", "logg", "feh", "label"]].copy()
    out["ae64_prob"] = res_ae["probs_all"]
    out["ae64_std"] = res_ae["probs_all_std"]
    out["spec_prob"] = res_spec["probs_all"]
    out["spec_std"] = res_spec["probs_all_std"]
    probs_path = Path("SpectraAE/results/ae_pu_probs.csv")
    out.to_csv(probs_path, index=False)
    print(f"Saved per-star probabilities to {probs_path}")
    print("Done.")
