"""End-to-end pipeline: AE pretraining → feature extraction → PU-Bagging → comparison.

Usage:
    python SpectraAE/run_ae_pipeline.py                     # full pipeline
    python SpectraAE/run_ae_pipeline.py --epochs 100         # shorter training
    python SpectraAE/run_ae_pipeline.py --skip-pretrain      # use existing checkpoint
"""

import sys, time, warnings, argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

warnings.filterwarnings("ignore")

from SpectraAE.pretrain import pretrain_autoencoder
from SpectraAE.extract_features import extract_features
from ML.pu_bagging import load_pu_data, run_pu_bagging, build_comparison_df

DEFAULT_DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="SpectraAE: Conv AE → Bottleneck Features → XGBoost PU-Bagging"
    )
    parser.add_argument("--epochs", type=int, default=200, help="AE training epochs")
    parser.add_argument("--latent-dim", type=int, default=64, help="Bottleneck dimension")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pu-T", type=int, default=500, help="PU-Bagging iterations")
    parser.add_argument("--skip-pretrain", action="store_true",
                        help="Use existing checkpoint (skip AE training)")
    parser.add_argument("--skip-spectra", action="store_true",
                        help="Skip 700px baseline PU-Bagging")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    args = parser.parse_args()

    t_total = time.time()

    # ═══════════════════════════════════════════════════════════
    # Step 1: Load data
    # ═══════════════════════════════════════════════════════════
    X = np.load("ML/_cache/X_clean.npy").astype(np.float32)
    print(f"[1/4] Loaded X_clean: {X.shape}")

    # ═══════════════════════════════════════════════════════════
    # Step 2: Pretrain Autoencoder
    # ═══════════════════════════════════════════════════════════
    ckpt_path = Path("SpectraAE/checkpoints/ae_best.pt")

    if args.skip_pretrain:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"[2/4] Skipping pretrain (using {ckpt_path})")

        # Reconstruct model from checkpoint
        import torch
        from SpectraAE.models.autoencoder import ConvAutoencoder
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model = ConvAutoencoder(in_channels=1, latent_dim=args.latent_dim)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        history = {
            "scaler_mean": ckpt["scaler_mean"],
            "scaler_std": ckpt["scaler_std"],
            "best_epoch": ckpt["epoch"],
            "best_val_loss": ckpt["val_loss"],
        }
    else:
        print(f"[2/4] Pretraining AE ({args.epochs} epochs, latent={args.latent_dim}...)")
        model, history = pretrain_autoencoder(
            X,
            latent_dim=args.latent_dim,
            n_epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
        )

    # ═══════════════════════════════════════════════════════════
    # Step 3: Extract bottleneck features
    # ═══════════════════════════════════════════════════════════
    print("[3/4] Extracting bottleneck features...")
    ae_features = extract_features(
        model, X,
        scaler_mean=history["scaler_mean"],
        scaler_std=history["scaler_std"],
        device=args.device,
    )

    cache_dir = Path("SpectraAE/_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "ae_features_64d.npy", ae_features)
    print(f"  Saved: {ae_features.shape}")

    # ═══════════════════════════════════════════════════════════
    # Step 4: PU-Bagging + Comparison
    # ═══════════════════════════════════════════════════════════
    print("[4/4] Running PU-Bagging...")
    data = load_pu_data()

    # Standardize AE features for PU-Bagging
    X_ae = ae_features.copy().astype(np.float32)
    X_ae = StandardScaler().fit_transform(X_ae).astype(np.float32)

    print(f"\n{'='*60}")
    print(f"PU-Bagging: AE-64d (T={args.pu_T})")
    print(f"{'='*60}")
    res_ae = run_pu_bagging(
        X_ae, data["y_all"], data["tr_idx"], data["test_idx"],
        data["cluster_ids"], data["df_model"],
        T=args.pu_T, name="AE-64d",
    )

    res_spec = None
    if not args.skip_spectra:
        print(f"\n{'='*60}")
        print(f"PU-Bagging: Raw Spectra 700-D (T={args.pu_T})")
        print(f"{'='*60}")
        res_spec = run_pu_bagging(
            data["X_spec"], data["y_all"], data["tr_idx"], data["test_idx"],
            data["cluster_ids"], data["df_model"],
            T=args.pu_T, name="Raw Spectra 700-D",
        )

    # ── Print comparison ──────────────────────────────────────
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 260)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print(f"\n{'='*70}")
    print("RESULTS: AE-64d vs Raw Spectra 700-D")
    print(f"{'='*70}")

    if res_spec is not None:
        comp_df = build_comparison_df(res_ae, res_spec)
        print(comp_df.to_string(index=False))

        results_dir = Path("SpectraAE/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        comp_df.to_csv(results_dir / "pu_bagging_ae_comparison.csv", index=False)
        print(f"\nSaved to SpectraAE/results/pu_bagging_ae_comparison.csv")

        # Save per-star probabilities
        out = data["df_model"][["teff", "logg", "feh", "label"]].copy()
        out["ae64_prob"] = res_ae["probs_all"]
        out["ae64_std"] = res_ae["probs_all_std"]
        out["spec_prob"] = res_spec["probs_all"]
        out["spec_std"] = res_spec["probs_all_std"]
        out.to_csv(results_dir / "ae_pu_probs.csv", index=False)
    else:
        print(f"\nAE-64d result:")
        print(f"  PR={res_ae['PR']:.4f}  ROC={res_ae['ROC']:.4f}  "
              f"P@100={res_ae['P@100']:.4f}  Within-r={res_ae['Within-r']:.4f}")

    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("Done.")


if __name__ == "__main__":
    main()
