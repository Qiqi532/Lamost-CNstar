"""Main pipeline: FT_cands-augmented binary classification for CN-star detection.

Key changes vs Deep/gcs_classifier.py:
- FT_cands (42 field CN-star candidates) as augmentation seed instead of GCS
- Wider wavelength range: 3800-5000Å (~1200 pix) instead of narrow-band
- Ensemble training (5 models) for robust probability estimates
- GCS (29) + CNstar (106) held out entirely for validation
- Supports both 'conv1d' (spectral) and 'mlp' (physics features) backbones
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import Deep.spectra_io as sio
from Deep.data import (
    SpectraDataset,
    load_spectra_data,
    normalize_datacube_segment,
    compute_physics_features_masked_cluster,
    NARROW_BAND_RANGES,
)
from Deep.augmentation import load_gcs_spectra

from .augmentation import augment_spectra
from .trainer import BinaryTrainer, EnsembleTrainer
from .evaluate import evaluate_ensemble, export_candidates


# ═══════════════════════════════════════════════════════════════════
# Wavelength slicing
# ═══════════════════════════════════════════════════════════════════

WAVE_START = 3800.0
WAVE_END = 5000.0


def slice_wave_range(
    spectra: np.ndarray,
    common_wave: np.ndarray,
    w_start: float = WAVE_START,
    w_end: float = WAVE_END,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice spectra to a target wavelength range."""
    mask = (common_wave >= w_start) & (common_wave <= w_end)
    return spectra[:, mask], common_wave[mask].copy()


# ═══════════════════════════════════════════════════════════════════
# FT_cands seed loading
# ═══════════════════════════════════════════════════════════════════

def load_ft_seed_spectra(
    dataset: SpectraDataset,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Extract FT_cands spectra from the main dataset (label=1).

    Returns (spectra, metadata) for FT_cands stars.
    """
    pos_mask = dataset.y == 1
    X_pos = dataset.X[pos_mask]
    meta_pos = dataset.meta.iloc[pos_mask].reset_index(drop=True)
    return X_pos, meta_pos


# ═══════════════════════════════════════════════════════════════════
# Validation set extraction via coordinate matching
# ═══════════════════════════════════════════════════════════════════

def match_catalog_to_dataset(
    dataset_meta: pd.DataFrame,
    catalog_csv: str,
    tolerance_arcsec: float = 1.0,
) -> np.ndarray:
    """Find dataset indices that match a catalog by sky coordinates.

    Returns boolean mask aligned with dataset rows.
    """
    from Deep.spectra_io import label_stars_by_cn_catalog

    labeled, _ = label_stars_by_cn_catalog(
        stars_df=dataset_meta.copy(),
        cn_catalogs=[catalog_csv],
        tolerance_arcsec=tolerance_arcsec,
        positive_label=1,
        unlabeled_value=-1,
    )
    return labeled["label"].values == 1


# ═══════════════════════════════════════════════════════════════════
# MLP feature preparation helper
# ═══════════════════════════════════════════════════════════════════

def _prepare_mlp_features(
    dataset, X_gcs_sliced, X_ft_aug, neg_pool, X_val, X_cnstar,
    val_pos_idx, val_neg_idx, train_pos_mask, wave_slice,
    exclude_mask, cnstar_mask,
    n_noise, n_rv, n_tilt, n_mixup, n_depth, noise_std, random_seed, verbose,
):
    """Swap all arrays from spectral space to physics-feature space for MLP encoder."""
    if verbose:
        print("  Computing physics features...")
    phys_result = compute_physics_features_masked_cluster(
        dataset, n_clusters=45, k_neighbors=180, n_pca=3, random_seed=random_seed,
    )
    dataset_phys = phys_result["dataset"]

    # Re-augment FT_cands in physics space
    X_ft_phys = dataset_phys.X[dataset.y == 1]
    X_ft_aug_phys, _ = augment_spectra(
        X_ft_phys, dataset_phys.wave,
        n_noise=n_noise, n_rv=n_rv, n_tilt=n_tilt,
        n_mixup=n_mixup, n_depth=n_depth,
        noise_std=noise_std * 0.5, random_seed=random_seed,
    )
    X_train_pos = X_ft_aug_phys[train_pos_mask[:len(X_ft_aug_phys)]]

    neg_pool_phys = dataset_phys.X[~exclude_mask[:len(dataset_phys)]]
    X_val_phys = np.vstack([
        X_ft_aug_phys[val_pos_idx[:len(X_ft_aug_phys)]],
        neg_pool_phys[val_neg_idx[:len(neg_pool_phys)]],
    ])
    X_cnstar_phys = dataset_phys.X[cnstar_mask[:len(dataset_phys)]]

    # GCS physics features
    gcs_cat = pd.read_csv("GCS.csv")
    n_gcs = len(X_gcs_sliced)
    gcs_meta = pd.DataFrame({
        "label": [1] * n_gcs,
        "teff": gcs_cat["Teff"].values[:n_gcs],
        "logg": gcs_cat["logg"].values[:n_gcs],
        "feh": gcs_cat["[Fe/H]"].values[:n_gcs],
    })
    gcs_ds = SpectraDataset(X=X_gcs_sliced, y=np.ones(n_gcs), meta=gcs_meta, wave=wave_slice)
    from Deep.data import compute_physics_features
    gcs_phys = compute_physics_features(gcs_ds, n_pca=3, random_seed=random_seed)

    return X_train_pos, neg_pool_phys, X_val_phys, X_cnstar_phys, gcs_phys.X, dataset_phys


# ═══════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════

def run_pipeline(
    # Data paths
    stars_csv: str = "stars.csv",
    spectra_folder: str = "dr13_new",
    gcs_folder: str = "GCS",
    # Wavelength
    w_start: float = WAVE_START,
    w_end: float = WAVE_END,
    # Augmentation
    n_noise: int = 12,
    n_rv: int = 6,
    n_tilt: int = 4,
    n_mixup: int = 8,
    n_depth: int = 6,
    noise_std: float = 0.005,
    # Model
    encoder_type: str = "conv1d",  # 'conv1d' or 'mlp'
    latent_dim: int = 64,
    dropout: float = 0.35,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    n_epochs: int = 80,
    # Ensemble
    n_ensemble: int = 5,
    # Output
    output_candidates_csv: Optional[str] = None,
    verbose: bool = True,
    random_seed: int = 42,
    device: str = "cpu",
) -> Dict:
    """Run the FT_cands-augmented binary classification pipeline.

    Returns dict with: ensemble, results, candidates_df, dataset, wave_slice.
    """
    common_wave_full = np.arange(3710.0, 8890.0, 1.0)
    if output_candidates_csv is None:
        output_candidates_csv = f"BinaryClassifier/candidates_ft_{encoder_type}.csv"

    rng = np.random.RandomState(random_seed)

    # ═══════════════════════════════════════════════════════════
    # 1. Load main dataset — label only FT_cands
    # ═══════════════════════════════════════════════════════════
    if verbose:
        print("=" * 60)
        print("Step 1: Loading main LAMOST dataset (FT_cands labeled)")
        print("=" * 60)

    dataset_full = load_spectra_data(
        stars_csv=stars_csv,
        spectra_folder=spectra_folder,
        cn_catalogs=["FT_cands.csv"],  # Only FT_cands as training seed
        common_wave=common_wave_full,
        seg_width=200.0,
        verbose=verbose,
    )

    # Slice to target wavelength range
    X_sliced, wave_slice = slice_wave_range(
        dataset_full.X, common_wave_full, w_start, w_end,
    )
    dataset = SpectraDataset(
        X=X_sliced, y=dataset_full.y.copy(),
        meta=dataset_full.meta.copy(), wave=wave_slice,
    )

    if verbose:
        print(f"\n  Full dataset: {dataset_full.X.shape} → sliced: {X_sliced.shape}")
        print(f"  Wavelength: {wave_slice[0]:.0f}-{wave_slice[-1]:.0f} A ({X_sliced.shape[1]} pixels)")
        print(f"  FT_cands labeled: {dataset.n_positive}")
        print(f"  Unlabeled: {dataset.n_unlabeled}")

    # ═══════════════════════════════════════════════════════════
    # 2. Extract FT_cands seed spectra
    # ═══════════════════════════════════════════════════════════
    X_ft, meta_ft = load_ft_seed_spectra(dataset)
    n_ft = len(X_ft)

    if verbose:
        print(f"\n  FT_cands seed: {n_ft} spectra")

    # ═══════════════════════════════════════════════════════════
    # 3. Augment FT_cands
    # ═══════════════════════════════════════════════════════════
    if verbose:
        print("\n" + "=" * 60)
        print("Step 2: Augmenting FT_cands spectra")
        print("=" * 60)

    X_ft_aug, y_ft_aug = augment_spectra(
        X_ft, wave_slice,
        n_noise=n_noise, n_rv=n_rv, n_tilt=n_tilt,
        n_mixup=n_mixup, n_depth=n_depth,
        noise_std=noise_std,
        random_seed=random_seed,
    )

    target_pos = n_ft * (n_noise + n_rv + n_tilt + n_mixup + n_depth + 1)
    if verbose:
        print(f"  Augmented: {len(X_ft_aug)} positives (target ~{target_pos})")
        print(f"  Factor: x{len(X_ft_aug)/max(n_ft,1):.1f}")

    # ═══════════════════════════════════════════════════════════
    # 4. Build validation sets: GCS + CNstar
    # ═══════════════════════════════════════════════════════════
    if verbose:
        print("\n" + "=" * 60)
        print("Step 3: Building validation sets")
        print("=" * 60)

    # CNstar: match to main dataset
    cnstar_mask = match_catalog_to_dataset(dataset.meta, "CNstar.csv")
    X_cnstar = dataset.X[cnstar_mask]

    # GCS: load separately then slice to same wave range
    gcs_raw, _ = load_gcs_spectra(gcs_folder, common_wave_full)
    gcs_norm = normalize_datacube_segment(gcs_raw, common_wave_full, seg_width=200.0, deg=2, show_progress=False)
    X_gcs_sliced, _ = slice_wave_range(gcs_norm, common_wave_full, w_start, w_end)

    if verbose:
        print(f"  CNstar validation: {len(X_cnstar)} spectra")
        print(f"  GCS validation:    {len(X_gcs_sliced)} spectra")

    # ═══════════════════════════════════════════════════════════
    # 5. Negative sampling
    # ═══════════════════════════════════════════════════════════
    if verbose:
        print("\n" + "=" * 60)
        print("Step 4: Negative sampling")
        print("=" * 60)

    # Exclude all known positives from negative pool
    exclude_mask = np.zeros(len(dataset), dtype=bool)
    exclude_mask[dataset.y == 1] = True  # FT_cands
    exclude_mask[cnstar_mask] = True     # CNstar
    # GCS stars might not be in the main dataset (different programs)

    neg_pool = dataset.X[~exclude_mask]
    if verbose:
        print(f"  Negative pool: {len(neg_pool)} stars "
              f"({len(neg_pool)/len(dataset)*100:.1f}% of dataset)")

    n_neg_target = max(len(X_ft_aug), 500)

    # ═══════════════════════════════════════════════════════════
    # 6. Train-validation split
    # ═══════════════════════════════════════════════════════════
    n_val_pos = max(int(len(X_ft_aug) * 0.1), 5)
    val_pos_idx = rng.choice(len(X_ft_aug), n_val_pos, replace=False)
    train_pos_mask = np.ones(len(X_ft_aug), dtype=bool)
    train_pos_mask[val_pos_idx] = False

    X_train_pos = X_ft_aug[train_pos_mask]
    n_val_neg = n_val_pos
    val_neg_idx = rng.choice(len(neg_pool), n_val_neg, replace=False)
    X_val = np.vstack([X_ft_aug[val_pos_idx], neg_pool[val_neg_idx]])
    y_val = np.concatenate([np.ones(n_val_pos), np.zeros(n_val_neg)]).astype(np.float32)

    if verbose:
        print(f"  Train pos: {len(X_train_pos)} | Val: {n_val_pos} pos + {n_val_neg} neg")

    # ═══════════════════════════════════════════════════════════
    # 7. Train ensemble
    # ═══════════════════════════════════════════════════════════
    if verbose:
        print("\n" + "=" * 60)
        print(f"Step 5: Training ensemble ({n_ensemble} models, encoder={encoder_type})")
        print("=" * 60)

    n_neg_per_model = len(X_train_pos)  # balanced

    trainer_kwargs = dict(
        input_dim=X_sliced.shape[1],
        latent_dim=latent_dim,
        dropout=dropout,
        encoder_type=encoder_type,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        pos_weight=1.0,
        mixup_alpha=0.2,
        n_epochs=n_epochs,
        early_stopping_patience=15,
        device=device,
        save_dir="BinaryClassifier/checkpoints",
    )

    # If MLP mode, swap to physics features
    dataset_phys = None
    if encoder_type == "mlp":
        X_train_pos, neg_pool, X_val, X_cnstar, X_gcs_sliced, dataset_phys = \
            _prepare_mlp_features(
                dataset, X_gcs_sliced, X_ft_aug, neg_pool, X_val, X_cnstar,
                val_pos_idx, val_neg_idx, train_pos_mask, wave_slice,
                exclude_mask, cnstar_mask,
                n_noise, n_rv, n_tilt, n_mixup, n_depth, noise_std, random_seed,
                verbose,
            )
        trainer_kwargs["input_dim"] = X_train_pos.shape[1]

    ensemble = EnsembleTrainer(n_models=n_ensemble, base_seed=random_seed, **trainer_kwargs)
    ensemble.train(
        X_pos=X_train_pos,
        X_neg_pool=neg_pool,
        n_neg_per_model=min(n_neg_per_model, len(neg_pool)),
        X_val=X_val,
        y_val=y_val,
        verbose=verbose,
    )

    # Evaluate
    if verbose:
        print("\n" + "=" * 60)
        print("Step 6: Evaluation")
        print("=" * 60)

    if dataset_phys is not None:
        unl_mask = ~exclude_mask[:len(dataset_phys)]
        X_unl, meta_unl = dataset_phys.X[unl_mask], dataset_phys.meta.iloc[unl_mask].reset_index(drop=True)
    else:
        unl_mask = ~exclude_mask
        X_unl, meta_unl = dataset.X[unl_mask], dataset.meta.iloc[unl_mask].reset_index(drop=True)

    eval_results = evaluate_ensemble(
        ensemble=ensemble, X_gcs=X_gcs_sliced, X_cnstar=X_cnstar,
        X_unlabeled=X_unl, meta_unlabeled=meta_unl, batch_size=512, device=device,
    )

    if verbose:
        print(eval_results["summary"])

    # ═══════════════════════════════════════════════════════════
    # 9. Export candidates
    # ═══════════════════════════════════════════════════════════
    if eval_results["candidates_df"] is not None:
        export_candidates(
            eval_results["candidates_df"],
            top_n=200, min_prob=0.5,
            output_csv=output_candidates_csv,
        )

    return {
        "ensemble": ensemble,
        "results": eval_results["results"],
        "candidates_df": eval_results["candidates_df"],
        "dataset": dataset,
        "wave_slice": wave_slice,
        "X_ft": X_ft,
        "X_ft_aug": X_ft_aug,
        "X_cnstar": X_cnstar,
        "X_gcs": X_gcs_sliced,
    }
