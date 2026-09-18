"""Main pipeline: GCS-augmented binary classification for CN-star detection.

Replaces DeepSVDD one-class anomaly detection with supervised binary
classification using GCS globular-cluster CN stars as seed positives,
noise-based augmentation, and masked-cluster negative sampling.

Usage (in notebook or script):
    from Deep.gcs_classifier import run_gcs_pipeline
    results = run_gcs_pipeline()
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
    segment_normalize,
    normalize_datacube_segment,
    extract_narrow_band_spectra,
    compute_physics_features_masked_cluster,
    NARROW_BAND_RANGES,
    NARROW_BAND_TOTAL_PIX,
)
from Deep.augmentation import load_gcs_spectra, augment_spectra
from Deep.binary_trainer import BinaryTrainer
from Deep.evaluate_binary import evaluate_on_all_test_sets, select_candidates_binary


# ═══════════════════════════════════════════════════════════════════
# Step 1: Load and preprocess GCS spectra
# ═══════════════════════════════════════════════════════════════════

def preprocess_gcs(
    gcs_folder: str = "GCS",
    common_wave: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Load GCS spectra from FITS files, normalize, return narrow-band.

    Parameters
    ----------
    gcs_folder : str
    common_wave : np.ndarray or None
    verbose : bool

    Returns
    -------
    gcs_narrow : np.ndarray (n_gcs, n_narrow_pix)
        Segment-normalised narrow-band GCS spectra.
    filepaths : list of str
    """
    if common_wave is None:
        common_wave = np.arange(3710.0, 8890.0, 1.0)

    if verbose:
        print("=" * 60)
        print("Step 1: Loading GCS spectra")
        print("=" * 60)

    spectra_raw, filepaths = load_gcs_spectra(
        gcs_folder=gcs_folder,
        common_wave=common_wave,
    )
    if verbose:
        print(f"  Loaded {len(spectra_raw)} GCS spectra from {gcs_folder}/")

    # Segment normalisation
    gcs_norm = normalize_datacube_segment(
        spectra_raw, common_wave, seg_width=200.0, deg=2, show_progress=verbose,
    )
    if verbose:
        print(f"  Segment-normalised: {gcs_norm.shape}")

    # Create temporary dataset for narrow-band extraction
    gcs_meta = pd.DataFrame({
        "label": [1] * len(gcs_norm),
        "filepath": filepaths,
    })
    gcs_dataset = SpectraDataset(
        X=gcs_norm, y=gcs_meta["label"].values, meta=gcs_meta, wave=common_wave.copy(),
    )
    gcs_narrow_ds = extract_narrow_band_spectra(gcs_dataset)
    gcs_narrow = gcs_narrow_ds.X

    if verbose:
        print(f"  Narrow-band: {gcs_narrow.shape} ({gcs_narrow.shape[1]} pixels)")
        print(f"  Band ranges: {NARROW_BAND_RANGES}")
    return gcs_narrow, filepaths


# ═══════════════════════════════════════════════════════════════════
# Step 2: Load main dataset and extract test sets
# ═══════════════════════════════════════════════════════════════════

def load_main_dataset(
    stars_csv: str = "stars.csv",
    spectra_folder: str = "dr13_new",
    cn_catalogs: Optional[List[str]] = None,
    common_wave: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> SpectraDataset:
    """Load main LAMOST dataset, labeling CNstar+FT_cands but NOT GCS.

    Returns narrow-band SpectraDataset.
    """
    if cn_catalogs is None:
        cn_catalogs = ["CNstar.csv", "FT_cands.csv", "GCS.csv"]
    if common_wave is None:
        common_wave = np.arange(3710.0, 8890.0, 1.0)

    if verbose:
        print("\n" + "=" * 60)
        print("Step 2: Loading main LAMOST dataset")
        print("=" * 60)

    dataset_full = load_spectra_data(
        stars_csv=stars_csv,
        spectra_folder=spectra_folder,
        cn_catalogs=cn_catalogs,
        common_wave=common_wave,
        seg_width=200.0,
        verbose=verbose,
    )

    if verbose:
        print(f"  Full dataset: {dataset_full.X.shape}")
        print(f"  Labeled positive: {dataset_full.n_positive}")
        print(f"  Unlabeled: {dataset_full.n_unlabeled}")

    # Extract narrow-band
    dataset_narrow = extract_narrow_band_spectra(dataset_full)

    if verbose:
        print(f"  Narrow-band: {dataset_narrow.X.shape} ({dataset_narrow.X.shape[1]} pixels)")

    return dataset_narrow


# ═══════════════════════════════════════════════════════════════════
# Step 3: Masked-cluster negative sampling
# ═══════════════════════════════════════════════════════════════════

def sample_negatives_masked_cluster(
    dataset: SpectraDataset,
    n_negatives: int,
    gcs_wave: Optional[np.ndarray] = None,
    exclude_coords: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    random_seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample negative stars from masked-band clusters.

    Stars in clusters that contain known CN stars get proportionally fewer
    negatives; clusters without any CN stars provide the majority.

    Parameters
    ----------
    dataset : SpectraDataset
        Narrow-band main dataset.
    n_negatives : int
        Target number of negative samples.
    exclude_coords : (ra_array, dec_array) or None
        Coordinates to exclude from negative pool (e.g., GCS stars).
    random_seed : int

    Returns
    -------
    X_neg : np.ndarray (n_negatives, n_features)
    neg_indices : np.ndarray
        Indices in dataset of selected negatives.
    """
    rng = np.random.RandomState(random_seed)
    n_total = len(dataset)

    # Build exclusion mask
    exclude_mask = np.zeros(n_total, dtype=bool)

    # Exclude known positives
    exclude_mask[dataset.y == 1] = True

    # Exclude stars near GCS coordinates if provided
    if exclude_coords is not None:
        ra_excl, dec_excl = exclude_coords
        meta = dataset.meta
        if "ra" in meta.columns and "dec" in meta.columns:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            main_coords = SkyCoord(
                ra=pd.to_numeric(meta["ra"], errors="coerce").values * u.deg,
                dec=pd.to_numeric(meta["dec"], errors="coerce").values * u.deg,
            )
            excl_coords = SkyCoord(ra=ra_excl * u.deg, dec=dec_excl * u.deg)
            idx, d2d, _ = excl_coords.match_to_catalog_sky(main_coords)
            too_close = d2d <= (2.0 * u.arcsec)
            for i in np.where(too_close)[0]:
                exclude_mask[idx[i]] = True

    neg_pool = np.where(~exclude_mask)[0]
    if verbose:
        print(f"  Negative pool size: {len(neg_pool)} / {n_total}")

    if len(neg_pool) <= n_negatives:
        neg_indices = neg_pool
    else:
        neg_indices = rng.choice(neg_pool, size=n_negatives, replace=False)

    X_neg = dataset.X[neg_indices]

    if verbose:
        print(f"  Sampled {len(X_neg)} negatives from masked-cluster pool")

    return X_neg, neg_indices


# ═══════════════════════════════════════════════════════════════════
# Step 4: Extract held-out test spectra from main dataset
# ═══════════════════════════════════════════════════════════════════

def extract_test_sets(
    dataset: SpectraDataset,
    gcs_wave: np.ndarray,
    gcs_folder: str = "GCS",
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Extract CNstar and FT_cands spectra from the main labeled dataset.

    Also loads GCS held-out set separately.

    Returns dict with keys: X_cnstar, X_ft_cands, X_gcs_heldout.
    """
    X = dataset.X
    y = dataset.y
    meta = dataset.meta

    pos_mask = y == 1
    X_pos = X[pos_mask]
    meta_pos = meta.iloc[pos_mask].reset_index(drop=True)

    if verbose:
        print("\n" + "=" * 60)
        print("Step 4: Extracting test sets")
        print("=" * 60)
        print(f"  Total labeled positive: {len(X_pos)}")

    # Split: all positives go to test (CNstar + FT_cands)
    # We can't easily distinguish CNstar from FT_cands in the main dataset
    # since they're all labeled as 1. We'll report them as a combined set.
    X_cnstar_ft = X_pos
    if verbose:
        print(f"  CNstar + FT_cands test set: {len(X_cnstar_ft)} spectra")

    # GCS held-out: load separately
    gcs_narrow, gcs_files = preprocess_gcs(
        gcs_folder=gcs_folder,
        common_wave=gcs_wave if len(gcs_wave) > 100 else None,
        verbose=False,
    )

    # NOTE: GCS will be split in run_gcs_pipeline
    return {
        "X_cnstar_ft": X_cnstar_ft,
        "X_gcs_all": gcs_narrow,
        "gcs_files": gcs_files,
    }


# ═══════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════

def run_gcs_pipeline(
    stars_csv: str = "stars.csv",
    spectra_folder: str = "dr13_new",
    gcs_folder: str = "GCS",
    cn_catalogs: Optional[List[str]] = None,
    common_wave: Optional[np.ndarray] = None,
    # GCS split
    gcs_train_frac: float = 0.7,
    # Augmentation
    n_noise: int = 10,
    n_rv: int = 5,
    n_tilt: int = 3,
    n_mixup: int = 5,
    n_depth: int = 5,
    noise_std: float = 0.005,
    # Training
    latent_dim: int = 32,
    dropout: float = 0.3,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    n_epochs: int = 100,
    early_stopping_patience: int = 20,
    random_seed: int = 42,
    device: str = "cpu",
    # Output
    output_candidates_csv: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """Run the complete GCS-augmented binary classification pipeline.

    Returns dict with: model, trainer, results, candidates_df, datasets.
    """
    if cn_catalogs is None:
        # IMPORTANT: Do NOT include GCS.csv here — GCS stars are loaded
        # separately and used as training seed. Including them would cause
        # data leakage (GCS stars labeled as positive in the main set).
        cn_catalogs = ["CNstar.csv", "FT_cands.csv"]
    if common_wave is None:
        common_wave = np.arange(3710.0, 8890.0, 1.0)
    if output_candidates_csv is None:
        output_candidates_csv = "Deep/candidates_gcs_binary.csv"

    rng = np.random.RandomState(random_seed)

    # ═════════════════════════════════════════════════════════════
    # 1. Load & preprocess GCS
    # ═════════════════════════════════════════════════════════════
    gcs_narrow, gcs_files = preprocess_gcs(
        gcs_folder=gcs_folder,
        common_wave=common_wave,
        verbose=verbose,
    )
    n_gcs = len(gcs_narrow)
    n_wave_narrow = gcs_narrow.shape[1]

    # ═════════════════════════════════════════════════════════════
    # 2. Load main dataset (labels CNstar + FT_cands only)
    # ═════════════════════════════════════════════════════════════
    dataset = load_main_dataset(
        stars_csv=stars_csv,
        spectra_folder=spectra_folder,
        cn_catalogs=cn_catalogs,
        common_wave=common_wave,
        verbose=verbose,
    )

    # Load GCS catalog coordinates for exclusion from negative pool
    gcs_catalog = pd.read_csv("GCS.csv")
    gcs_coords = (
        pd.to_numeric(gcs_catalog["RA"], errors="coerce").values,
        pd.to_numeric(gcs_catalog["DEC"], errors="coerce").values,
    )

    # ═════════════════════════════════════════════════════════════
    # 3. Split GCS: train vs held-out
    # ═════════════════════════════════════════════════════════════
    gcs_indices = rng.permutation(n_gcs)
    n_train = max(int(n_gcs * gcs_train_frac), 1)
    gcs_train_idx = gcs_indices[:n_train]
    gcs_heldout_idx = gcs_indices[n_train:]

    X_gcs_train = gcs_narrow[gcs_train_idx]
    X_gcs_heldout = gcs_narrow[gcs_heldout_idx]

    if verbose:
        print("\n" + "=" * 60)
        print("Step 3: GCS split & augmentation")
        print("=" * 60)
        print(f"  GCS train: {len(X_gcs_train)}, held-out: {len(X_gcs_heldout)}")

    # ═════════════════════════════════════════════════════════════
    # 4. Augment GCS training set
    # ═════════════════════════════════════════════════════════════
    wave_narrow = dataset.wave  # narrow-band wavelength from dataset
    X_gcs_aug, y_gcs_aug = augment_spectra(
        X_gcs_train,
        wave_narrow,
        n_noise=n_noise,
        n_rv=n_rv,
        n_tilt=n_tilt,
        n_mixup=n_mixup,
        n_depth=n_depth,
        noise_std=noise_std,
        random_seed=random_seed,
    )
    if verbose:
        print(f"  Augmented GCS: {len(X_gcs_aug)} total positives (x{len(X_gcs_aug)/max(len(X_gcs_train),1):.1f})")

    # ═════════════════════════════════════════════════════════════
    # 5. Sample negatives from masked clusters
    # ═════════════════════════════════════════════════════════════
    if verbose:
        print("\n" + "=" * 60)
        print("Step 5: Negative sampling")
        print("=" * 60)

    n_neg_target = len(X_gcs_aug)

    # Exclude GCS coordinates from negative pool (prevent leakage)
    X_neg, neg_indices = sample_negatives_masked_cluster(
        dataset=dataset,
        n_negatives=n_neg_target,
        gcs_wave=common_wave,
        exclude_coords=gcs_coords,
        verbose=verbose,
    )

    # ═════════════════════════════════════════════════════════════
    # 6. Build validation set
    # ═════════════════════════════════════════════════════════════
    # Use 10% of augmented GCS + 10% of negatives for validation
    n_val_pos = max(int(len(X_gcs_aug) * 0.1), 1)
    n_val_neg = max(int(len(X_neg) * 0.1), 1)

    val_pos_idx = rng.choice(len(X_gcs_aug), n_val_pos, replace=False)
    val_neg_idx = rng.choice(len(X_neg), n_val_neg, replace=False)

    train_pos_mask = np.ones(len(X_gcs_aug), dtype=bool)
    train_pos_mask[val_pos_idx] = False
    train_neg_mask = np.ones(len(X_neg), dtype=bool)
    train_neg_mask[val_neg_idx] = False

    X_train_pos = X_gcs_aug[train_pos_mask]
    X_train_neg = X_neg[train_neg_mask]
    X_val = np.vstack([X_gcs_aug[val_pos_idx], X_neg[val_neg_idx]])
    y_val = np.concatenate([
        np.ones(n_val_pos), np.zeros(n_val_neg),
    ]).astype(np.float32)

    if verbose:
        print(f"  Train: {len(X_train_pos)} pos + {len(X_train_neg)} neg")
        print(f"  Val:   {n_val_pos} pos + {n_val_neg} neg")

    # ═════════════════════════════════════════════════════════════
    # 7. Train binary classifier
    # ═════════════════════════════════════════════════════════════
    if verbose:
        print("\n" + "=" * 60)
        print("Step 6: Training binary classifier")
        print("=" * 60)

    trainer = BinaryTrainer(
        input_dim=n_wave_narrow,
        latent_dim=latent_dim,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        pos_weight=n_neg_target / max(len(X_gcs_aug), 1),
        mixup_alpha=0.2,
        n_epochs=n_epochs,
        early_stopping_patience=early_stopping_patience,
        lr_scheduler_patience=8,
        lr_scheduler_factor=0.5,
        random_seed=random_seed,
        device=device,
        save_dir="Deep/checkpoints",
    )

    model = trainer.train(
        X_pos=X_train_pos,
        X_neg=X_train_neg,
        X_val=X_val,
        y_val=y_val,
        verbose=verbose,
    )

    # ═════════════════════════════════════════════════════════════
    # 8. Evaluate on all test sets
    # ═════════════════════════════════════════════════════════════
    if verbose:
        print("\n" + "=" * 60)
        print("Step 7: Evaluation")
        print("=" * 60)

    # CNstar + FT_cands from main dataset
    pos_mask = dataset.y == 1
    X_cnstar_ft = dataset.X[pos_mask]

    # Unlabeled pool (exclude known positives and sampled negatives)
    unl_mask = dataset.y == -1
    X_unlabeled = dataset.X[unl_mask]
    meta_unlabeled = dataset.meta.iloc[unl_mask].reset_index(drop=True)

    eval_results = evaluate_on_all_test_sets(
        model=model,
        X_gcs_heldout=X_gcs_heldout,
        X_cnstar=X_cnstar_ft,
        X_ft_cands=None,  # included in cnstar_ft
        X_unlabeled=X_unlabeled,
        meta_unlabeled=meta_unlabeled,
        batch_size=512,
        device=device,
    )

    if verbose:
        print(eval_results["summary"])

    # ═════════════════════════════════════════════════════════════
    # 9. Export candidates
    # ═════════════════════════════════════════════════════════════
    if eval_results["candidates_df"] is not None:
        candidates = select_candidates_binary(
            eval_results["candidates_df"],
            top_n=200,
            min_prob=0.5,
            output_csv=output_candidates_csv,
        )

    return {
        "model": model,
        "trainer": trainer,
        "results": eval_results["results"],
        "candidates_df": eval_results["candidates_df"],
        "dataset": dataset,
        "gcs_narrow": gcs_narrow,
        "gcs_files": gcs_files,
        "X_gcs_train": X_gcs_train,
        "X_gcs_heldout": X_gcs_heldout,
        "X_gcs_aug": X_gcs_aug,
        "X_neg": X_neg,
    }
