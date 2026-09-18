"""Data loading utilities for LAMOST spectra.

Uses the preprocessing pipeline from spectra_io.py with segment-based
continuum normalisation, and wraps data into PyTorch-compatible objects.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import Deep.spectra_io as sio


# ═══════════════════════════════════════════════════════════════════
# Segment-based continuum normalisation
# ═══════════════════════════════════════════════════════════════════

def segment_normalize(
    flux: np.ndarray,
    wave: np.ndarray,
    seg_width: float = 200.0,
    deg: int = 2,
) -> np.ndarray:
    """Continuum-normalise a spectrum by fitting polynomials in segments.

    Divides the wavelength range into ~seg_width Å segments, fits a
    low-order polynomial to each, and divides the flux by the fitted
    continuum.  Much less aggressive than Gaussian-kernel convolution:
    narrow absorption features (including CN bands) are preserved.

    Parameters
    ----------
    flux : np.ndarray (n_pix,)
        Interpolated flux (not yet normalised).
    wave : np.ndarray (n_pix,)
        Wavelength grid.
    seg_width : float
        Approximate segment width in Å.
    deg : int
        Polynomial degree for the continuum fit (2 = quadratic).

    Returns
    -------
    np.ndarray (n_pix,)
        Continuum-normalised flux.
    """
    n = len(flux)
    # Divide range into segments of roughly seg_width Å
    wave_range = wave[-1] - wave[0]
    n_segs = max(1, int(np.ceil(wave_range / seg_width)))
    seg_edges = np.linspace(0, n, n_segs + 1, dtype=int)

    norm = np.empty_like(flux)

    for i in range(n_segs):
        start = seg_edges[i]
        end = seg_edges[i + 1] if i < n_segs - 1 else n
        if end <= start:
            continue
        x_seg = wave[start:end]
        y_seg = flux[start:end].astype(float)

        # Fit polynomial continuum
        try:
            coeffs = np.polyfit(x_seg, y_seg, deg)
            continuum = np.polyval(coeffs, x_seg)
        except (np.linalg.LinAlgError, ValueError):
            continuum = np.full_like(y_seg, np.nanmedian(y_seg))

        # Avoid division by zero / tiny values
        continuum = np.where(np.abs(continuum) < 1e-8,
                            np.nanmedian(continuum), continuum)
        with np.errstate(invalid='ignore', divide='ignore'):
            norm[start:end] = y_seg / continuum
        norm[start:end] = np.nan_to_num(norm[start:end], nan=1.0, posinf=1.0, neginf=1.0)

    # Clip extreme outliers (instrumental artifacts)
    norm = np.clip(norm, 0.0, 3.0)

    return norm.astype(np.float32)


def normalize_datacube_segment(
    datacube: np.ndarray,
    wave: np.ndarray,
    seg_width: float = 200.0,
    deg: int = 2,
    show_progress: bool = True,
) -> np.ndarray:
    """Apply segment normalisation to every spectrum in a datacube."""
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(range(len(datacube)), desc="Segment norm")
    else:
        iterator = range(len(datacube))

    out = np.empty_like(datacube, dtype=np.float32)
    for i in iterator:
        out[i] = segment_normalize(datacube[i], wave, seg_width, deg)
    return out


# ═══════════════════════════════════════════════════════════════════
# CN band index computation (for pre-cleaning training data)
# ═══════════════════════════════════════════════════════════════════

CN_BAND_DEFS = {
    "CN3839": {"band": (3830, 3883), "blue": (3894, 3910), "red": (4000, 4020)},
    "CN4142": {"band": (4120, 4216), "blue": (4055, 4080), "red": (4240, 4280)},
    "CH4300": {"band": (4285, 4315), "blue": (4240, 4280), "red": (4390, 4460)},
}

# Narrow-band windows (band cores only — spectra are continuum-normalised)
NARROW_BAND_RANGES = [
    (3830, 3883),   # CN3839  (~53 pix)
    (4120, 4216),   # CN4142  (~96 pix)
    (4285, 4315),   # CH4300  (~30 pix)
]
NARROW_BAND_TOTAL_PIX = sum(b - a for a, b in NARROW_BAND_RANGES)


def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) > 0 else np.nan


def compute_band_index(
    wave: np.ndarray, flux: np.ndarray,
    band: tuple, blue: tuple, red: tuple,
) -> float:
    band_mask = (wave >= band[0]) & (wave <= band[1])
    blue_mask = (wave >= blue[0]) & (wave <= blue[1])
    red_mask = (wave >= red[0]) & (wave <= red[1])
    f_band = _safe_mean(flux[band_mask])
    f_blue = _safe_mean(flux[blue_mask])
    f_red = _safe_mean(flux[red_mask])
    if not all(np.isfinite(v) for v in [f_band, f_blue, f_red]):
        return np.nan
    f_cont = 0.5 * (f_blue + f_red)
    if f_band <= 0 or f_cont <= 0:
        return np.nan
    return float(-2.5 * np.log10(f_band / f_cont))


def compute_cn_scores(dataset: "SpectraDataset") -> np.ndarray:
    cn3839 = np.array([
        compute_band_index(dataset.wave, dataset.X[i],
                           **{k: v for k, v in CN_BAND_DEFS["CN3839"].items()})
        for i in range(len(dataset))
    ])
    cn4142 = np.array([
        compute_band_index(dataset.wave, dataset.X[i],
                           **{k: v for k, v in CN_BAND_DEFS["CN4142"].items()})
        for i in range(len(dataset))
    ])
    return np.nan_to_num(cn3839, nan=0) + np.nan_to_num(cn4142, nan=0)


def get_clean_training_mask(
    dataset: "SpectraDataset",
    percentile: float = 95.0,
    min_positive: int = 5,
    verbose: bool = True,
) -> np.ndarray:
    if dataset.n_positive < min_positive:
        if verbose:
            print(f"CN pre-clean: skipped (< {min_positive} known positives)")
        return np.ones(len(dataset), dtype=bool)

    cn_scores = compute_cn_scores(dataset)
    unl_mask = dataset.y == -1
    unl_scores = cn_scores[unl_mask]
    threshold = np.nanpercentile(unl_scores, percentile)
    pos_scores = cn_scores[dataset.y == 1]

    clean_unl = unl_scores <= threshold
    n_removed = int((~clean_unl).sum())

    full_mask = np.ones(len(dataset), dtype=bool)
    full_mask[unl_mask] = clean_unl

    if verbose:
        print(f"CN pre-clean: threshold={threshold:.4f}, "
              f"removed {n_removed} unlabeled "
              f"({100*n_removed/max(unl_mask.sum(),1):.1f}%), "
              f"known-CN median={np.nanmedian(pos_scores):.4f}")
    return full_mask


# ═══════════════════════════════════════════════════════════════════
# SpectraDataset
# ═══════════════════════════════════════════════════════════════════

class SpectraDataset:
    """Container for spectral data and metadata.

    Attributes
    ----------
    X : np.ndarray (n_samples, n_pix)
        Normalised flux matrix (used for training).
    X_raw : np.ndarray (n_samples, n_pix) or None
        Interpolated but NOT continuum-normalised spectra — kept for
        visual inspection / line identification.
    y : np.ndarray (n_samples,)
        Labels: 1 = known CN star, -1 = unlabeled.
    meta : pd.DataFrame
        Metadata aligned with X.
    wave : np.ndarray (n_pix,)
        Wavelength grid.
    """

    def __init__(
        self, X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, wave: np.ndarray,
        X_raw: Optional[np.ndarray] = None,
    ):
        self.X = X
        self.X_raw = X_raw
        self.y = y
        self.meta = meta
        self.wave = wave

    def __len__(self) -> int: return len(self.X)

    @property
    def n_features(self) -> int: return self.X.shape[1]

    @property
    def n_positive(self) -> int: return int((self.y == 1).sum())

    @property
    def n_unlabeled(self) -> int: return int((self.y == -1).sum())

    def get_unlabeled(self) -> Tuple[np.ndarray, pd.DataFrame]:
        mask = self.y == -1
        return self.X[mask], self.meta.iloc[mask].reset_index(drop=True)

    def get_labeled(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        mask = self.y == 1
        return self.X[mask], self.y[mask], self.meta.iloc[mask].reset_index(drop=True)

    def to_torch_dataset(self, data: np.ndarray) -> TensorDataset:
        return TensorDataset(torch.from_numpy(data).float())


# ═══════════════════════════════════════════════════════════════════
# Main data-loading pipeline (segment normalisation)
# ═══════════════════════════════════════════════════════════════════

def load_spectra_data(
    stars_csv: str = "stars.csv",
    spectra_folder: str = "dr13_new",
    cn_catalogs: Optional[list] = None,
    common_wave: Optional[np.ndarray] = None,
    seg_width: float = 200.0,
    verbose: bool = True,
) -> SpectraDataset:
    """Load & preprocess LAMOST spectra with SEGMENT normalisation.

    Pipeline:  label → load → segment-norm → dedup → anomaly filter.

    Parameters
    ----------
    stars_csv : str
    spectra_folder : str
    cn_catalogs : list | None
        Default: ['CNstar.csv','FT_cands.csv','GCS.csv'].
    common_wave : np.ndarray | None
        Default: 3710–8890 Å at 1 Å step.
    seg_width : float
        Segment width for polynomial continuum fitting (Å).
    verbose : bool

    Returns
    -------
    SpectraDataset  (X = normalised, X_raw = interpolated un-normalised)
    """
    if cn_catalogs is None:
        cn_catalogs = ["CNstar.csv", "FT_cands.csv", "GCS.csv"]
    if common_wave is None:
        common_wave = np.arange(3710.0, 8890.0, 1.0)

    if verbose:
        print("=" * 60)
        print("Loading LAMOST spectra (segment normalisation)")
        print("=" * 60)

    # Use spectra_io pipeline with segment normalisation (now default)
    pipe = sio.run_screening_preprocess_from_files(
        stars_csv=stars_csv,
        common_wave=common_wave,
        folder=spectra_folder,
        cn_catalogs=cn_catalogs,
        cn_match_tolerance_arcsec=1.0,
        uid_col="uid",
        snr_col="snru",
        anomaly_low_pct=0.5,
        anomaly_high_pct=99.5,
        show_progress=verbose,
        norm_mode="segment",
    )

    X_clean = pipe["datacube_clean"]
    stars_clean = pipe["stars_clean"]
    # Raw spectra are available from the pipeline
    X_raw = pipe.get("datacube_raw")
    if X_raw is not None and "valid_indices" in pipe:
        # Re-align raw spectra to match cleaned data (same filtering steps)
        # The raw cube has different length than cleaned — just store for ref
        pass

    if verbose:
        print("\nPipeline summary:")
        for key, value in pipe["summary"].items():
            print(f"  {key}: {value}")
        print(f"  n_positive (final): {int((stars_clean['label'] == 1).sum())}")
        print(f"  n_unlabeled (final): {int((stars_clean['label'] == -1).sum())}")
        print(f"  Spectra shape: {X_clean.shape}")
        print("=" * 60)

    return SpectraDataset(
        X=X_clean,
        X_raw=None,  # raw from pipeline has different filtering — use dataset_full.X_raw for viz
        y=stars_clean["label"].values.astype(int),
        meta=stars_clean,
        wave=common_wave.copy(),
    )


# ═══════════════════════════════════════════════════════════════════
# DataLoaders
# ═══════════════════════════════════════════════════════════════════

def create_dataloaders(
    dataset: SpectraDataset,
    batch_size: int = 256,
    val_split: float = 0.1,
    random_seed: int = 42,
    clean_mask: Optional[np.ndarray] = None,
) -> Dict:
    rng = np.random.RandomState(random_seed)

    unl_base = dataset.y == -1
    train_eligible = unl_base & clean_mask if clean_mask is not None else unl_base
    X_eligible = dataset.X[train_eligible]

    n_el = len(X_eligible)
    indices = rng.permutation(n_el)
    n_val = int(n_el * val_split)
    X_train = X_eligible[indices[n_val:]]
    X_val_normal = X_eligible[indices[:n_val]]

    X_pos, y_pos, _ = dataset.get_labeled()
    X_val_full = np.vstack([X_val_normal, X_pos]) if len(X_pos) else X_val_normal
    y_val_full = np.concatenate([
        np.full(len(X_val_normal), -1), np.ones(len(X_pos)),
    ]) if len(X_pos) else np.full(len(X_val_normal), -1)

    return {
        "train": DataLoader(dataset.to_torch_dataset(X_train),
                           batch_size=batch_size, shuffle=True, drop_last=False),
        "val_normal": DataLoader(dataset.to_torch_dataset(X_val_normal),
                                batch_size=batch_size, shuffle=False),
        "val_full": DataLoader(TensorDataset(
            torch.from_numpy(X_val_full).float(),
            torch.from_numpy(y_val_full).long()),
            batch_size=batch_size, shuffle=False),
        "n_train_clean": len(X_train),
        "n_train_total": int(unl_base.sum()),
    }


# ═══════════════════════════════════════════════════════════════════
# Narrow-band extraction
# ═══════════════════════════════════════════════════════════════════

def extract_narrow_band_spectra(dataset: SpectraDataset) -> SpectraDataset:
    wave = dataset.wave
    band_slices, band_waves = [], []
    for lo, hi in NARROW_BAND_RANGES:
        mask = (wave >= lo) & (wave <= hi)
        band_slices.append(dataset.X[:, mask])
        band_waves.append(wave[mask])

    X_narrow = np.concatenate(band_slices, axis=1).astype(np.float32)
    wave_narrow = np.concatenate(band_waves)
    # Also slice raw spectra if available
    X_raw_narrow = None
    if dataset.X_raw is not None:
        raw_slices = [dataset.X_raw[:, (wave >= lo) & (wave <= hi)]
                      for lo, hi in NARROW_BAND_RANGES]
        X_raw_narrow = np.concatenate(raw_slices, axis=1).astype(np.float32)

    return SpectraDataset(
        X=X_narrow, X_raw=X_raw_narrow,
        y=dataset.y.copy(), meta=dataset.meta.copy(), wave=wave_narrow,
    )


# ═══════════════════════════════════════════════════════════════════
# Physics feature computation (12-D)
# ═══════════════════════════════════════════════════════════════════

def compute_physics_features(
    dataset: SpectraDataset,
    n_pca: int = 3,
    random_seed: int = 42,
) -> SpectraDataset:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    meta, wave, X = dataset.meta.copy(), dataset.wave, dataset.X
    n = len(X)

    # 1. Stellar parameters
    teff = pd.to_numeric(meta["teff"], errors="coerce").values
    logg = pd.to_numeric(meta["logg"], errors="coerce").values
    feh = pd.to_numeric(meta["feh"], errors="coerce").values

    # 2. Band indices
    def band_arr(band_name):
        return np.array([
            compute_band_index(wave, X[i],
                               **{k: v for k, v in CN_BAND_DEFS[band_name].items()})
            for i in range(n)
        ])
    cn3839, cn4142, ch4300 = band_arr("CN3839"), band_arr("CN4142"), band_arr("CH4300")
    for arr in [teff, logg, feh, cn3839, cn4142, ch4300]:
        arr[np.isnan(arr)] = np.nanmedian(arr)

    # 3. KNN delta features
    params = np.column_stack([teff, logg, feh, cn3839, cn4142, ch4300])
    X_params = StandardScaler().fit_transform(params)
    K = min(180, n - 1)
    knn = NearestNeighbors(n_neighbors=K + 1, metric="euclidean").fit(X_params)
    _, nn_idx = knn.kneighbors(X_params)
    nn_idx = nn_idx[:, 1:]  # exclude self

    def nb_med(vals):
        return np.array([np.nanmedian(vals[nn_idx[i]]) for i in range(n)])

    delta_cn3839 = cn3839 - nb_med(cn3839)
    delta_cn4142 = cn4142 - nb_med(cn4142)
    delta_ch4300 = ch4300 - nb_med(ch4300)

    # 4. PCA
    pca = PCA(n_components=n_pca, random_state=random_seed).fit_transform(X)

    X_feat = np.column_stack([
        teff, logg, feh, cn3839, cn4142, ch4300,
        delta_cn3839, delta_cn4142, delta_ch4300,
        pca[:, 0], pca[:, 1], pca[:, 2],
    ]).astype(np.float32)

    # Standardise (features have very different scales)
    X_feat = StandardScaler().fit_transform(X_feat).astype(np.float32)

    feat_names = [
        "teff","logg","feh","CN3839","CN4142","CH4300",
        "delta_CN3839","delta_CN4142","delta_CH4300",
        "pca_1","pca_2","pca_3",
    ]
    feat_meta = meta.copy()
    for j, name in enumerate(feat_names):
        feat_meta[name] = X_feat[:, j]

    return SpectraDataset(X=X_feat, y=dataset.y.copy(), meta=feat_meta,
                          wave=np.arange(len(feat_names), dtype=float))


# ═══════════════════════════════════════════════════════════════════
# V3: Cluster-aware physics feature computation
# ═══════════════════════════════════════════════════════════════════

def _cluster_knn_delta(
    band_values: np.ndarray,
    nn_idx: np.ndarray,
) -> np.ndarray:
    """Compute delta = value - median of KNN neighbors."""
    n = len(band_values)
    return np.array([band_values[i] - np.nanmedian(band_values[nn_idx[i]])
                     for i in range(n)])


def compute_physics_features_param_cluster(
    dataset: SpectraDataset,
    n_clusters: int = 45,
    k_neighbors: int = 180,
    n_pca: int = 3,
    random_seed: int = 42,
) -> dict:
    """V3 Approach 1: cluster by teff/logg/feh, then KNN within cluster.

    Returns dict with keys: dataset, cluster_labels, cluster_mean_spectra.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    meta, wave, X = dataset.meta.copy(), dataset.wave, dataset.X
    n = len(X)
    rng = np.random.RandomState(random_seed)

    # 1. Stellar params + band indices
    teff = pd.to_numeric(meta["teff"], errors="coerce").values
    logg = pd.to_numeric(meta["logg"], errors="coerce").values
    feh = pd.to_numeric(meta["feh"], errors="coerce").values

    def band_arr(name):
        return np.array([
            compute_band_index(wave, X[i],
                               **{k: v for k, v in CN_BAND_DEFS[name].items()})
            for i in range(n)
        ])
    cn3839, cn4142, ch4300 = band_arr("CN3839"), band_arr("CN4142"), band_arr("CH4300")
    for arr in [teff, logg, feh, cn3839, cn4142, ch4300]:
        arr[np.isnan(arr)] = np.nanmedian(arr)

    # 2. Cluster by teff/logg/feh only
    params_phys = np.column_stack([teff, logg, feh])
    params_phys_s = StandardScaler().fit_transform(params_phys)
    n_cl = int(np.clip(np.sqrt(n) / 2, 10, 45))
    kmeans = KMeans(n_clusters=n_cl, random_state=random_seed, n_init=20)
    cluster_labels = kmeans.fit_predict(params_phys_s)

    print(f"  Param-cluster: {n_cl} clusters (teff/logg/feh)")

    # 3. KNN within cluster using physical params only
    nn_idx_all = np.zeros((n, k_neighbors), dtype=int)
    global_knn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, n), metric="euclidean")
    global_knn.fit(params_phys_s)
    _, g_idx = global_knn.kneighbors(params_phys_s)
    g_idx = g_idx[:, 1:]

    for cid in np.unique(cluster_labels):
        cmask = cluster_labels == cid
        c_idx = np.where(cmask)[0]
        if len(c_idx) >= 3:
            local_X = params_phys_s[cmask]
            local_k = min(k_neighbors + 1, len(local_X))
            local_knn = NearestNeighbors(n_neighbors=local_k, metric="euclidean")
            local_knn.fit(local_X)
            pos = int(np.where(c_idx == np.arange(n)[cmask])[0][0])
            # process each star
            for j, gi in enumerate(c_idx):
                local_pos = int(np.where(c_idx == gi)[0][0])
                d_loc, idx_loc = local_knn.kneighbors(local_X[local_pos].reshape(1, -1))
                nb_local = c_idx[idx_loc[0]]
                nb = nb_local[nb_local != gi].tolist()
                # fill from global
                for gk in range(1, g_idx.shape[1]):
                    gv = g_idx[gi, gk]
                    if gv not in nb:
                        nb.append(int(gv))
                    if len(nb) >= k_neighbors:
                        break
                nn_idx_all[gi, :] = np.array(nb[:k_neighbors], dtype=int)
        else:
            nn_idx_all[cmask] = g_idx[cmask, :k_neighbors]

    # 4. Delta features
    delta_cn3839 = _cluster_knn_delta(cn3839, nn_idx_all)
    delta_cn4142 = _cluster_knn_delta(cn4142, nn_idx_all)
    delta_ch4300 = _cluster_knn_delta(ch4300, nn_idx_all)

    # 5. PCA on full spectra
    pca_comp = PCA(n_components=n_pca, random_state=random_seed).fit_transform(X)

    # 6. Assemble
    X_feat = np.column_stack([
        teff, logg, feh, cn3839, cn4142, ch4300,
        delta_cn3839, delta_cn4142, delta_ch4300,
        pca_comp[:, 0], pca_comp[:, 1], pca_comp[:, 2],
    ]).astype(np.float32)
    X_feat = StandardScaler().fit_transform(X_feat).astype(np.float32)

    feat_names = [
        "teff","logg","feh","CN3839","CN4142","CH4300",
        "delta_CN3839","delta_CN4142","delta_CH4300",
        "pca_1","pca_2","pca_3",
    ]
    feat_meta = meta.copy()
    for j, name in enumerate(feat_names):
        feat_meta[name] = X_feat[:, j]

    # Compute cluster mean spectra for visualization
    cluster_mean_spectra = {}
    for cid in np.unique(cluster_labels):
        cmask = cluster_labels == cid
        cluster_mean_spectra[int(cid)] = X[cmask].mean(axis=0)

    dataset_out = SpectraDataset(
        X=X_feat, y=dataset.y.copy(), meta=feat_meta,
        wave=np.arange(len(feat_names), dtype=float),
    )

    return {
        "dataset": dataset_out,
        "cluster_labels": cluster_labels,
        "cluster_mean_spectra": cluster_mean_spectra,
        "n_clusters": n_cl,
    }


def compute_physics_features_masked_cluster(
    dataset: SpectraDataset,
    n_clusters: int = 45,
    k_neighbors: int = 180,
    n_pca: int = 3,
    random_seed: int = 42,
) -> dict:
    """V3 Approach 2: mask CN/CH bands, cluster by PCA on masked spectra.

    Returns dict with keys: dataset, cluster_labels, cluster_mean_spectra.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    meta, wave, X = dataset.meta.copy(), dataset.wave, dataset.X
    n = len(X)
    rng = np.random.RandomState(random_seed)

    # 1. Stellar params + band indices
    teff = pd.to_numeric(meta["teff"], errors="coerce").values
    logg = pd.to_numeric(meta["logg"], errors="coerce").values
    feh = pd.to_numeric(meta["feh"], errors="coerce").values

    def band_arr(name):
        return np.array([
            compute_band_index(wave, X[i],
                               **{k: v for k, v in CN_BAND_DEFS[name].items()})
            for i in range(n)
        ])
    cn3839, cn4142, ch4300 = band_arr("CN3839"), band_arr("CN4142"), band_arr("CH4300")
    for arr in [teff, logg, feh, cn3839, cn4142, ch4300]:
        arr[np.isnan(arr)] = np.nanmedian(arr)

    # 2. Mask CN/CH bands → replace with median flux
    masked_X = X.copy()
    band_masks = []
    for lo, hi in [(3830, 3883), (4120, 4216), (4285, 4315)]:
        m = (wave >= lo) & (wave <= hi)
        band_masks.append(m)
    fill_val = np.nanmedian(masked_X[:, ~np.any(np.array(band_masks), axis=0)], axis=1)
    for m in band_masks:
        masked_X[:, m] = fill_val[:, None]

    # 3. PCA on masked spectra → KMeans
    n_comp = int(min(25, masked_X.shape[1], max(2, n - 1)))
    X_masked_pca = PCA(n_components=n_comp, random_state=random_seed).fit_transform(masked_X)
    n_cl = int(np.clip(np.sqrt(n) / 2, 10, 45))
    kmeans = KMeans(n_clusters=n_cl, random_state=random_seed, n_init=20)
    cluster_labels = kmeans.fit_predict(X_masked_pca)

    print(f"  Masked-cluster: {n_cl} clusters (PCA-{n_comp} on masked spectra)")

    # 4. KNN within cluster using physical params
    params_phys = np.column_stack([teff, logg, feh])
    params_phys_s = StandardScaler().fit_transform(params_phys)
    nn_idx_all = np.zeros((n, k_neighbors), dtype=int)
    global_knn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, n), metric="euclidean")
    global_knn.fit(params_phys_s)
    _, g_idx = global_knn.kneighbors(params_phys_s)
    g_idx = g_idx[:, 1:]

    for cid in np.unique(cluster_labels):
        cmask = cluster_labels == cid
        c_idx = np.where(cmask)[0]
        if len(c_idx) >= 3:
            local_X = params_phys_s[cmask]
            local_k = min(k_neighbors + 1, len(local_X))
            local_knn = NearestNeighbors(n_neighbors=local_k, metric="euclidean")
            local_knn.fit(local_X)
            for gi in c_idx:
                local_pos = int(np.where(c_idx == gi)[0][0])
                d_loc, idx_loc = local_knn.kneighbors(local_X[local_pos].reshape(1, -1))
                nb_local = c_idx[idx_loc[0]]
                nb = nb_local[nb_local != gi].tolist()
                for gk in range(1, g_idx.shape[1]):
                    gv = g_idx[gi, gk]
                    if gv not in nb:
                        nb.append(int(gv))
                    if len(nb) >= k_neighbors:
                        break
                nn_idx_all[gi, :] = np.array(nb[:k_neighbors], dtype=int)
        else:
            nn_idx_all[cmask] = g_idx[cmask, :k_neighbors]

    # 5. Delta features
    delta_cn3839 = _cluster_knn_delta(cn3839, nn_idx_all)
    delta_cn4142 = _cluster_knn_delta(cn4142, nn_idx_all)
    delta_ch4300 = _cluster_knn_delta(ch4300, nn_idx_all)

    # 6. PCA on full (unmasked) spectra
    pca_comp = PCA(n_components=n_pca, random_state=random_seed).fit_transform(X)

    # 7. Assemble
    X_feat = np.column_stack([
        teff, logg, feh, cn3839, cn4142, ch4300,
        delta_cn3839, delta_cn4142, delta_ch4300,
        pca_comp[:, 0], pca_comp[:, 1], pca_comp[:, 2],
    ]).astype(np.float32)
    X_feat = StandardScaler().fit_transform(X_feat).astype(np.float32)

    feat_names = [
        "teff","logg","feh","CN3839","CN4142","CH4300",
        "delta_CN3839","delta_CN4142","delta_CH4300",
        "pca_1","pca_2","pca_3",
    ]
    feat_meta = meta.copy()
    for j, name in enumerate(feat_names):
        feat_meta[name] = X_feat[:, j]

    # Cluster mean spectra for visualization
    cluster_mean_spectra = {}
    for cid in np.unique(cluster_labels):
        cmask = cluster_labels == cid
        cluster_mean_spectra[int(cid)] = X[cmask].mean(axis=0)

    dataset_out = SpectraDataset(
        X=X_feat, y=dataset.y.copy(), meta=feat_meta,
        wave=np.arange(len(feat_names), dtype=float),
    )

    return {
        "dataset": dataset_out,
        "cluster_labels": cluster_labels,
        "cluster_mean_spectra": cluster_mean_spectra,
        "n_clusters": n_cl,
    }
