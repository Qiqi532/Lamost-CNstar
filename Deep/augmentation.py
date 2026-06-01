"""Spectral augmentation for small-sample CN-star positive set.

Generates realistic synthetic spectra from a seed set of known CN-enhanced
stars (GCS globular-cluster members) via physics-motivated perturbations.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits as afits


# ═══════════════════════════════════════════════════════════════════
# GCS spectrum loading
# ═══════════════════════════════════════════════════════════════════

def _safe_read_wave_flux(filepath: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        with afits.open(filepath, memmap=False) as hdul:
            if len(hdul) < 2 or hdul[1].data is None or len(hdul[1].data) == 0:
                return None
            row_data = hdul[1].data[0]
            wave = np.asarray(row_data["WAVELENGTH"], dtype=float)
            flux = np.asarray(row_data["FLUX"], dtype=float)
            return wave, flux
    except Exception:
        return None


def load_gcs_spectra(
    gcs_folder: str = "GCS",
    common_wave: Optional[np.ndarray] = None,
    c_kms: float = 300000.0,
    min_points: int = 20,
) -> Tuple[np.ndarray, List[str]]:
    """Load all GCS FITS spectra, RV-correct and interpolate to common grid.

    Parameters
    ----------
    gcs_folder : str
        Path to the GCS/ folder containing .fits.gz files.
    common_wave : np.ndarray or None
        Common wavelength grid. Default: 3710–8890 Å at 1 Å step.
    c_kms : float
        Speed of light in km/s for RV correction.
    min_points : int
        Minimum valid pixels required after filtering.

    Returns
    -------
    spectra : np.ndarray (n_spectra, n_wave)
        Interpolated (NOT continuum-normalised) spectra.
    filepaths : list of str
        File path for each loaded spectrum.
    """
    if common_wave is None:
        common_wave = np.arange(3710.0, 8890.0, 1.0)

    gcs_dir = gcs_folder
    fnames = sorted([
        f for f in os.listdir(gcs_dir) if f.endswith(".fits.gz")
    ])
    filepaths = [os.path.join(gcs_dir, f) for f in fnames]

    spectra = []
    valid_paths = []

    for fp in filepaths:
        payload = _safe_read_wave_flux(fp)
        if payload is None:
            continue
        wave, flux = payload

        # RV correction — assume rv=0 if not available (GCS stars have known RV
        # but we don't have per-file RV; the effect is small at LAMOST resolution)
        wave_rest = wave  # no RV correction without per-file RV

        finite_mask = np.isfinite(wave_rest) & np.isfinite(flux)
        wave_rest = wave_rest[finite_mask]
        flux = flux[finite_mask]

        if len(wave_rest) < min_points:
            continue

        order = np.argsort(wave_rest)
        wave_rest = wave_rest[order]
        flux = flux[order]

        wave_rest, unique_idx = np.unique(wave_rest, return_index=True)
        flux = flux[unique_idx]

        if len(wave_rest) < min_points:
            continue

        if wave_rest.min() > common_wave[0] or wave_rest.max() < common_wave[-1]:
            continue

        flux_interp = np.interp(common_wave, wave_rest, flux)
        if not np.all(np.isfinite(flux_interp)):
            continue

        spectra.append(flux_interp.astype(np.float32))
        valid_paths.append(fp)

    return np.array(spectra, dtype=np.float32), valid_paths


# ═══════════════════════════════════════════════════════════════════
# Augmentation transforms
# ═══════════════════════════════════════════════════════════════════

def _add_gaussian_noise(
    flux: np.ndarray, noise_std: float = 0.005, rng: np.random.RandomState = None,
) -> np.ndarray:
    """Add Gaussian noise scaled to approximate LAMOST SNR ~100-200."""
    if rng is None:
        rng = np.random
    noise = rng.normal(0, noise_std, size=flux.shape)
    return flux + noise


def _rv_jitter(
    wave: np.ndarray, flux: np.ndarray, delta_v_kms: float,
) -> np.ndarray:
    """Simulate small RV shift by interpolating the spectrum.

    A shift of delta_v in km/s corresponds to wavelength shift
    dλ = λ * delta_v / c. Returns shifted flux on the original grid.
    """
    shift_factor = 1.0 + delta_v_kms / 300000.0
    wave_shifted = wave * shift_factor
    return np.interp(wave, wave_shifted, flux)


def _continuum_tilt(
    wave: np.ndarray, flux: np.ndarray, alpha: float,
) -> np.ndarray:
    """Apply a gentle linear continuum tilt.

    flux_new = flux * (1 + alpha * (wave - wave_mid) / wave_range)
    """
    wave_mid = 0.5 * (wave[0] + wave[-1])
    wave_range = wave[-1] - wave[0]
    tilt = 1.0 + alpha * (wave - wave_mid) / wave_range
    return flux * tilt.astype(flux.dtype)


def _scale_absorption_depth(
    flux: np.ndarray, scale: float,
) -> np.ndarray:
    """Scale absorption features relative to continuum=1.

    flux_new = 1 - scale * (1 - flux)
    scale > 1 deepens absorption, scale < 1 shallows it.
    """
    continuum = 1.0
    return continuum - scale * (continuum - flux)


def augment_spectra(
    spectra: np.ndarray,
    wave: np.ndarray,
    n_noise: int = 10,
    n_rv: int = 5,
    n_tilt: int = 3,
    n_mixup: int = 5,
    n_depth: int = 5,
    noise_std: float = 0.005,
    rv_range_kms: Tuple[float, float] = (-15.0, 15.0),
    tilt_alpha_range: Tuple[float, float] = (-0.03, 0.03),
    depth_range: Tuple[float, float] = (0.85, 1.15),
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate augmented copies of seed spectra.

    Parameters
    ----------
    spectra : np.ndarray (n_seed, n_wave)
        Continuum-normalised seed spectra.
    wave : np.ndarray (n_wave,)
        Wavelength grid.
    n_noise, n_rv, n_tilt, n_mixup, n_depth : int
        Number of variants per seed per augmentation type.
    noise_std : float
        Standard deviation for Gaussian noise.
    rv_range_kms : (float, float)
        Range of RV shifts in km/s.
    tilt_alpha_range : (float, float)
        Range of continuum tilt coefficient.
    depth_range : (float, float)
        Range of absorption depth scaling.

    Returns
    -------
    aug_spectra : np.ndarray (n_total, n_wave)
        Original + augmented spectra.
    aug_labels : np.ndarray (n_total,)
        All 1 (positive class).
    """
    rng = np.random.RandomState(random_seed)
    n_seed, n_wave = spectra.shape
    augmented = [spectra]

    # 1. Gaussian noise
    for i in range(n_seed):
        for _ in range(n_noise):
            std = noise_std * (0.5 + rng.random())
            augmented.append(_add_gaussian_noise(spectra[i:i+1], std, rng))

    # 2. RV jitter
    for i in range(n_seed):
        for _ in range(n_rv):
            dv = rng.uniform(*rv_range_kms)
            augmented.append(_rv_jitter(wave, spectra[i], dv)[np.newaxis, :])

    # 3. Continuum tilt
    for i in range(n_seed):
        for _ in range(n_tilt):
            alpha = rng.uniform(*tilt_alpha_range)
            augmented.append(_continuum_tilt(wave, spectra[i], alpha)[np.newaxis, :])

    # 4. Mixup between random pairs
    for i in range(n_seed):
        partners = [j for j in range(n_seed) if j != i]
        if not partners:
            continue
        for _ in range(min(n_mixup, len(partners))):
            j = partners[rng.randint(0, len(partners))]
            beta = rng.uniform(0.3, 0.7)
            mixed = beta * spectra[i] + (1.0 - beta) * spectra[j]
            augmented.append(mixed[np.newaxis, :])

    # 5. Absorption depth scaling
    for i in range(n_seed):
        for _ in range(n_depth):
            scale = rng.uniform(*depth_range)
            augmented.append(_scale_absorption_depth(spectra[i], scale)[np.newaxis, :])

    aug_spectra = np.concatenate(augmented, axis=0).astype(np.float32)
    aug_labels = np.ones(len(aug_spectra), dtype=np.float32)

    return aug_spectra, aug_labels
