"""Spectral augmentation for extreme-imbalance PU learning.

Given only ~50 positive training samples, augmentation is essential to
prevent overfitting.  Generates realistic variants via physics-motivated
perturbations: noise, RV jitter, continuum tilt, absorption scaling,
and mixup between positive samples.
"""

from typing import Tuple

import numpy as np


def augment_positive_spectra(
    spectra: np.ndarray,          # (n_pos, n_pixels)
    wave: np.ndarray,             # (n_pixels,)
    n_noise: int = 5,
    n_rv: int = 3,
    n_tilt: int = 2,
    n_depth: int = 3,
    n_mixup: int = 4,
    noise_std: float = 0.005,
    rv_range: Tuple[float, float] = (-15.0, 15.0),
    tilt_range: Tuple[float, float] = (-0.03, 0.03),
    depth_range: Tuple[float, float] = (0.85, 1.15),
    random_seed: int = 42,
) -> np.ndarray:
    """Augment positive samples with physics-motivated transforms.

    Parameters
    ----------
    spectra : (n_pos, n_pix) continuum-normalised positive spectra.
    wave : (n_pix,) wavelength grid.
    n_noise, n_rv, n_tilt, n_depth, n_mixup : int
        Number of variants per seed spectrum per augmentation type.
    noise_std : float
        Std of additive Gaussian noise.
    rv_range : (min, max) km/s
        Range of RV jitter.
    tilt_range : (min, max)
        Range of linear continuum tilt coefficient.
    depth_range : (min, max)
        Range of absorption depth scaling factor.

    Returns
    -------
    aug_spectra : (n_total, n_pix) original + augmented spectra.
    """
    rng = np.random.RandomState(random_seed)
    n_pos, n_pix = spectra.shape
    augmented = [spectra]

    # 1. Gaussian noise
    for i in range(n_pos):
        for _ in range(n_noise):
            std = noise_std * (0.5 + rng.random())
            noise = rng.normal(0, std, size=n_pix)
            augmented.append((spectra[i] + noise).astype(np.float32)[np.newaxis, :])

    # 2. RV jitter (small wavelength shift via interpolation)
    for i in range(n_pos):
        for _ in range(n_rv):
            dv = rng.uniform(*rv_range)
            shift = 1.0 + dv / 300000.0
            wave_shifted = wave * shift
            shifted = np.interp(wave, wave_shifted, spectra[i])
            augmented.append(shifted.astype(np.float32)[np.newaxis, :])

    # 3. Continuum tilt
    wave_mid = 0.5 * (wave[0] + wave[-1])
    wave_range = wave[-1] - wave[0]
    for i in range(n_pos):
        for _ in range(n_tilt):
            alpha = rng.uniform(*tilt_range)
            tilt = 1.0 + alpha * (wave - wave_mid) / wave_range
            augmented.append((spectra[i] * tilt).astype(np.float32)[np.newaxis, :])

    # 4. Absorption depth scaling
    continuum = 1.0
    for i in range(n_pos):
        for _ in range(n_depth):
            scale = rng.uniform(*depth_range)
            scaled = continuum - scale * (continuum - spectra[i])
            augmented.append(scaled.astype(np.float32)[np.newaxis, :])

    # 5. Mixup between random positive pairs
    if n_pos >= 2:
        for i in range(n_pos):
            partners = [j for j in range(n_pos) if j != i]
            for _ in range(min(n_mixup, len(partners))):
                j = partners[rng.randint(0, len(partners))]
                lam = rng.uniform(0.3, 0.7)
                mixed = lam * spectra[i] + (1.0 - lam) * spectra[j]
                augmented.append(mixed.astype(np.float32)[np.newaxis, :])

    return np.concatenate(augmented, axis=0).astype(np.float32)
