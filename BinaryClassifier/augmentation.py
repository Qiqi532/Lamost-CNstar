"""Spectral augmentation for small-sample CN-star positive set.

Generates realistic synthetic spectra via physics-motivated perturbations:
Gaussian noise, RV jitter, continuum tilt, mixup, absorption depth scaling.
"""

"""Spectral augmentation for small-sample CN-star positive set."""

from typing import Tuple

import numpy as np

def _add_gaussian_noise(
    flux: np.ndarray, noise_std: float = 0.005, rng: np.random.RandomState = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random
    noise = rng.normal(0, noise_std, size=flux.shape)
    return flux + noise


def _rv_jitter(
    wave: np.ndarray, flux: np.ndarray, delta_v_kms: float,
) -> np.ndarray:
    shift_factor = 1.0 + delta_v_kms / 300000.0
    wave_shifted = wave * shift_factor
    return np.interp(wave, wave_shifted, flux)


def _continuum_tilt(
    wave: np.ndarray, flux: np.ndarray, alpha: float,
) -> np.ndarray:
    wave_mid = 0.5 * (wave[0] + wave[-1])
    wave_range = wave[-1] - wave[0]
    tilt = 1.0 + alpha * (wave - wave_mid) / wave_range
    return flux * tilt.astype(flux.dtype)


def _scale_absorption_depth(flux: np.ndarray, scale: float) -> np.ndarray:
    return 1.0 - scale * (1.0 - flux)


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

    Each seed produces ~n_noise + n_rv + n_tilt + n_mixup + n_depth variants.
    """
    rng = np.random.RandomState(random_seed)
    n_seed = len(spectra)
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

    # 4. Mixup
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
