"""Deep learning module for LAMOST CN-star detection.

Provides DeepSVDD for one-class anomaly detection on stellar spectra,
plus GCS-augmented binary classification, segment-based continuum
normalisation, narrow-band extraction, and physics feature computation.
"""

from .config import DeepSVDDConfig
from .data import (
    SpectraDataset,
    load_spectra_data,
    segment_normalize,
    normalize_datacube_segment,
    extract_narrow_band_spectra,
    compute_physics_features,
    compute_physics_features_param_cluster,
    compute_physics_features_masked_cluster,
    NARROW_BAND_RANGES,
    NARROW_BAND_TOTAL_PIX,
)
from .models import DeepSVDD, CNStarEncoder, CNStarConvEncoder
from .train import DeepSVDDTrainer
from .evaluate import evaluate_model, select_candidates, summarize_results

# Binary classifier modules
from .augmentation import load_gcs_spectra, augment_spectra
from .binary_trainer import BinaryClassifier, BinaryTrainer
from .evaluate_binary import (
    evaluate_classifier,
    evaluate_on_all_test_sets,
    summarize_binary_results,
    select_candidates_binary,
)
from .gcs_classifier import run_gcs_pipeline

__all__ = [
    # Config
    "DeepSVDDConfig",
    # Data
    "SpectraDataset",
    "load_spectra_data",
    "segment_normalize",
    "normalize_datacube_segment",
    "extract_narrow_band_spectra",
    "compute_physics_features",
    "NARROW_BAND_RANGES",
    "NARROW_BAND_TOTAL_PIX",
    # DeepSVDD
    "DeepSVDD",
    "CNStarEncoder",
    "CNStarConvEncoder",
    "DeepSVDDTrainer",
    "evaluate_model",
    "select_candidates",
    "summarize_results",
    # Binary classifier (GCS-augmented)
    "load_gcs_spectra",
    "augment_spectra",
    "BinaryClassifier",
    "BinaryTrainer",
    "evaluate_classifier",
    "evaluate_on_all_test_sets",
    "summarize_binary_results",
    "select_candidates_binary",
    "run_gcs_pipeline",
]
