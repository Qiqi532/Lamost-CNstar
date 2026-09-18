"""BinaryClassifier — binary classification for CN-star detection."""

from .models import BinaryClassifier, SpectralConvEncoder, PhysicsMLPEncoder
from .augmentation import augment_spectra
from .trainer import BinaryTrainer, EnsembleTrainer
from .evaluate import evaluate_model, evaluate_ensemble, export_candidates
from .pipeline import run_pipeline, slice_wave_range, WAVE_START, WAVE_END
