"""Run CN-aware AE training with output captured to file."""
import sys, pickle, time, os
from pathlib import Path

# Setup project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Redirect stdout to log file
log_path = Path("SpectraAE/_cache/cn_ae_train_full.log")
log_path.parent.mkdir(parents=True, exist_ok=True)
log_fh = open(log_path, "w", buffering=1)

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.__stdout__, log_fh)

import numpy as np
import torch

from SpectraAE.cn_aware_pretrain import cn_aware_pretrain_autoencoder

print("=" * 60)
print("CN-Aware AE Training")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# Load data
print("\nLoading X_clean.npy...")
X = np.load("ML/_cache/X_clean.npy").astype(np.float32)
print(f"Loaded: {X.shape} ({X.nbytes/1024/1024:.1f} MB)")

# Train
t0 = time.time()
model, history = cn_aware_pretrain_autoencoder(
    X,
    latent_dim=64, base_ch=32, band_weight=5.0,
    n_epochs=200, batch_size=256, lr=1e-3, patience=30,
    device="cuda", checkpoint_dir="SpectraAE/checkpoints/cn_aware",
    verbose=True,
)
elapsed = time.time() - t0

print(f"\n{'='*60}")
print(f"Training Complete")
print(f"{'='*60}")
print(f"Best epoch: {history['best_epoch']}")
print(f"Best val_loss: {history['best_val_loss']:.6f}")
print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"Band losses @ best: train={history['train_band_losses'][history['best_epoch']-1]:.4f}")
print(f"Cont losses @ best:  train={history['train_cont_losses'][history['best_epoch']-1]:.4f}")

# Save history
cache_dir = Path("SpectraAE/_cache")
with open(cache_dir / "cn_aware_history.pkl", "wb") as f:
    pickle.dump(history, f)
print("History saved.")

log_fh.close()
print("Done. Log at SpectraAE/_cache/cn_ae_train_full.log")
