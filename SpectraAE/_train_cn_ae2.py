"""CN-aware AE training with log file output."""
import sys, pickle, time
from pathlib import Path
import numpy as np

log = open("SpectraAE/_cache/cn_ae_train.log", "w", buffering=1)

def log_print(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from SpectraAE.cn_aware_pretrain import cn_aware_pretrain_autoencoder

log_print("Loading X_clean.npy...", flush=True)
X = np.load("ML/_cache/X_clean.npy").astype(np.float32)
log_print(f"Loaded: {X.shape}  size={X.nbytes/1024/1024:.1f}MB")

log_print("\n=== Starting CN-aware AE training (band_weight=5.0) ===")
t0 = time.time()

model, history = cn_aware_pretrain_autoencoder(
    X,
    latent_dim=64, base_ch=32, band_weight=5.0,
    n_epochs=200, batch_size=256, lr=1e-3, patience=30,
    device="cuda", checkpoint_dir="SpectraAE/checkpoints/cn_aware",
    verbose=True,
)

elapsed = time.time() - t0
log_print(f"\nTraining done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
log_print(f"Best epoch: {history['best_epoch']}")
log_print(f"Best val_loss: {history['best_val_loss']:.6f}")

# Save history
cache_dir = Path("SpectraAE/_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
with open(cache_dir / "cn_aware_history.pkl", "wb") as f:
    pickle.dump(history, f)
log_print("History saved to SpectraAE/_cache/cn_aware_history.pkl")

# Quick eval
import torch
import torch.nn as nn
ckpt = torch.load("SpectraAE/checkpoints/cn_aware/ae_best.pt", map_location="cuda", weights_only=False)
log_print(f"Checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")
log_print(f"  val_band_loss={ckpt.get('val_band_loss', 'N/A')}")
log_print(f"  val_cont_loss={ckpt.get('val_cont_loss', 'N/A')}")

log_print("\nDone.")
log.close()
