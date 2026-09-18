"""Quick training script for CN-aware AE."""
import sys, pickle
from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from SpectraAE.cn_aware_pretrain import cn_aware_pretrain_autoencoder

X = np.load("ML/_cache/X_clean.npy").astype(np.float32)
print(f"Loaded X_clean: {X.shape}")

model, history = cn_aware_pretrain_autoencoder(
    X,
    latent_dim=64, base_ch=32, band_weight=5.0,
    n_epochs=200, batch_size=256, lr=1e-3, patience=30,
    device="cuda", checkpoint_dir="SpectraAE/checkpoints/cn_aware",
    verbose=True,
)

print(f"\nTraining complete.")
print(f"Best epoch: {history['best_epoch']}, best_val_loss: {history['best_val_loss']:.6f}")
print(f"Time: {history['elapsed_seconds']:.0f}s ({history['elapsed_seconds']/60:.1f}min)")

# Save history
cache_dir = Path("SpectraAE/_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
with open(cache_dir / "cn_aware_history.pkl", "wb") as f:
    pickle.dump(history, f)
print("History saved.")
