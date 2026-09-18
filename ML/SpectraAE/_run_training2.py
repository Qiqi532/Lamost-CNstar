"""Run CN-aware AE training - simpler version that just writes to stdout (no redirect)."""
import sys, pickle, time
from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from SpectraAE.cn_aware_pretrain import cn_aware_pretrain_autoencoder

print("Loading X_clean.npy...", flush=True)
X = np.load("ML/_cache/X_clean.npy").astype(np.float32)
print(f"Loaded: {X.shape} ({X.nbytes/1024/1024:.1f} MB)", flush=True)

print("Starting training...", flush=True)
t0 = time.time()

model, history = cn_aware_pretrain_autoencoder(
    X,
    latent_dim=64, base_ch=32, band_weight=5.0,
    n_epochs=200, batch_size=256, lr=1e-3, patience=30,
    device="cuda", checkpoint_dir="SpectraAE/checkpoints/cn_aware",
    verbose=True,
)
elapsed = time.time() - t0

print(f"\nTraining Complete: best_epoch={history['best_epoch']}, best_val={history['best_val_loss']:.6f}, time={elapsed:.0f}s", flush=True)

# Save history
cache_dir = Path("SpectraAE/_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
with open(cache_dir / "cn_aware_history.pkl", "wb") as f:
    pickle.dump(history, f)
print("History saved.", flush=True)
print("Done.", flush=True)
