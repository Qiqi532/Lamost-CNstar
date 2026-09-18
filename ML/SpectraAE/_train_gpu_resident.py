"""GPU-resident CN-aware AE training — 200x faster than DataLoader approach.

All data pre-loaded to GPU memory. 33,589 spectra use ~180MB GPU.
Expected: ~3-5 seconds per epoch on RTX 4060.
"""
import sys, pickle, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from SpectraAE.models.autoencoder import ConvAutoencoder
from SpectraAE.cn_aware_pretrain import create_band_weight_mask, WeightedMSELoss, CN_BAND_DEFS

# Config
BAND_WEIGHT = 5.0
N_EPOCHS = 200
PATIENCE = 30
BATCH_SIZE = 512
LR = 1e-3
LATENT_DIM = 64
BASE_CH = 32
DEVICE = "cuda"
SEED = 42

rng = np.random.RandomState(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print(f"CN-Aware AE (GPU-resident)")
print(f"band_weight={BAND_WEIGHT}x  epochs={N_EPOCHS}  patience={PATIENCE}")
print("=" * 60, flush=True)

# Load & normalize data
print("Loading X_clean.npy...", flush=True)
X = np.load("ML/_cache/X_clean.npy").astype(np.float32)
print(f"Loaded: {X.shape} ({X.nbytes/1024/1024:.1f} MB)", flush=True)

lo = float(np.percentile(X, 1))
hi = float(np.percentile(X, 99))
X_clipped = np.clip(X, lo, hi)
n_total = len(X_clipped)
n_val = int(n_total * 0.1)
indices = rng.permutation(n_total)
val_idx = indices[:n_val]
tr_idx = indices[n_val:]
scaler_mean = float(X_clipped[tr_idx].mean())
scaler_std = float(X_clipped[tr_idx].std())
X_train_np = ((X_clipped[tr_idx] - scaler_mean) / scaler_std).astype(np.float32)
X_val_np = ((X_clipped[val_idx] - scaler_mean) / scaler_std).astype(np.float32)
print(f"Train: {len(X_train_np):,}  Val: {len(X_val_np):,}", flush=True)

# Pre-load ALL data to GPU
print("Moving data to GPU...", flush=True)
X_train = torch.from_numpy(X_train_np).unsqueeze(1).to(DEVICE)  # (N, 1, 700)
X_val = torch.from_numpy(X_val_np).unsqueeze(1).to(DEVICE)
n_train = len(X_train)
n_val_samples = len(X_val)
print(f"GPU memory: train={X_train.element_size() * X_train.numel() / 1024/1024:.0f}MB  "
      f"val={X_val.element_size() * X_val.numel() / 1024/1024:.0f}MB", flush=True)

# Weight mask
weight_mask = create_band_weight_mask(n_pixels=700, band_weight=BAND_WEIGHT).to(DEVICE)
band_mask_binary = (weight_mask > 1.1).float()
n_band_px = int(band_mask_binary.sum().item())
print(f"Band pixels: {n_band_px}  Continuum: {700 - n_band_px}", flush=True)

# Model
model = ConvAutoencoder(in_channels=1, base_ch=BASE_CH, latent_dim=LATENT_DIM).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} params", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)
criterion = WeightedMSELoss(weight_mask)

# Checkpoint
ckpt_dir = Path("SpectraAE/checkpoints/cn_aware")
ckpt_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = ckpt_dir / "ae_best.pt"

def compute_band_cont_loss(recon, target):
    """Fast GPU-native band/continuum loss computation. Returns per-pixel MSE."""
    B = recon.shape[0]
    sq_err = (recon - target) ** 2
    total = sq_err.mean().item()
    band_sq = sq_err * band_mask_binary
    band_loss = band_sq.sum().item() / (B * n_band_px)
    cont_mask = 1.0 - band_mask_binary
    cont_sq = sq_err * cont_mask
    cont_loss = cont_sq.sum().item() / (B * (700 - n_band_px))
    return total, band_loss, cont_loss

# Training
best_val_loss = float("inf")
best_epoch = 0
no_improve = 0
train_losses, val_losses = [], []
train_band_losses, val_band_losses = [], []
train_cont_losses, val_cont_losses = [], []
t0 = time.time()

for epoch in range(1, N_EPOCHS + 1):
    # Shuffle training data
    perm = torch.randperm(n_train, device=DEVICE)
    X_train_shuffled = X_train[perm]

    # Train
    model.train()
    tr_loss_sum = 0.0
    tr_band_sum = 0.0
    tr_cont_sum = 0.0
    for i in range(0, n_train, BATCH_SIZE):
        batch_x = X_train_shuffled[i:i + BATCH_SIZE]
        optimizer.zero_grad()
        recon, _ = model(batch_x)
        loss = criterion(recon, batch_x)
        loss.backward()
        optimizer.step()
        tr_loss_sum += loss.item() * len(batch_x)
        total_b, band_b, cont_b = compute_band_cont_loss(recon, batch_x)
        tr_band_sum += band_b * len(batch_x)
        tr_cont_sum += cont_b * len(batch_x)
    tr_loss = tr_loss_sum / n_train
    train_losses.append(tr_loss)
    train_band_losses.append(tr_band_sum / n_train)
    train_cont_losses.append(tr_cont_sum / n_train)
    scheduler.step()

    # Validate
    model.eval()
    val_loss_sum = 0.0
    val_band_sum = 0.0
    val_cont_sum = 0.0
    with torch.no_grad():
        for i in range(0, n_val_samples, BATCH_SIZE * 2):
            batch_x = X_val[i:i + BATCH_SIZE * 2]
            recon, _ = model(batch_x)
            val_loss_sum += criterion(recon, batch_x).item() * len(batch_x)
            _, band_b, cont_b = compute_band_cont_loss(recon, batch_x)
            val_band_sum += band_b * len(batch_x)
            val_cont_sum += cont_b * len(batch_x)
    val_loss = val_loss_sum / n_val_samples
    val_losses.append(val_loss)
    val_band_losses.append(val_band_sum / n_val_samples)
    val_cont_losses.append(val_cont_sum / n_val_samples)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        no_improve = 0
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss, "val_band_loss": val_band_losses[-1],
            "val_cont_loss": val_cont_losses[-1],
            "scaler_mean": scaler_mean, "scaler_std": scaler_std,
            "band_weight": BAND_WEIGHT, "band_defs": CN_BAND_DEFS,
        }, ckpt_path)
    else:
        no_improve += 1

    if epoch % 10 == 0 or epoch == 1:
        lr_now = scheduler.get_last_lr()[0]
        marker = "*" if no_improve == 0 else " "
        elapsed = time.time() - t0
        print(f"E {epoch:3d}/{N_EPOCHS} | {marker} "
              f"tr={tr_loss:.4f} val={val_loss:.4f} "
              f"b={val_band_losses[-1]:.4f} c={val_cont_losses[-1]:.4f} "
              f"lr={lr_now:.2e} [{elapsed:.0f}s/{elapsed/epoch:.1f}s per ep]", flush=True)

    if no_improve >= PATIENCE:
        print(f"Early stop @ epoch {epoch} (best={best_val_loss:.4f} @ {best_epoch})", flush=True)
        break

elapsed = time.time() - t0
print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f}min)  epochs={epoch}  best_epoch={best_epoch}", flush=True)

# Restore best
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

history = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "train_band_losses": train_band_losses,
    "train_cont_losses": train_cont_losses,
    "val_band_losses": val_band_losses,
    "val_cont_losses": val_cont_losses,
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss,
    "scaler_mean": scaler_mean,
    "scaler_std": scaler_std,
    "elapsed_seconds": elapsed,
    "band_weight": BAND_WEIGHT,
}

cache_dir = Path("SpectraAE/_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
with open(cache_dir / "cn_aware_history.pkl", "wb") as f:
    pickle.dump(history, f)
print(f"History saved. Done.", flush=True)
