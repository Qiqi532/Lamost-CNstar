"""Training and evaluation for GNN node classification."""

import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from .gnn_models import build_model, normalize_adjacency, sparse_mx_to_torch

# Use LabelSpreading's graph utilities when available
try:
    from LabelSpreading.graph_utils import compute_cluster_zscore
except ImportError:
    compute_cluster_zscore = None


def prepare_gnn_data(
    X_features: np.ndarray,
    adj,
    y_labels: np.ndarray,
    device: torch.device,
) -> Dict:
    """Prepare features, adjacency, and labels for GNN training.

    Parameters
    ----------
    X_features : (N, D) feature matrix (spectra PCA or physics features).
    adj : scipy sparse affinity matrix.
    y_labels : (N,) labels: 1=positive, 0=negative, -1=unlabeled.
    device : torch device.

    Returns dict with keys: features, adj_norm, y_true, pos_mask, unlabeled_mask.
    """
    features = torch.FloatTensor(X_features.astype(np.float32)).to(device)
    adj_norm = normalize_adjacency(adj)
    adj_norm = sparse_mx_to_torch(adj_norm).to(device)

    y_true = torch.FloatTensor((y_labels == 1).astype(np.float32)).to(device)
    pos_mask = y_labels == 1
    unlabeled_mask = y_labels == -1

    return {
        "features": features,
        "adj_norm": adj_norm,
        "y_true": y_true,
        "pos_mask": pos_mask,
        "unlabeled_mask": unlabeled_mask,
    }


def train_gnn(
    model: nn.Module,
    data: Dict,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    pos_weight: float = 100.0,
    patience: int = 30,
    verbose: bool = True,
) -> Dict:
    """Train a GNN model.

    Parameters
    ----------
    model : GCN or GAT model.
    data : dict from prepare_gnn_data().
    train_mask, val_mask : boolean arrays of training/validation indices.
    epochs, lr, weight_decay, pos_weight, patience : training hyperparams.

    Returns dict with training history and best model state.
    """
    features = data["features"]
    adj_norm = data["adj_norm"]
    y_true = data["y_true"]
    device = features.device

    pos_weight_tensor = torch.tensor([pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_pr = -1
    best_state = None
    history = {"train_loss": [], "val_pr_auc": [], "val_roc_auc": []}
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj_norm)
        loss = criterion(logits[train_mask], y_true[train_mask])
        loss.backward()
        optimizer.step()

        history["train_loss"].append(float(loss))

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features, adj_norm)
            val_prob = torch.sigmoid(val_logits[val_mask]).cpu().numpy()
            val_y = y_true[val_mask].cpu().numpy()

        if val_y.sum() > 0:
            val_pr = average_precision_score(val_y, val_prob)
            val_roc = roc_auc_score(val_y, val_prob)
        else:
            val_pr = 0.0
            val_roc = 0.5

        history["val_pr_auc"].append(float(val_pr))
        history["val_roc_auc"].append(float(val_roc))

        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch+1}, best val PR-AUC={best_val_pr:.4f}")
            break

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  [{epoch+1:3d}/{epochs}] loss={loss:.4f}  val PR={val_pr:.4f}  val ROC={val_roc:.4f}")

    model.load_state_dict(best_state)
    return {"history": history, "best_val_pr_auc": best_val_pr}


def predict_all(model: nn.Module, data: Dict) -> np.ndarray:
    """Generate predictions for all nodes."""
    model.eval()
    with torch.no_grad():
        logits = model(data["features"], data["adj_norm"])
        prob = torch.sigmoid(logits).cpu().numpy()
    return prob


def evaluate_gnn_topk(
    prob_cn: np.ndarray,
    y_labels: np.ndarray,
    topk_list: list = [50, 100, 200],
) -> Dict:
    """Compute top-K precision using known labels."""
    y_true = (y_labels == 1).astype(int)
    n_pos = max(int(y_true.sum()), 1)
    order = np.argsort(prob_cn)[::-1]
    metrics = {}
    for k in topk_list:
        top_k = y_true[order[:k]]
        metrics[f"precision@{k}"] = float(top_k.mean())
        metrics[f"recall@{k}"] = float(top_k.sum() / n_pos)
    metrics["n_pos"] = n_pos
    metrics["median_prob_known"] = float(np.median(prob_cn[y_true == 1]))
    return metrics
