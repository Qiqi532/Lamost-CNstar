"""nnPU (non-negative PU learning) loss implementation.

Reference:
  Kiryo et al. (2017) "Positive-Unlabeled Learning with Non-Negative Risk Estimator"
  NIPS 2017.  https://arxiv.org/abs/1703.00593

The unbiased PU risk estimator (uPU):
  R_pu = π_p * E_p[ℓ(f(x), +1)] + E_u[ℓ(f(x), -1)] - π_p * E_p[ℓ(f(x), -1)]

The non-negative correction (nnPU) prevents overfitting by clamping:
  R_nnpu = π_p * E_p[ℓ(f(x), +1)] + max(0, E_u[ℓ(f(x), -1)] - π_p * E_p[ℓ(f(x), -1)])

where π_p = p(y=1) is the class prior (~73/33565 ≈ 0.00217).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Sigmoid loss: ℓ(f(x), y) = log(1 + exp(-y * f(x))) = softplus(-y * f(x)).

    Args:
        logits: (N,) raw logits from model.
        y: (N,) labels in {+1, -1}.

    Returns:
        (N,) per-sample loss.
    """
    return F.softplus(-y * logits)


def upu_risk(
    logits_pos: torch.Tensor,
    logits_unl: torch.Tensor,
    pi_p: float,
    loss_func: str = "sigmoid",
) -> dict:
    """Compute uPU (unbiased PU) risk and its components.

    Args:
        logits_pos: (n_pos,) logits for positive samples.
        logits_unl: (n_unl,) logits for unlabeled samples.
        pi_p: class prior p(y=1).
        loss_func: "sigmoid" (default) or "logistic".

    Returns:
        dict with keys: risk (total), r_pos_plus, r_unl_minus, r_pos_minus,
        clamped_term (always 0 for uPU), loss_p, loss_n.
    """
    if loss_func == "sigmoid":
        # ℓ(f(x), +1) → softplus(-logits)
        # ℓ(f(x), -1) → softplus(logits)
        loss_pos_plus = F.softplus(-logits_pos)   # loss when labeled as +1
        loss_pos_minus = F.softplus(logits_pos)    # loss when labeled as -1
        loss_unl_minus = F.softplus(logits_unl)    # loss when labeled as -1
    else:
        # logistic loss
        loss_pos_plus = F.logsigmoid(logits_pos).neg()   # -log σ(f)
        loss_pos_minus = F.logsigmoid(-logits_pos).neg() # -log σ(-f)
        loss_unl_minus = F.logsigmoid(-logits_unl).neg() # -log σ(-f)

    # Empirical expectations
    r_pos_plus = loss_pos_plus.mean()           # E_p[ℓ(f(x), +1)]
    r_pos_minus = loss_pos_minus.mean()         # E_p[ℓ(f(x), -1)]
    r_unl_minus = loss_unl_minus.mean()         # E_u[ℓ(f(x), -1)]

    # uPU risk: π_p * R^+_p + R^-_u - π_p * R^-_p
    risk = pi_p * r_pos_plus + r_unl_minus - pi_p * r_pos_minus

    return {
        "risk": risk,
        "r_pos_plus": r_pos_plus,
        "r_unl_minus": r_unl_minus,
        "r_pos_minus": r_pos_minus,
        "clamped_term": torch.tensor(0.0, device=logits_pos.device),
        "loss_p": loss_pos_plus,
        "loss_n": loss_unl_minus,
    }


def nnpu_risk(
    logits_pos: torch.Tensor,
    logits_unl: torch.Tensor,
    pi_p: float,
    loss_func: str = "sigmoid",
) -> dict:
    """Compute nnPU (non-negative PU) risk.

    Clamps the negative-class risk term to be non-negative:
      R_nnpu = π_p * R^+_p + max(0, R^-_u - π_p * R^-_p)

    Args:
        logits_pos: (n_pos,) logits for positive samples.
        logits_unl: (n_unl,) logits for unlabeled samples.
        pi_p: class prior p(y=1).
        loss_func: "sigmoid" (default) or "logistic".

    Returns:
        dict with same keys as upu_risk().
    """
    if loss_func == "sigmoid":
        loss_pos_plus = F.softplus(-logits_pos)
        loss_pos_minus = F.softplus(logits_pos)
        loss_unl_minus = F.softplus(logits_unl)
    else:
        loss_pos_plus = F.logsigmoid(logits_pos).neg()
        loss_pos_minus = F.logsigmoid(-logits_pos).neg()
        loss_unl_minus = F.logsigmoid(-logits_unl).neg()

    r_pos_plus = loss_pos_plus.mean()
    r_pos_minus = loss_pos_minus.mean()
    r_unl_minus = loss_unl_minus.mean()

    # Non-negative clamp
    clamped = torch.clamp(r_unl_minus - pi_p * r_pos_minus, min=0.0)
    risk = pi_p * r_pos_plus + clamped

    return {
        "risk": risk,
        "r_pos_plus": r_pos_plus,
        "r_unl_minus": r_unl_minus,
        "r_pos_minus": r_pos_minus,
        "clamped_term": clamped,
        "loss_p": loss_pos_plus,
        "loss_n": loss_unl_minus,
    }


class nnPULoss(nn.Module):
    """nnPU loss as a PyTorch module.

    Usage:
        criterion = nnPULoss(pi_p=0.002, clamp=True)
        loss_dict = criterion(logits, labels)  # labels: 1=positive, 0=unlabeled
        loss = loss_dict["risk"]
    """

    def __init__(
        self,
        pi_p: float = 0.002,
        clamp: bool = True,
        loss_func: str = "sigmoid",
    ):
        super().__init__()
        self.pi_p = pi_p
        self.clamp = clamp
        self.loss_func = loss_func

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
        """Compute nnPU/uPU loss.

        Args:
            logits: (B,) raw logits.
            labels: (B,) 1 = positive, 0 = unlabeled.

        Returns:
            dict with risk and all components.
        """
        pos_mask = labels == 1
        unl_mask = labels == 0

        logits_pos = logits[pos_mask]
        logits_unl = logits[unl_mask]

        if len(logits_pos) == 0:
            # No positives in batch → fall back to treating all as unlabeled
            result = {
                "risk": F.softplus(logits_unl).mean() if len(logits_unl) > 0
                        else torch.tensor(0.0, device=logits.device, requires_grad=True),
                "r_pos_plus": torch.tensor(0.0, device=logits.device),
                "r_unl_minus": torch.tensor(0.0, device=logits.device),
                "r_pos_minus": torch.tensor(0.0, device=logits.device),
                "clamped_term": torch.tensor(0.0, device=logits.device),
                "loss_p": torch.tensor(0.0, device=logits.device),
                "loss_n": torch.tensor(0.0, device=logits.device),
            }
            return result

        if self.clamp:
            result = nnpu_risk(logits_pos, logits_unl, self.pi_p, self.loss_func)
        else:
            result = upu_risk(logits_pos, logits_unl, self.pi_p, self.loss_func)
        return result
