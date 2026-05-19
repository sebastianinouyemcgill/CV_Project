"""Depth estimation losses (NYU / indoor monocular depth)."""

import torch
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)


def scale_invariant_log_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Scale-invariant logarithmic loss (Eigen et al.).
    Operates in log-depth space; stable for indoor scenes.
    """
    pred = pred.clamp(min=eps)
    target = target.clamp(min=eps)
    log_diff = torch.log(pred) - torch.log(target)
    m = mask > 0
    if m.sum() == 0:
        return pred.sum() * 0.0
    ld = log_diff[m]
    return torch.sqrt((ld ** 2).mean() - 0.5 * (ld.mean() ** 2).clamp(min=0.0) + eps)


def berhu_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reverse Huber (BerHu) loss — robust to outliers."""
    diff = torch.abs(pred - target) * mask
    valid = mask.sum() + eps
    abs_diff = diff[mask > 0]
    if abs_diff.numel() == 0:
        return pred.sum() * 0.0
    delta = 0.2 * abs_diff.max().detach().clamp(min=1e-3)
    l1 = abs_diff
    l2 = (abs_diff ** 2 + delta ** 2) / (2 * delta)
    berhu = torch.where(abs_diff <= delta, l1, l2)
    return berhu.mean()


def gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """L1 on spatial gradients — preserves edges and thin structures."""
    def grads(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    pred_dx, pred_dy = grads(pred)
    tgt_dx, tgt_dy = grads(target)
    mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    loss_dx = _masked_mean(torch.abs(pred_dx - tgt_dx), mask_dx)
    loss_dy = _masked_mean(torch.abs(pred_dy - tgt_dy), mask_dy)
    return loss_dx + loss_dy


def edge_aware_smoothness(
    pred: torch.Tensor,
    image: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize depth gradients where the RGB image is smooth;
    allows sharp depth changes at object boundaries.
    """
    pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

    img_dx = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), dim=1, keepdim=True)
    img_dy = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), dim=1, keepdim=True)

    wx = torch.exp(-img_dx)
    wy = torch.exp(-img_dy)

    mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    smooth_x = _masked_mean(pred_dx * wx, mask_dx)
    smooth_y = _masked_mean(pred_dy * wy, mask_dy)
    return smooth_x + smooth_y


class DepthLoss(torch.nn.Module):
    """Combined depth loss for NYU indoor depth estimation."""

    def __init__(
        self,
        sil_weight: float = 1.0,
        berhu_weight: float = 0.5,
        grad_weight: float = 0.1,
        smooth_weight: float = 0.01,
    ):
        super().__init__()
        self.sil_weight = sil_weight
        self.berhu_weight = berhu_weight
        self.grad_weight = grad_weight
        self.smooth_weight = smooth_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        sil = scale_invariant_log_loss(pred, target, mask)
        berhu = berhu_loss(pred, target, mask)
        grad = gradient_loss(pred, target, mask)

        total = self.sil_weight * sil + self.berhu_weight * berhu + self.grad_weight * grad

        log = {
            "sil": float(sil.detach()),
            "berhu": float(berhu.detach()),
            "grad": float(grad.detach()),
        }

        if self.smooth_weight > 0 and image is not None:
            smooth = edge_aware_smoothness(pred, image, mask)
            total = total + self.smooth_weight * smooth
            log["smooth"] = float(smooth.detach())

        log["total"] = float(total.detach())
        return total, log
