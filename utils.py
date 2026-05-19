"""Training utilities: visualization, seeding, checkpoint I/O."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import IMAGENET_MEAN, IMAGENET_STD, MAX_DEPTH_M
from metrics import compute_depth_metrics


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def depth_to_meters(depth_tensor, max_depth_m=MAX_DEPTH_M):
    return depth_tensor.squeeze().detach().cpu().numpy() * max_depth_m


def unnormalize_rgb(rgb_tensor):
    """Convert ImageNet-normalized RGB back to [0, 1] for display."""
    mean = torch.tensor(IMAGENET_MEAN, device=rgb_tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=rgb_tensor.device).view(3, 1, 1)
    return (rgb_tensor * std + mean).clamp(0, 1)


def _prediction_figure(rgb, gt_depth, pred_depth, mask=None, max_depth_m=MAX_DEPTH_M):
    """Build RGB | GT | Pred | (optional error) figure; returns fig."""
    if rgb.dim() == 3 and rgb.shape[0] == 3:
        rgb_np = unnormalize_rgb(rgb).permute(1, 2, 0).cpu().numpy()
    else:
        rgb_np = np.clip(rgb, 0, 1)
    gt_m = depth_to_meters(gt_depth, max_depth_m)
    pred_m = depth_to_meters(pred_depth, max_depth_m)

    ncols = 4 if mask is not None else 3
    fig, ax = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 3:
        ax = list(ax)

    ax[0].imshow(np.clip(rgb_np, 0, 1))
    ax[0].set_title("RGB")
    ax[0].axis("off")
    ax[1].imshow(gt_m, cmap="plasma", vmin=0, vmax=max_depth_m)
    ax[1].set_title("Ground Truth (m)")
    ax[1].axis("off")
    ax[2].imshow(pred_m, cmap="plasma", vmin=0, vmax=max_depth_m)
    ax[2].set_title("Prediction (m)")
    ax[2].axis("off")

    if mask is not None:
        mask_np = mask.squeeze().cpu().numpy() > 0
        err = np.abs(pred_m - gt_m)
        err[~mask_np] = 0
        im = ax[3].imshow(err, cmap="hot", vmin=0, vmax=1.0)
        ax[3].set_title("|Error| (m)")
        ax[3].axis("off")
        plt.colorbar(im, ax=ax[3], fraction=0.046)

    plt.tight_layout()
    return fig


def show_prediction(rgb, gt_depth, pred_depth, max_depth_m=MAX_DEPTH_M):
    fig = _prediction_figure(rgb, gt_depth, pred_depth, max_depth_m=max_depth_m)
    plt.show()


def save_and_display_prediction(
    rgb,
    gt_depth,
    pred_depth,
    mask,
    save_path: Path,
    max_depth_m=MAX_DEPTH_M,
    display: bool = True,
) -> Path:
    """
    Save prediction figure to disk and display inline (Colab/Jupyter).
    Always saves to file first — reliable in headless Colab.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = _prediction_figure(rgb, gt_depth, pred_depth, mask, max_depth_m)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    if display:
        try:
            from IPython.display import Image as IPImage
            from IPython.display import display as ipy_display

            ipy_display(IPImage(filename=str(save_path)))
        except ImportError:
            show_prediction(rgb, gt_depth, pred_depth, max_depth_m)

    return save_path


def save_validation_figure(
    rgb,
    gt_depth,
    pred_depth,
    mask,
    save_path: Path,
    max_depth_m=MAX_DEPTH_M,
):
    """Save RGB | GT | Pred | error heatmap for one sample."""
    save_and_display_prediction(
        rgb, gt_depth, pred_depth, mask, save_path, max_depth_m, display=False
    )


def estimate_height(depth_map_m):
    valid = depth_map_m[depth_map_m > 0.1]
    if len(valid) == 0:
        return 0.0
    return float(valid.max() - valid.min())


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_rmse, scaler=None):
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "best_rmse": best_rmse,
        "scaler": scaler.state_dict() if scaler else None,
    }
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if optimizer and ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        return ckpt.get("epoch", 0), ckpt.get("best_rmse", float("inf"))
    # Legacy: raw state_dict only
    model.load_state_dict(ckpt)
    return 0, float("inf")
