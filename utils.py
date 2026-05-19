import numpy as np
import torch
import matplotlib.pyplot as plt

from config import MAX_DEPTH_M


def masked_mse_loss(pred, target, mask, eps=1e-6):
    diff = (pred - target) ** 2 * mask
    return diff.sum() / (mask.sum() + eps)


def depth_to_meters(depth_tensor, max_depth_m=MAX_DEPTH_M):
    return depth_tensor.squeeze().detach().cpu().numpy() * max_depth_m


def compute_depth_metrics(pred_m, gt_m, mask):
    """Standard NYU metrics on valid pixels (pred and gt in meters)."""
    valid = mask.astype(bool)
    if valid.sum() == 0:
        return {}

    pred = pred_m[valid]
    gt = gt_m[valid]

    thresh = np.maximum(pred / gt, gt / pred)
    abs_rel = np.mean(np.abs(pred - gt) / gt)
    rmse = np.sqrt(np.mean((pred - gt) ** 2))

    return {
        "abs_rel": float(abs_rel),
        "rmse": float(rmse),
        "delta1": float((thresh < 1.25).mean()),
        "delta2": float((thresh < 1.25 ** 2).mean()),
        "delta3": float((thresh < 1.25 ** 3).mean()),
    }


def show_prediction(rgb, gt_depth, pred_depth, max_depth_m=MAX_DEPTH_M):
    rgb = rgb.permute(1, 2, 0).cpu().numpy()
    gt_depth = depth_to_meters(gt_depth, max_depth_m)
    pred_depth = depth_to_meters(pred_depth, max_depth_m)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(np.clip(rgb, 0, 1))
    ax[0].set_title("RGB")

    ax[1].imshow(gt_depth, cmap="plasma", vmin=0, vmax=max_depth_m)
    ax[1].set_title("Ground Truth (m)")

    ax[2].imshow(pred_depth, cmap="plasma", vmin=0, vmax=max_depth_m)
    ax[2].set_title("Prediction (m)")

    plt.tight_layout()
    plt.show()


def estimate_height(depth_map_m):
    """Rough depth span in meters over valid pixels."""
    valid = depth_map_m[depth_map_m > 0.1]
    if len(valid) == 0:
        return 0.0
    return float(valid.max() - valid.min())
