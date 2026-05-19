"""Standard NYU Depth V2 evaluation metrics (meters, masked valid pixels)."""

import numpy as np

from config import MAX_DEPTH_M, MIN_DEPTH_M


def compute_depth_metrics(
    pred_m: np.ndarray,
    gt_m: np.ndarray,
    mask: np.ndarray,
    min_depth: float = MIN_DEPTH_M,
    max_depth: float = MAX_DEPTH_M,
) -> dict[str, float]:
    """
    NYU-style metrics on valid pixels.
    pred_m, gt_m: depth in meters (H, W)
    mask: valid pixel mask (H, W), float or bool
    """
    valid = (
        (mask > 0)
        & (gt_m > min_depth)
        & (gt_m < max_depth)
        & np.isfinite(pred_m)
        & np.isfinite(gt_m)
    )
    if valid.sum() == 0:
        return {}

    pred = pred_m[valid]
    gt = gt_m[valid]

    thresh = np.maximum(pred / gt, gt / pred)
    abs_rel = np.mean(np.abs(pred - gt) / gt)
    sq_rel = np.mean(((pred - gt) ** 2) / gt)
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    log10 = np.mean(np.abs(np.log10(pred + 1e-8) - np.log10(gt + 1e-8)))

    return {
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "rmse": float(rmse),
        "log10": float(log10),
        "delta1": float((thresh < 1.25).mean()),
        "delta2": float((thresh < 1.25 ** 2).mean()),
        "delta3": float((thresh < 1.25 ** 3).mean()),
    }


def aggregate_metrics(metric_list: list[dict[str, float]]) -> dict[str, float]:
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    return {k: float(np.mean([m[k] for m in metric_list if k in m])) for k in keys}
