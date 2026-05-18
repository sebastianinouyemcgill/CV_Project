import matplotlib.pyplot as plt
import numpy as np


def show_prediction(rgb, gt_depth, pred_depth):
    rgb = rgb.permute(1, 2, 0).cpu().numpy()
    gt_depth = gt_depth.squeeze().cpu().numpy()
    pred_depth = pred_depth.squeeze().detach().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(rgb)
    ax[0].set_title("RGB")

    ax[1].imshow(gt_depth, cmap='plasma')
    ax[1].set_title("Ground Truth")

    ax[2].imshow(pred_depth, cmap='plasma')
    ax[2].set_title("Prediction")

    plt.show()


def estimate_height(depth_map):
    """
    Simple example:
    estimate object height from depth range
    """

    valid = depth_map[depth_map > 0.1]

    if len(valid) == 0:
        return 0

    return valid.max() - valid.min()