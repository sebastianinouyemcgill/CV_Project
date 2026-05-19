"""Evaluate a saved checkpoint on the full NYU test split."""

import argparse

import torch
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, IMAGE_SIZE, TEST_CSV
from dataset import NYUDepthDataset
from model import UNet
from train import get_device, run_epoch
from utils import compute_depth_metrics, show_prediction


def evaluate(checkpoint_path, batch_size=16, num_workers=2, show_plot=True):
    device = get_device()
    test_dataset = NYUDepthDataset(TEST_CSV, image_size=IMAGE_SIZE)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = UNet().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    import train as train_module

    train_module.DEVICE = device

    val_loss, val_metrics = run_epoch(model, test_loader)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"val_loss={val_loss:.4f}")
    print(
        f"rmse={val_metrics['rmse']:.3f}m | "
        f"abs_rel={val_metrics['abs_rel']:.3f} | "
        f"delta1={val_metrics['delta1']:.3f} | "
        f"delta2={val_metrics.get('delta2', 0):.3f} | "
        f"delta3={val_metrics.get('delta3', 0):.3f}"
    )

    if show_plot:
        rgb, depth, mask = test_dataset[0]
        with torch.no_grad():
            pred = model(rgb.unsqueeze(0).to(device))
        show_prediction(rgb, depth, pred.cpu()[0])

    return val_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(CHECKPOINT_DIR / "best_unet.pt"),
        help="Path to model weights",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.batch_size, args.num_workers, show_plot=not args.no_plot)


if __name__ == "__main__":
    main()
