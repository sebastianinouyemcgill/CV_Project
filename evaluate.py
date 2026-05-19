"""Evaluate a saved checkpoint on the full NYU test split."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, IMAGE_SIZE, MODEL_NAME, TEST_CSV
from dataset import NYUDepthDataset
from losses import DepthLoss
from experiments import ExperimentRun
from model import build_model
from train import get_device, run_epoch
from utils import load_checkpoint, save_and_display_prediction


def evaluate(
    checkpoint_path,
    batch_size=16,
    num_workers=2,
    show_plot=True,
    save_path: Path | str | None = None,
    display: bool = True,
):
    device = get_device()
    checkpoint_path = Path(checkpoint_path)

    test_dataset = NYUDepthDataset(TEST_CSV, image_size=IMAGE_SIZE, augment=False, normalize=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(MODEL_NAME, pretrained=False).to(device)
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    criterion = DepthLoss()
    import train as train_module

    train_module.DEVICE = device

    val_loss, val_metrics, _ = run_epoch(model, test_loader, criterion, device=device)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"val_loss={val_loss:.4f}")
    print(
        f"rmse={val_metrics.get('rmse', 0):.3f}m | "
        f"abs_rel={val_metrics.get('abs_rel', 0):.3f} | "
        f"sq_rel={val_metrics.get('sq_rel', 0):.3f} | "
        f"delta1={val_metrics.get('delta1', 0):.3f} | "
        f"delta2={val_metrics.get('delta2', 0):.3f} | "
        f"delta3={val_metrics.get('delta3', 0):.3f}"
    )

    if save_path is None:
        save_path = checkpoint_path.parent / "eval_result.png"
    save_path = Path(save_path)

    rgb, depth, mask = test_dataset[0]
    with torch.no_grad():
        pred = model(rgb.unsqueeze(0).to(device))

    if show_plot or display:
        save_and_display_prediction(
            rgb,
            depth,
            pred.cpu()[0],
            mask,
            save_path,
            display=display,
        )
        print(f"Saved evaluation figure -> {save_path}")

    return val_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Path to .pt file (default: latest run's best.pt)",
    )
    parser.add_argument("--run-name", default="", help="Use checkpoints/<run-name>/best.pt")
    parser.add_argument("--save-path", default="", help="Where to save eval figure")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    elif args.run_name:
        ckpt = CHECKPOINT_DIR / args.run_name / "best.pt"
    else:
        run = ExperimentRun(CHECKPOINT_DIR, resume=True)
        ckpt = run.best_path

    save_path = Path(args.save_path) if args.save_path else ckpt.parent / "eval_result.png"

    evaluate(
        ckpt,
        args.batch_size,
        args.num_workers,
        show_plot=not args.no_plot,
        save_path=save_path,
        display=not args.no_display,
    )


if __name__ == "__main__":
    main()
