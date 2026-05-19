"""Quick training run to verify the pipeline (subset of NYU, few epochs)."""

import torch
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_DIR,
    DATA_ROOT,
    IMAGE_SIZE,
    LEARNING_RATE,
    TEST_CSV,
    TRAIN_CSV,
)
from dataset import NYUDepthDataset
from model import UNet
from train import get_device, run_epoch

# Small enough for M1 MacBook Air smoke test; scale up in Colab for real training.
TRAIN_SAMPLES = 64
TEST_SAMPLES = 32
BATCH_SIZE = 4
EPOCHS = 2
NUM_WORKERS = 0


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"NYU data root: {DATA_ROOT}")
    print(f"Device: {device}")

    train_dataset = NYUDepthDataset(TRAIN_CSV, image_size=IMAGE_SIZE, max_samples=TRAIN_SAMPLES)
    test_dataset = NYUDepthDataset(TEST_CSV, image_size=IMAGE_SIZE, max_samples=TEST_SAMPLES)
    print(f"Mini train samples: {len(train_dataset)}, test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    import train as train_module

    train_module.DEVICE = device

    for epoch in range(EPOCHS):
        train_loss, _ = run_epoch(model, train_loader, optimizer)
        val_loss, val_metrics = run_epoch(model, test_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_rmse={val_metrics['rmse']:.3f}m | "
            f"val_abs_rel={val_metrics['abs_rel']:.3f} | "
            f"val_delta1={val_metrics['delta1']:.3f}"
        )

    ckpt = CHECKPOINT_DIR / "smoke_unet.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved {ckpt}")
    print("Smoke training finished successfully.")


if __name__ == "__main__":
    main()
