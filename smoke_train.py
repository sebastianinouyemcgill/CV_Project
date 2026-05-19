"""Quick training run to verify the pipeline (subset of NYU, few epochs)."""

import torch
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_DIR,
    DATA_ROOT,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODEL_NAME,
    PRETRAINED,
    TEST_CSV,
    TRAIN_CSV,
    USE_AMP,
    WEIGHT_DECAY,
)
from dataset import NYUDepthDataset
from losses import DepthLoss
from model import build_model
from train import get_device, run_epoch
from utils import set_seed

TRAIN_SAMPLES = 64
TEST_SAMPLES = 32
BATCH_SIZE = 4
EPOCHS = 2
NUM_WORKERS = 0
SEED = 42


def main():
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    USE_AMP_RUNTIME = USE_AMP and device.type == "cuda"

    print(f"NYU data root: {DATA_ROOT}")
    print(f"Device: {device} | Model: {MODEL_NAME} | AMP: {USE_AMP_RUNTIME}")

    train_dataset = NYUDepthDataset(
        TRAIN_CSV, image_size=IMAGE_SIZE, max_samples=TRAIN_SAMPLES, augment=True, normalize=True
    )
    test_dataset = NYUDepthDataset(
        TEST_CSV, image_size=IMAGE_SIZE, max_samples=TEST_SAMPLES, augment=False, normalize=True
    )
    print(f"Mini train samples: {len(train_dataset)}, test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model(MODEL_NAME, pretrained=PRETRAINED).to(device)
    criterion = DepthLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=USE_AMP_RUNTIME)

    import train as train_module
    train_module.DEVICE = device

    for epoch in range(EPOCHS):
        train_loss, _, _ = run_epoch(
            model, train_loader, criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )
        val_loss, val_metrics, _ = run_epoch(
            model, test_loader, criterion,
            device=device,
            epoch=epoch,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_rmse={val_metrics.get('rmse', 0):.3f}m | "
            f"val_abs_rel={val_metrics.get('abs_rel', 0):.3f} | "
            f"val_delta1={val_metrics.get('delta1', 0):.3f}"
        )

    ckpt = CHECKPOINT_DIR / "smoke_resnet.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved {ckpt}")
    print("Smoke training finished successfully.")


if __name__ == "__main__":
    main()