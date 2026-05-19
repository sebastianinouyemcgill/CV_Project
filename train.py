import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_ROOT,
    EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    MAX_DEPTH_M,
    MAX_TRAIN_SAMPLES,
    NUM_WORKERS,
    TEST_CSV,
    TRAIN_CSV,
    TRAIN_SUBSAMPLE_SEED,
)
from dataset import NYUDepthDataset
from model import UNet
from utils import (
    compute_depth_metrics,
    depth_to_meters,
    estimate_height,
    masked_mse_loss,
    show_prediction,
)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    metric_sums = {"abs_rel": 0.0, "rmse": 0.0, "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}
    metric_count = 0

    for rgb, depth, mask in tqdm(loader, leave=False):
        rgb = rgb.to(DEVICE)
        depth = depth.to(DEVICE)
        mask = mask.to(DEVICE)

        pred = model(rgb)
        loss = masked_mse_loss(pred, depth, mask)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        if not is_train:
            pred_np = pred.detach().cpu().numpy()
            gt_np = depth.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()
            batch_size = pred_np.shape[0]
            for i in range(batch_size):
                pred_m = pred_np[i, 0] * MAX_DEPTH_M
                gt_m = gt_np[i, 0] * MAX_DEPTH_M
                metrics = compute_depth_metrics(pred_m, gt_m, mask_np[i, 0])
                if metrics:
                    for key in metric_sums:
                        metric_sums[key] += metrics[key]
                    metric_count += 1

    avg_loss = total_loss / max(len(loader), 1)
    avg_metrics = {k: v / max(metric_count, 1) for k, v in metric_sums.items()}
    return avg_loss, avg_metrics


def main():
    print(f"NYU data root: {DATA_ROOT}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Device: {DEVICE}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = NYUDepthDataset(
        TRAIN_CSV,
        image_size=IMAGE_SIZE,
        max_samples=MAX_TRAIN_SAMPLES,
        subsample_seed=TRAIN_SUBSAMPLE_SEED,
    )
    test_dataset = NYUDepthDataset(TEST_CSV, image_size=IMAGE_SIZE)
    train_note = (
        f"{len(train_dataset)} (random subset)"
        if MAX_TRAIN_SAMPLES
        else f"{len(train_dataset)} (full)"
    )
    print(f"Train samples: {train_note}, Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE == "cuda",
    )

    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_rmse = float("inf")
    best_path = CHECKPOINT_DIR / "best_unet.pt"

    for epoch in range(EPOCHS):
        train_loss, _ = run_epoch(model, train_loader, optimizer)
        val_loss, val_metrics = run_epoch(model, test_loader)

        saved = ""
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            torch.save(model.state_dict(), best_path)
            saved = f" | saved best -> {best_path}"

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_rmse={val_metrics['rmse']:.3f}m | "
            f"val_delta1={val_metrics['delta1']:.3f}"
            f"{saved}"
        )

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        print(f"Loaded best checkpoint for visualization (val_rmse={best_rmse:.3f}m)")

    model.eval()
    rgb, depth, mask = test_dataset[0]
    with torch.no_grad():
        pred = model(rgb.unsqueeze(0).to(DEVICE))

    show_prediction(rgb, depth, pred.cpu()[0])
    pred_m = depth_to_meters(pred.cpu()[0])
    print("Estimated depth span (m):", estimate_height(pred_m))


if __name__ == "__main__":
    main()
