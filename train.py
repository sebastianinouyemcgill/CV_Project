"""NYU Depth V2 training loop with pretrained encoder, combined loss, AMP, and logging."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_ROOT,
    EPOCHS,
    GRAD_ACCUM_STEPS,
    GRAD_CLIP_NORM,
    IMAGE_SIZE,
    LEARNING_RATE,
    LOSS_BERHU_WEIGHT,
    LOSS_GRAD_WEIGHT,
    LOSS_SIL_WEIGHT,
    LOSS_SMOOTH_WEIGHT,
    MAX_DEPTH_M,
    MAX_TRAIN_SAMPLES,
    MODEL_NAME,
    NUM_WORKERS,
    PERSISTENT_WORKERS,
    PRETRAINED,
    RUN_NAME,
    SAVE_EVERY_EPOCH,
    SEED,
    TEST_CSV,
    TRAIN_CSV,
    TRAIN_SUBSAMPLE_SEED,
    USE_AMP,
    USE_COMPILE,
    VIZ_EVERY,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from dataset import NYUDepthDataset
from experiments import ExperimentRun
from losses import DepthLoss
from metrics import aggregate_metrics, compute_depth_metrics
from model import build_model
from utils import (
    depth_to_meters,
    estimate_height,
    load_checkpoint,
    save_and_display_prediction,
    set_seed,
    show_prediction,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
USE_AMP_RUNTIME = USE_AMP and DEVICE.type == "cuda"


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.last_epoch = -1

    def step(self, epoch=None):
        if epoch is not None:
            self.last_epoch = epoch
        else:
            self.last_epoch += 1
        e = self.last_epoch
        if e < self.warmup_epochs:
            scale = (e + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (e - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base * scale

    def state_dict(self):
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state):
        self.last_epoch = state["last_epoch"]


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    scaler=None,
    device=DEVICE,
    epoch=0,
    writer=None,
    viz_dir=None,
):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    loss_parts = {"sil": 0.0, "berhu": 0.0, "grad": 0.0}
    metric_list = []
    grad_norm_sum = 0.0
    grad_steps = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for step, (rgb, depth, mask) in enumerate(tqdm(loader, leave=False)):
            rgb = rgb.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            if is_train and USE_AMP_RUNTIME:
                with autocast():
                    pred = model(rgb)
                    loss, parts = criterion(pred, depth, mask, image=rgb)
                    loss = loss / GRAD_ACCUM_STEPS
            else:
                pred = model(rgb)
                loss, parts = criterion(pred, depth, mask, image=rgb)
                if is_train:
                    loss = loss / GRAD_ACCUM_STEPS

            if not torch.isfinite(loss):
                print(f"WARNING: non-finite loss at step {step}, skipping batch")
                continue

            if is_train:
                if USE_AMP_RUNTIME:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    if USE_AMP_RUNTIME:
                        scaler.unscale_(optimizer)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    grad_norm_sum += float(gn)
                    grad_steps += 1
                    if USE_AMP_RUNTIME:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * (GRAD_ACCUM_STEPS if is_train else 1)
            for k in loss_parts:
                if k in parts:
                    loss_parts[k] += parts[k]

            if not is_train:
                pred_np = pred.detach().float().cpu().numpy()
                gt_np = depth.detach().float().cpu().numpy()
                mask_np = mask.detach().float().cpu().numpy()
                for i in range(pred_np.shape[0]):
                    m = compute_depth_metrics(
                        pred_np[i, 0] * MAX_DEPTH_M,
                        gt_np[i, 0] * MAX_DEPTH_M,
                        mask_np[i, 0],
                    )
                    if m:
                        metric_list.append(m)

    avg_loss = total_loss / max(len(loader), 1)
    avg_parts = {k: v / max(len(loader), 1) for k, v in loss_parts.items()}
    avg_metrics = aggregate_metrics(metric_list)
    avg_grad_norm = grad_norm_sum / max(grad_steps, 1)

    if writer and is_train:
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/grad_norm", avg_grad_norm, epoch)
        for k, v in avg_parts.items():
            writer.add_scalar(f"train/{k}", v, epoch)

    if writer and not is_train:
        writer.add_scalar("val/loss", avg_loss, epoch)
        for k, v in avg_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

    if not is_train and viz_dir is not None and len(loader.dataset) > 0:
        viz_path = Path(viz_dir) / f"epoch_{epoch + 1:03d}.png"
        rgb, depth, mask = loader.dataset[0]
        model.eval()
        with torch.no_grad():
            pred = model(rgb.unsqueeze(0).to(device))
        save_and_display_prediction(
            rgb, depth, pred.cpu()[0], mask, viz_path, display=False
        )

    return avg_loss, avg_metrics, avg_grad_norm


def build_loaders(device):
    train_dataset = NYUDepthDataset(
        TRAIN_CSV,
        image_size=IMAGE_SIZE,
        max_samples=MAX_TRAIN_SAMPLES,
        subsample_seed=TRAIN_SUBSAMPLE_SEED,
        augment=True,
        normalize=True,
    )
    test_dataset = NYUDepthDataset(
        TEST_CSV,
        image_size=IMAGE_SIZE,
        augment=False,
        normalize=True,
    )
    train_note = (
        f"{len(train_dataset)} (random subset)"
        if MAX_TRAIN_SAMPLES
        else f"{len(train_dataset)} (full)"
    )
    print(f"Train samples: {train_note}, Test samples: {len(test_dataset)}")

    pin = device.type == "cuda"
    persistent = PERSISTENT_WORKERS and NUM_WORKERS > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=persistent,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    return train_loader, test_loader, test_dataset


def _training_config_snapshot() -> dict:
    return {
        "MODEL_NAME": MODEL_NAME,
        "PRETRAINED": PRETRAINED,
        "MAX_TRAIN_SAMPLES": MAX_TRAIN_SAMPLES,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "IMAGE_SIZE": IMAGE_SIZE,
        "SEED": SEED,
    }


def main(resume: bool = False, run_name: str | None = None):
    set_seed(SEED)
    device = get_device()
    global DEVICE
    DEVICE = device

    run_name = run_name or RUN_NAME or None
    run = ExperimentRun(
        CHECKPOINT_DIR,
        run_name=run_name,
        resume=resume,
        config_snapshot=_training_config_snapshot(),
    )
    run.mark_active()

    log_dir = run.run_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"NYU data root: {DATA_ROOT}")
    print(f"Device: {device} | AMP: {USE_AMP_RUNTIME} | Model: {MODEL_NAME} (pretrained={PRETRAINED})")

    train_loader, test_loader, test_dataset = build_loaders(device)

    model = build_model(MODEL_NAME, pretrained=PRETRAINED).to(device)
    if USE_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = DepthLoss(
        sil_weight=LOSS_SIL_WEIGHT,
        berhu_weight=LOSS_BERHU_WEIGHT,
        grad_weight=LOSS_GRAD_WEIGHT,
        smooth_weight=LOSS_SMOOTH_WEIGHT,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, EPOCHS)
    scaler = GradScaler(enabled=USE_AMP_RUNTIME)

    start_epoch = 0
    best_rmse = float("inf")

    if resume and run.last_path.exists():
        start_epoch, best_rmse = load_checkpoint(
            run.last_path, model, optimizer, scheduler, scaler, device
        )
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}, best_rmse={best_rmse:.3f}")

    writer = SummaryWriter(log_dir=str(log_dir))

    for epoch in range(start_epoch, EPOCHS):
        scheduler.step(epoch)
        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("lr", lr, epoch)

        train_loss, _, grad_norm = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            writer=writer,
        )

        save_viz = (epoch + 1) % VIZ_EVERY == 0 or epoch == 0
        val_loss, val_metrics, _ = run_epoch(
            model,
            test_loader,
            criterion,
            device=device,
            epoch=epoch,
            writer=writer,
            viz_dir=run.viz_dir if save_viz else None,
        )

        run.record_epoch(epoch, train_loss, val_loss, val_metrics, lr, grad_norm)

        saved = ""
        rmse = val_metrics.get("rmse", float("inf"))
        if rmse < best_rmse:
            best_rmse = rmse
            run.save_best(epoch, model, optimizer, scheduler, best_rmse, scaler)
            saved = f" | saved best -> {run.best_path}"

        if SAVE_EVERY_EPOCH:
            epoch_path = run.save_epoch_checkpoint(
                epoch, model, optimizer, scheduler, best_rmse, scaler
            )
            saved += f" | epoch ckpt -> {epoch_path.name}"

        run.save_last(epoch, model, optimizer, scheduler, best_rmse, scaler)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | lr={lr:.2e} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"rmse={val_metrics.get('rmse', 0):.3f}m | "
            f"delta1={val_metrics.get('delta1', 0):.3f} | "
            f"grad={grad_norm:.2f}"
            f"{saved}"
        )
        print(f"  metrics plot -> {run.curves_path}")

    writer.close()

    if run.best_path.exists():
        load_checkpoint(run.best_path, model, device=device)
        print(f"Loaded best checkpoint (val_rmse={best_rmse:.3f}m)")

    model.eval()
    rgb, depth, mask = test_dataset[0]
    with torch.no_grad():
        pred = model(rgb.unsqueeze(0).to(device))

    eval_preview = run.run_dir / "eval_preview.png"
    save_and_display_prediction(
        rgb, depth, pred.cpu()[0], mask, eval_preview, display=False
    )
    print(f"Saved eval preview -> {eval_preview}")

    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            save_and_display_prediction(
                rgb, depth, pred.cpu()[0], mask, eval_preview, display=True
            )
        else:
            show_prediction(rgb, depth, pred.cpu()[0])
    except ImportError:
        show_prediction(rgb, depth, pred.cpu()[0])

    print("Estimated depth span (m):", estimate_height(depth_to_meters(pred.cpu()[0])))
    return run


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume latest run (last.pt)")
    parser.add_argument("--run-name", default="", help="e.g. run1 or my_experiment")
    args = parser.parse_args()
    main(resume=args.resume, run_name=args.run_name or None)
