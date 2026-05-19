"""Experiment runs: checkpoints/run1/, run2/, ... with per-epoch saves and metric plots."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    lr: float
    grad_norm: float
    metrics: dict[str, float] = field(default_factory=dict)


class ExperimentRun:
    """
    Manages one training run under CHECKPOINT_DIR/runN/ (or a custom RUN_NAME).

    Layout:
      run1/
        config.json
        metrics_history.json
        metrics_curves.png
        best.pt
        last.pt
        eval_preview.png
        epochs/
          epoch_001.pt
          epoch_002.pt
        visualizations/
          epoch_001.png
    """

    def __init__(
        self,
        base_dir: Path,
        run_name: str | None = None,
        resume: bool = False,
        config_snapshot: dict[str, Any] | None = None,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if run_name:
            self.run_dir = self.base_dir / run_name
        elif resume:
            self.run_dir = self._resolve_resume_dir()
        else:
            self.run_dir = self._next_run_dir()

        self.epochs_dir = self.run_dir / "epochs"
        self.viz_dir = self.run_dir / "visualizations"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.epochs_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        self.history_path = self.run_dir / "metrics_history.json"
        self.curves_path = self.run_dir / "metrics_curves.png"
        self.best_path = self.run_dir / "best.pt"
        self.last_path = self.run_dir / "last.pt"
        self.config_path = self.run_dir / "config.json"

        self.history: list[EpochRecord] = self._load_history()

        if config_snapshot is not None:
            self.save_config(config_snapshot)

        print(f"Experiment run directory: {self.run_dir}")

    def _next_run_dir(self) -> Path:
        pattern = re.compile(r"^run(\d+)$")
        nums = []
        for p in self.base_dir.iterdir():
            if p.is_dir() and (m := pattern.match(p.name)):
                nums.append(int(m.group(1)))
        n = max(nums, default=0) + 1
        return self.base_dir / f"run{n}"

    def _resolve_resume_dir(self) -> Path:
        import os

        explicit = os.environ.get("RUN_NAME", "").strip()
        if explicit:
            return self.base_dir / explicit
        last_file = self.base_dir / ".last_run"
        if last_file.exists():
            name = last_file.read_text().strip()
            if (self.base_dir / name).exists():
                return self.base_dir / name
        candidates = sorted(
            [p for p in self.base_dir.glob("run*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
        return self._next_run_dir()

    def _load_history(self) -> list[EpochRecord]:
        if not self.history_path.exists():
            return []
        data = json.loads(self.history_path.read_text())
        return [EpochRecord(**row) for row in data]

    def save_config(self, config: dict[str, Any]):
        self.config_path.write_text(json.dumps(config, indent=2, default=str))

    def mark_active(self):
        (self.base_dir / ".last_run").write_text(self.run_dir.name)

    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: dict[str, float],
        lr: float,
        grad_norm: float,
    ):
        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=lr,
            grad_norm=grad_norm,
            metrics=val_metrics,
        )
        # Replace if re-logging same epoch
        self.history = [h for h in self.history if h.epoch != epoch]
        self.history.append(record)
        self.history.sort(key=lambda h: h.epoch)
        self._save_history()
        self.plot_metrics()

    def _save_history(self):
        payload = []
        for h in self.history:
            row = asdict(h)
            payload.append(row)
        self.history_path.write_text(json.dumps(payload, indent=2))

    def plot_metrics(self):
        if not self.history:
            return

        epochs = [h.epoch + 1 for h in self.history]
        train_loss = [h.train_loss for h in self.history]
        val_loss = [h.val_loss for h in self.history]
        rmse = [h.metrics.get("rmse", 0) for h in self.history]
        delta1 = [h.metrics.get("delta1", 0) for h in self.history]
        abs_rel = [h.metrics.get("abs_rel", 0) for h in self.history]
        lr = [h.lr for h in self.history]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        axes[0, 0].plot(epochs, train_loss, label="train", marker="o", ms=3)
        axes[0, 0].plot(epochs, val_loss, label="val", marker="o", ms=3)
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, rmse, color="crimson", marker="o", ms=3)
        axes[0, 1].set_title("Val RMSE (m)")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(epochs, delta1, color="green", marker="o", ms=3)
        axes[1, 0].set_title("Val δ₁")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(epochs, abs_rel, color="purple", marker="o", ms=3, label="abs_rel")
        ax2 = axes[1, 1].twinx()
        ax2.plot(epochs, lr, color="gray", linestyle="--", alpha=0.7, label="lr")
        axes[1, 1].set_title("Val AbsRel & LR")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(f"Training metrics — {self.run_dir.name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(self.curves_path, dpi=130, bbox_inches="tight")
        plt.close()

    def save_epoch_checkpoint(
        self,
        epoch: int,
        model,
        optimizer,
        scheduler,
        best_rmse: float,
        scaler=None,
    ):
        from utils import save_checkpoint

        path = self.epochs_dir / f"epoch_{epoch + 1:03d}.pt"
        save_checkpoint(path, epoch, model, optimizer, scheduler, best_rmse, scaler)
        return path

    def save_best(self, epoch, model, optimizer, scheduler, best_rmse, scaler=None):
        from utils import save_checkpoint

        save_checkpoint(self.best_path, epoch, model, optimizer, scheduler, best_rmse, scaler)

    def save_last(self, epoch, model, optimizer, scheduler, best_rmse, scaler=None):
        from utils import save_checkpoint

        save_checkpoint(self.last_path, epoch, model, optimizer, scheduler, best_rmse, scaler)
