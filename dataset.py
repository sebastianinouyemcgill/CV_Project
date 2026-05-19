import csv
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import DATA_ROOT, IMAGENET_MEAN, IMAGENET_STD, MAX_DEPTH_M


def _resolve_path(csv_path: str) -> str:
    rel = csv_path.strip()
    if rel.startswith("data/"):
        rel = rel[len("data/") :]
    return str(DATA_ROOT / rel)


def load_depth_meters(depth_path: str, max_depth_m: float = MAX_DEPTH_M) -> np.ndarray:
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_path}")

    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {depth.shape} from {depth_path}")

    if depth.dtype == np.uint16 or depth.max() > 255:
        depth_m = depth.astype(np.float32) / 1000.0
    else:
        depth_m = (depth.astype(np.float32) / 255.0) * max_depth_m

    return depth_m


class NYUDepthDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_size=256,
        max_depth_m=MAX_DEPTH_M,
        max_samples=None,
        subsample_seed=42,
        augment=False,
        normalize=True,
    ):
        self.image_size = image_size
        self.max_depth_m = max_depth_m
        self.augment = augment
        self.normalize = normalize
        all_samples = []

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                all_samples.append((_resolve_path(row[0]), _resolve_path(row[1])))

        if not all_samples:
            raise RuntimeError(f"No samples listed in {csv_path}")

        if max_samples is not None and len(all_samples) > max_samples:
            rng = random.Random(subsample_seed)
            self.samples = rng.sample(all_samples, max_samples)
        else:
            self.samples = all_samples

    def __len__(self):
        return len(self.samples)

    def _apply_augmentations(self, rgb, depth_m, mask):
        """Geometry-safe augmentations for depth estimation."""
        if random.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            depth_m = np.ascontiguousarray(depth_m[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        # Mild photometric jitter (does not affect depth geometry)
        if random.random() < 0.5:
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            rgb = np.clip((rgb - 0.5) * contrast + 0.5, 0, 1)
            rgb = np.clip(rgb * brightness, 0, 1)

        if random.random() < 0.3:
            gamma = random.uniform(0.85, 1.15)
            rgb = np.clip(rgb ** gamma, 0, 1)

        return rgb, depth_m, mask

    def __getitem__(self, idx):
        rgb_path, depth_path = self.samples[idx]

        rgb = cv2.imread(rgb_path)
        if rgb is None:
            raise FileNotFoundError(f"Could not read RGB image: {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth_m = load_depth_meters(depth_path, self.max_depth_m)
        mask = (depth_m > 0).astype(np.float32)

        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        depth_m = cv2.resize(depth_m, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        rgb = rgb.astype(np.float32) / 255.0

        if self.augment:
            rgb, depth_m, mask = self._apply_augmentations(rgb, depth_m, mask)

        depth_norm = np.clip(depth_m / self.max_depth_m, 0.0, 1.0)

        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
        if self.normalize:
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            rgb = (rgb - mean) / std

        depth = torch.from_numpy(depth_norm).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return rgb, depth, mask
