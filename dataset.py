import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset


class DepthDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, image_size=256):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.image_size = image_size

        self.files = sorted(os.listdir(rgb_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]

        rgb_path = os.path.join(self.rgb_dir, file_name)
        depth_path = os.path.join(self.depth_dir, file_name)

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

        rgb = cv2.resize(rgb, (self.image_size, self.image_size))
        depth = cv2.resize(depth, (self.image_size, self.image_size))

        rgb = rgb.astype(np.float32) / 255.0
        depth = depth.astype(np.float32) / 255.0

        rgb = torch.tensor(rgb).permute(2, 0, 1)
        depth = torch.tensor(depth).unsqueeze(0)

        return rgb, depth