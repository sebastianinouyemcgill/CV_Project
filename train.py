import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import DepthDataset
from model import UNet
from utils import show_prediction, estimate_height

from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = DepthDataset(
    rgb_dir="data/rgb",
    depth_dir="data/depth"
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet().to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0

    for rgb, depth in tqdm(loader):
        rgb = rgb.to(DEVICE)
        depth = depth.to(DEVICE)

        pred = model(rgb)

        loss = criterion(pred, depth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss / len(loader)}")

# visualize one sample

model.eval()

rgb, depth = dataset[0]

with torch.no_grad():
    pred = model(rgb.unsqueeze(0).to(DEVICE))

show_prediction(
    rgb,
    depth,
    pred.cpu()[0]
)

height = estimate_height(pred.cpu()[0].squeeze().numpy())

print("Estimated Height:", height)