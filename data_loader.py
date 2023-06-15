import torch
import os
import config
import torchvision.transforms as transforms
from torchvision import datasets
data_folder = os.path.exists("data")
if not data_folder:
    os.makedirs("data/", exist_ok=True)
    data_folder = True

transform=transforms.Compose(
    [
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])
    ]
)
print(f"data_folder: {data_folder}")

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/",
        train=True,
        download=data_folder,
        transform=transform,
    ),
    batch_size=config.BATCH_SIZE,
    shuffle=True,
)

