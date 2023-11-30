# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
import torchvision
from torch import nn

from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.models.dinov2.vision_transformer import vit_small_patch14_224_dinov2

from lightly.models.simclr import SimCLR

backbone = vit_small_patch14_224_dinov2(pretrained_path="pretrained/dinov2_vits14_pretrain.pth")
model = SimCLR(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

transform = SimCLRTransform(input_size=224, gaussian_blur=0.0)
# dataset = torchvision.datasets.CIFAR10(
#     "datasets/cifar10", download=True, transform=transform
# )
# or create a dataset from a folder containing images or videos:
dataset = LightlyDataset("dataset/Market-1501-v15.09.15/bounding_box_train",
                         "dataset/market1501_to_duke/Market-1501-v15.09.15/bounding_box_train",
                         transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = NTXentLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for batch in dataloader:
        x0, x1 = batch[0]
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")