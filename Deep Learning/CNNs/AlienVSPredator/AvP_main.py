import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

# ---------------
# data generators
# ---------------
normalize = transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'validation': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])
}

image_datasets = {
    'train': datasets.ImageFolder('data/train', transform=data_transforms['train']),
    'validation': datasets.ImageFolder('data/validation', transform=data_transforms['validation'])
}

data_loaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'],
                                         batch_size=32,
                                         shuffle=True,
                                         num_workers=4),
    'validation': torch.utils.data.DataLoader(image_datasets['validation'],
                                         batch_size=32,
                                         shuffle=True,
                                         num_workers=4)
}

# ------------------
# create the network
# ------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet50(pretrained=True).to(device)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(2048,128),
    nn.ReLU(inplace=True),
    nn.Linear(128,2)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.fc.parameters())

if __name__ == '__main__':
    pass