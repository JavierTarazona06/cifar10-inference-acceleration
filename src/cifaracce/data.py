import csv
import numpy as np

import torch
import torchvision
from torchvision import datasets, transforms

from time import perf_counter
from datetime import datetime
from pathlib import Path

# ------------------------------------------- #
# CIFAR-10 Preparation and Transformations
# ------------------------------------------- #

BATCH_SIZE_TRAIN = 128  # For trainning
BATCH_SIZE_TEST = 1     # For latence benchmark

"""
For trainning, transformation is augmentations for data variability and normalisation for the model
"""
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(), # Turn images 50% of time
    transforms.RandomCrop(32, padding=4), # Extend area by 4 pixels -> 40*40 -> Variability without loosing information
    transforms.RandomRotation(15), # 15, general rule to keep it consistently with real world
    transforms.ToTensor(), # Image to tensor for PyTorch and data from uint8 to float32
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    ) # Values per channel, info calculated from CIFAR-10 dataset
])

"""
For testing, transformation is just normalisation for the model
"""
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])


# Load CIFAR-10 (train)
cifar10_train = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform_train
)

# Load CIFAR-10 (test)
cifar10_test = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform_test
)


train_loader = torch.utils.data.DataLoader(
    cifar10_train,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True, # Mix data, not overfitting
    num_workers=4,
    pin_memory=True # CPU -> GPU
)

test_loader = torch.utils.data.DataLoader(
    cifar10_test,
    batch_size=BATCH_SIZE_TEST,
    shuffle=False, # Don't mix, fair comparisons
    num_workers=0, # No parallelism, just one pass through without overhead
    pin_memory=True
)


# Verification
print(f"- Train dataset: {len(cifar10_train)} images")
print(f"- Test dataset: {len(cifar10_test)} images")
print(f"- Nombre de classes: {len(cifar10_train.classes)}")
print(f"- Classes: {cifar10_train.classes}")

# Verify one image
sample_img, sample_label = cifar10_train[0]
print(f"Shape image train: {sample_img.shape}")
print(f"Label sample: {sample_label} ({cifar10_train.classes[sample_label]})")