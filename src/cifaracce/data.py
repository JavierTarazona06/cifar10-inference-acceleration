import csv
import numpy as np

import torch
import torchvision
from torchvision import datasets, transforms
import io
from contextlib import redirect_stdout

from time import perf_counter
from datetime import datetime
from pathlib import Path

# ========================================================================
# DATA PREPARATION MODULE
# ========================================================================
# 
# What does this file do?
# Downloads and prepares CIFAR-10 images for model usage.
#
# Step by step:
# 1. Downloads CIFAR-10 dataset (60,000 images from 10 classes: 
#    airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)
# 2. Transforms images:
#    - For training: flips, rotates, crops (data augmentation)
#    - For testing: only normalizes values
# 3. Creates DataLoaders that deliver images in batches:
#    - Train: 128 images at a time
#    - Test: 1 image at a time (to measure individual latency)
#
# Result: You have train_loader and test_loader ready to use.
# ========================================================================

# ------------------------------------------- #
# CIFAR-10 Preparation and Transformations
# ------------------------------------------- #

BATCH_SIZE_TRAIN = 128  # For trainning
BATCH_SIZE_TEST = 1     # For latence benchmark

"""
For training: stronger but standard CIFAR-10 pipeline (crop + flip + normalize).
"""
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])

"""
For testing: only normalization (no augmentation).
"""
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])


# Load CIFAR-10 (train/test) with stdout suppressed (avoids torchvision messages on import)
with redirect_stdout(io.StringIO()):
    cifar10_train = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )

with redirect_stdout(io.StringIO()):
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


if __name__ == "__main__":
    # Verification (only when running this file directly)
    print(f"- Train dataset: {len(cifar10_train)} images")
    print(f"- Test dataset: {len(cifar10_test)} images")
    print(f"- Nombre de classes: {len(cifar10_train.classes)}")
    print(f"- Classes: {cifar10_train.classes}")

    sample_img, sample_label = cifar10_train[0]
    print(f"Shape image train: {sample_img.shape}")
    print(f"Label sample: {sample_label} ({cifar10_train.classes[sample_label]})")