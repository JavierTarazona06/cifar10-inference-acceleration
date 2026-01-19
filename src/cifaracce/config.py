import torch
from pathlib import Path

# Global configuration for training, benchmarking, and paths.

# -------------------------------
# Environment
# -------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

# -------------------------------
# Data
# -------------------------------
# Root folder for torchvision datasets (CIFAR-10 will be stored here)
DATA_ROOT = Path('data')

# -------------------------------
# Training (MobileNetV3 J2)
# -------------------------------
TRAIN_EPOCHS = 60
EARLY_STOP_EPOCH = 40
EARLY_STOP_ACC = 85.0  # percent

OPTIMIZER = {
    'name': 'SGD',
    'lr': 0.12,
    'momentum': 0.9,
    'weight_decay': 5e-4,
}

SCHEDULER = {
    'name': 'CosineAnnealingLR',
    'T_max': TRAIN_EPOCHS,
}

# -------------------------------
# Training (ResNet-18 Teacher J3)
# -------------------------------
RESNET_TRAIN = {
    'epochs': 200,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'scheduler': 'CosineAnnealingLR',
    'scheduler_T_max': 200,
    'target_acc': 85.0,  # percent
    'early_stop_floor': 120,
}

# -------------------------------
# Checkpoints & Logs
# -------------------------------
CHECKPOINTS = {
    'root': Path('checkpoints'),
    'mobilenet_dir': Path('checkpoints/mobilenetv3'),
    'distill_dir': Path('checkpoints/distill'),
    'resnet18_dir': Path('checkpoints/resnet18'),
}

LOGS = {
    'training_mobilenet_csv': Path('results_j2_mobilenet_training.csv'),
    'training_resnet_csv': Path('results_resnet18_training.csv'),
}

# -------------------------------
# Benchmark Defaults
# -------------------------------
WARM_UP_ITERS = 50
MEASURE_ITERS = 500

BENCHMARK = {
    'channels_last': False,
    'use_amp': False,
    'amp_dtype': 'float16',  # options: 'float16', 'bfloat16'
}

# Optional per-model benchmark overrides
RESNET_LATENCY = {
    'warmup_iters': 100,
    'measure_iters': 1000,
}
