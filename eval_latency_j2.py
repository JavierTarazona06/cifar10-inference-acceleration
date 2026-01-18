"""
J2-09: Latency measurement for MobileNetV3-Small trained model.
Measures GPU latency (mean and P95) on CIFAR-10 test set.
"""

import os
import torch
from cifaracce.data import test_loader
from cifaracce.models import MobileNetV3Small
from cifaracce.bench import benchmark_latency
from cifaracce.utils.seed import set_seed

# Set seed for reproducibility
set_seed(seed=42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load trained model
model = MobileNetV3Small(
    num_classes=10,
    learning_rate=0.1,
    momentum=0.9,
    weight_decay=5e-4
)

checkpoint_path = "checkpoints/mobilenetv3/mobilenetv3_best.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✓ Loaded checkpoint from {checkpoint_path}")
else:
    print(f"✗ Checkpoint not found at {checkpoint_path}")
    print("   Please run 'python train_mobilenet_j2.py' first.")
    exit(1)

model.model.to(device)
model.model.eval()

# Measure latency
print("\nMeasuring GPU latency on CIFAR-10 test set...")
print("(warm-up + measurement iterations)")

stats = benchmark_latency(
    model.model,
    test_loader,
    device=device,
    warmup_iters=100,
    measure_iters=100
)

# Print results
print("\n" + "="*50)
print("J2-09 LATENCY RESULTS (MobileNetV3-Small FP32)")
print("="*50)
print(f"Mean Latency:  {stats['mean']:.4f} ms")
print(f"P95 Latency:   {stats['p95']:.4f} ms")
print(f"Std Dev:       {stats['std']:.4f} ms")
print(f"Min Latency:   {stats['min']:.4f} ms")
print(f"Max Latency:   {stats['max']:.4f} ms")
print("="*50)
