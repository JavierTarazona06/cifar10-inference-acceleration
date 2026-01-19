"""
J3-11 | Documenter la taille du modèle

Calcular:
- Número de parámetros totales (trainable + non-trainable)
- Tamaño del checkpoint (MB)
- Taille estimée en memoria (MB)

Usage:
    python model_size_resnet18.py
"""

import sys
import torch
from pathlib import Path
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cifaracce.models.resnet18 import ResNet18
from cifaracce.utils.seed import set_seed


def count_parameters(model):
    """Count total, trainable, and non-trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return total_params, trainable_params, non_trainable_params


def estimate_memory_usage(model):
    """Estimate memory usage in MB (assumes FP32, 4 bytes per parameter)."""
    total_params, _, _ = count_parameters(model)
    
    # FP32: 4 bytes per parameter
    # Also account for optimizer state (2x for SGD with momentum)
    param_memory_mb = (total_params * 4) / (1024 ** 2)
    
    # During training: model weights + gradients + optimizer state
    # Rough estimate: 3x the parameter memory (weights + grads + optimizer)
    training_memory_mb = param_memory_mb * 3
    
    # During inference: just model weights
    inference_memory_mb = param_memory_mb
    
    return param_memory_mb, training_memory_mb, inference_memory_mb


def get_checkpoint_size(checkpoint_path):
    """Get checkpoint file size in MB."""
    if not checkpoint_path.exists():
        return None
    
    size_bytes = checkpoint_path.stat().st_size
    size_mb = size_bytes / (1024 ** 2)
    return size_mb


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("J3-11 | Model Size Documentation - ResNet-18")
    print("=" * 60 + "\n")
    
    # Initialize model
    model = ResNet18(num_classes=10).to(device)
    
    # Count parameters
    total_params, trainable_params, non_trainable_params = count_parameters(model)
    
    # Estimate memory
    param_memory_mb, training_memory_mb, inference_memory_mb = estimate_memory_usage(model)
    
    # Get checkpoint size
    checkpoint_path = Path("checkpoints/resnet18/resnet18_cifar_best.pth")
    if not checkpoint_path.exists():
        checkpoint_path = Path("checkpoints/resnet18/resnet18_best.pt")
    
    checkpoint_size_mb = get_checkpoint_size(checkpoint_path)
    
    # Create results tables
    print("\n1. PARAMETERS")
    print("-" * 60)
    param_results = [
        ["Total parameters", f"{total_params:,}"],
        ["Trainable parameters", f"{trainable_params:,}"],
        ["Non-trainable parameters", f"{non_trainable_params:,}"],
        ["Total (in millions)", f"{total_params / 1e6:.2f}M"],
    ]
    print(tabulate(param_results, headers=["Metric", "Value"], tablefmt="grid"))
    
    print("\n2. MEMORY USAGE (FP32)")
    print("-" * 60)
    memory_results = [
        ["Parameter memory", f"{param_memory_mb:.2f} MB"],
        ["Memory during training*", f"{training_memory_mb:.2f} MB"],
        ["Memory during inference", f"{inference_memory_mb:.2f} MB"],
    ]
    print(tabulate(memory_results, headers=["Scenario", "Size"], tablefmt="grid"))
    print("* Includes model weights, gradients, and optimizer state (SGD)")
    
    print("\n3. CHECKPOINT SIZE")
    print("-" * 60)
    if checkpoint_size_mb is not None:
        checkpoint_results = [
            ["Checkpoint path", str(checkpoint_path)],
            ["Checkpoint size", f"{checkpoint_size_mb:.2f} MB"],
        ]
        print(tabulate(checkpoint_results, headers=["Metric", "Value"], tablefmt="grid"))
    else:
        print("⚠ Checkpoint not found yet")
    
    print("\n4. SUMMARY")
    print("-" * 60)
    print(f"Model: ResNet-18 adapted for CIFAR-10")
    print(f"  - Architecture: 3×3 conv (no maxpool), 10 output classes")
    print(f"  - Parameters: {total_params / 1e6:.2f}M")
    print(f"  - Inference memory: {inference_memory_mb:.2f} MB (FP32)")
    if checkpoint_size_mb is not None:
        print(f"  - Checkpoint size: {checkpoint_size_mb:.2f} MB")
        compression_ratio = (total_params * 4) / (1024**2) / checkpoint_size_mb
        print(f"  - Checkpoint/Full params ratio: {compression_ratio:.2f}x")
    
    print("\n" + "=" * 60)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_memory_mb": param_memory_mb,
        "checkpoint_size_mb": checkpoint_size_mb,
    }


if __name__ == "__main__":
    main()
