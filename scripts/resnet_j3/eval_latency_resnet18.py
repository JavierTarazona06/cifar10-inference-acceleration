"""
J3-10 | Mesurer la latence GPU (FP32)

Benchmark de latencia del ResNet-18 sobre CIFAR-10 test set.
- Batch size = 1
- Warm-up + mesure (moyenne + p95)
- Entrada sur GPU

Usage:
    python benchmark_latency_resnet18.py
"""

import sys
import torch
import numpy as np
from pathlib import Path
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cifaracce.data import test_loader
from cifaracce.models.resnet18 import ResNet18
from cifaracce.utils.seed import set_seed
from cifaracce import config as cfg


def benchmark_latency(model, test_loader, device, num_warmup=100, num_runs=1000):
    """Benchmark latency of model on test set with batch_size=1."""
    
    model.eval()
    
    latencies = []
    
    print(f"Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if batch_idx >= num_warmup:
                break
            
            inputs = inputs.to(device)
            
            # Synchronize GPU
            if device == "cuda":
                torch.cuda.synchronize()
            
            outputs = model(inputs)
            
            if device == "cuda":
                torch.cuda.synchronize()
    
    print(f"Measuring latency ({num_runs} runs)...")
    with torch.no_grad():
        run_count = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if run_count >= num_runs:
                break
            
            inputs = inputs.to(device)
            
            # Synchronize and measure
            if device == "cuda":
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                outputs = model(inputs)
                end.record()
                
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))  # milliseconds
            else:
                import time
                start = time.perf_counter()
                outputs = model(inputs)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # convert to ms
            
            run_count += 1
    
    return np.array(latencies)


def main():
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    print(f"Using device: {device}\n")
    
    if device == "cpu":
        print(" Warning: GPU not available, using CPU (latency measurements will be slower)")
    
    # Load checkpoint
    checkpoint_path = cfg.CHECKPOINTS['resnet18_dir'] / "resnet18_cifar_best.pth"
    if not checkpoint_path.exists():
        checkpoint_path = cfg.CHECKPOINTS['resnet18_dir'] / "resnet18_best.pt"
    
    if not checkpoint_path.exists():
        print(f" Checkpoint not found")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = ResNet18(num_classes=10).to(device)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    print(" Model loaded\n")
    
    print("=" * 60)
    print("J3-10 | Benchmark Latence GPU ResNet-18 (FP32)")
    print("=" * 60 + "\n")
    
    # Run benchmark
    warmup = cfg.RESNET_LATENCY.get('warmup_iters', 100)
    runs = cfg.RESNET_LATENCY.get('measure_iters', 1000)
    latencies = benchmark_latency(model, test_loader, device, num_warmup=warmup, num_runs=runs)
    
    # Calculate statistics (in milliseconds)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    results = [
        ["Mean (ms)", f"{mean_latency:.4f}"],
        ["Std (ms)", f"{std_latency:.4f}"],
        ["Min (ms)", f"{min_latency:.4f}"],
        ["Max (ms)", f"{max_latency:.4f}"],
        ["p50 (ms)", f"{p50_latency:.4f}"],
        ["p95 (ms)", f"{p95_latency:.4f}"],
        ["p99 (ms)", f"{p99_latency:.4f}"],
    ]
    
    print("\n" + "=" * 60)
    print("LATENCY STATISTICS (batch_size=1, FP32)")
    print("=" * 60)
    print(tabulate(results, headers=["Metric", "Value"], tablefmt="grid"))
    print("=" * 60)
    
    print(f"\n📊 Summary:")
    print(f"  Device: {device}")
    print(f"  Batch size: 1")
    print(f"  Precision: FP32")
    print(f"  Number of runs: {len(latencies)}")
    print(f"  Mean latency: {mean_latency:.4f} ms")
    print(f"  p95 latency: {p95_latency:.4f} ms")
    
    return {
        "mean": mean_latency,
        "std": std_latency,
        "p95": p95_latency,
        "p99": p99_latency,
    }


if __name__ == "__main__":
    main()
