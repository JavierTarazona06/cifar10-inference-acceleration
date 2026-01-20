"""
J4-02 | Mesurer la latence FP16 (modèle léger)

Benchmark de latence MobileNetV3-Small sur CIFAR-10 (batch=1).
- Compare FP32 vs FP16 (autocast)
- Warm-up + mesure (moyenne + p95)

Usage:
    python eval_latency_mobilenetv3.py [--num-warmup 100] [--num-runs 1000]
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cifaracce.data import test_loader
from cifaracce.models import MobileNetV3Small
from cifaracce.utils.seed import set_seed


def _forward_with_precision(model, inputs, device: str, precision: str = "fp32"):
    if device == "cuda" and precision == "fp16":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            return model(inputs)
    return model(inputs)


def benchmark_latency(model, loader, device: str, precision: str,
                      num_warmup: int = 100, num_runs: int = 1000):
    model.eval()
    latencies_ms = []

    # Warm-up
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= num_warmup:
                break
            x = x.to(device, non_blocking=True)
            if device == "cuda":
                torch.cuda.synchronize()
            _ = _forward_with_precision(model, x, device, precision)
            if device == "cuda":
                torch.cuda.synchronize()

    # Timed runs
    with torch.no_grad():
        runs = 0
        for _, (x, _) in enumerate(loader):
            if runs >= num_runs:
                break
            x = x.to(device, non_blocking=True)

            if device == "cuda":
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = _forward_with_precision(model, x, device, precision)
                end.record()
                torch.cuda.synchronize()
                latencies_ms.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = _forward_with_precision(model, x, device, precision)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)
            runs += 1

    return np.asarray(latencies_ms, dtype=np.float64)


def summarize(latencies_ms: np.ndarray, title: str):
    mean = float(np.mean(latencies_ms))
    std = float(np.std(latencies_ms))
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    lmin = float(np.min(latencies_ms))
    lmax = float(np.max(latencies_ms))

    rows = [
        ["Mean (ms)", f"{mean:.4f}"],
        ["Std (ms)", f"{std:.4f}"],
        ["Min (ms)", f"{lmin:.4f}"],
        ["Max (ms)", f"{lmax:.4f}"],
        ["p50 (ms)", f"{p50:.4f}"],
        ["p95 (ms)", f"{p95:.4f}"],
        ["p99 (ms)", f"{p99:.4f}"],
    ]

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="grid"))
    print("=" * 60)

    return {"mean": mean, "std": std, "p95": p95, "p99": p99}


def load_student(device: str):
    model = MobileNetV3Small(num_classes=10, device=device).to(device)

    ckpt_dir = Path("checkpoints/mobilenetv3")
    ckpt_path = ckpt_dir / "mobilenetv3_best.pt"
    if ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])  # for completeness
        else:
            model.load_state_dict(state)
        print(" Model weights loaded")
    else:
        print(" No checkpoint found, using random-initialized weights (latency only)")

    return model


def main():
    parser = argparse.ArgumentParser(description="J4-02 FP16 latency benchmark for MobileNetV3-Small")
    parser.add_argument("--num-warmup", type=int, default=100)
    parser.add_argument("--num-runs", type=int, default=1000)
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    if device == "cpu":
        print(" Warning: GPU not available, FP16 will fall back to FP32 on CPU")

    model = load_student(device)

    print("\n== MobileNetV3-Small | FP32 benchmark ==")
    lat_fp32 = benchmark_latency(
        model, test_loader, device, precision="fp32",
        num_warmup=args.num_warmup, num_runs=args.num_runs,
    )
    stats_fp32 = summarize(lat_fp32, "LATENCY STATISTICS (batch=1, FP32)")

    print("\n== MobileNetV3-Small | FP16 benchmark ==")
    lat_fp16 = benchmark_latency(
        model, test_loader, device, precision="fp16",
        num_warmup=args.num_warmup, num_runs=args.num_runs,
    )
    stats_fp16 = summarize(lat_fp16, "LATENCY STATISTICS (batch=1, FP16)")

    # Headline comparison
    mean32, mean16 = stats_fp32["mean"], stats_fp16["mean"]
    p95_32, p95_16 = stats_fp32["p95"], stats_fp16["p95"]
    speedup_mean = mean32 / mean16 if mean16 > 0 else float("inf")
    speedup_p95 = p95_32 / p95_16 if p95_16 > 0 else float("inf")

    print("\n📊 Summary (MobileNetV3-Small, batch=1):")
    print(f"  FP32 mean: {mean32:.4f} ms | p95: {p95_32:.4f} ms")
    print(f"  FP16 mean: {mean16:.4f} ms | p95: {p95_16:.4f} ms")
    print(f"  Speedup (mean): ×{speedup_mean:.2f}")
    print(f"  Speedup (p95):  ×{speedup_p95:.2f}")


if __name__ == "__main__":
    main()
