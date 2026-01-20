"""
J4-04 | Mesurer la latence FP16 (ResNet-18) + vérifier l'accuracy FP16

Benchmark ResNet-18 (teacher) sur CIFAR-10 batch=1 et compare l'accuracy FP32/FP16.
- Latence: FP32 vs FP16 (autocast), warm-up + mesure, stats (mean, p95, ...)
- Accuracy: FP32 vs FP16, delta et verdict vs seuil (default 0.5 pp)

Usage principal (tout):
    python scripts/fp16/eval_resnet18_fp16.py --num-warmup 100 --num-runs 1000

Options:
    --threshold 0.5    # seuil de chute d'accuracy tolérée (en points de pourcentage)
    --limit N          # limite d'échantillons test pour un smoke test rapide
    --no-latency       # saute le benchmark de latence
    --no-accuracy      # saute la vérification d'accuracy

Exemples rapides:
    # Smoke latence rapide
    python scripts/fp16/eval_resnet18_fp16.py --num-warmup 5 --num-runs 20 --no-accuracy
    # Smoke accuracy sur 200 échantillons
    python scripts/fp16/eval_resnet18_fp16.py --limit 200 --no-latency
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
from cifaracce.models.resnet18 import ResNet18
from cifaracce.utils.seed import set_seed


def _forward(model, x, device: str, precision: str):
    if device == "cuda" and precision == "fp16":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            return model(x)
    return model(x)


def benchmark_latency(model, loader, device: str, precision: str,
                      num_warmup: int = 100, num_runs: int = 1000):
    model.eval()
    latencies = []

    # Warm-up
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= num_warmup:
                break
            x = x.to(device, non_blocking=True)
            if device == "cuda":
                torch.cuda.synchronize()
            _ = _forward(model, x, device, precision)
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
                _ = _forward(model, x, device, precision)
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = _forward(model, x, device, precision)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)
            runs += 1

    return np.asarray(latencies, dtype=np.float64)


def summarize(lat_ms: np.ndarray, title: str):
    mean = float(np.mean(lat_ms))
    std = float(np.std(lat_ms))
    lmin = float(np.min(lat_ms))
    lmax = float(np.max(lat_ms))
    p50 = float(np.percentile(lat_ms, 50))
    p95 = float(np.percentile(lat_ms, 95))
    p99 = float(np.percentile(lat_ms, 99))

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


def load_teacher(device: str):
    model = ResNet18(num_classes=10).to(device)

    ckpt_candidates = [
        Path("checkpoints/resnet18/resnet18_cifar_best.pth"),
        Path("checkpoints/resnet18/resnet18_best.pt"),
    ]
    ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)

    if ckpt_path is None:
        print(" No checkpoint found, using random weights (latency only)")
        return model

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    print(" Teacher weights loaded")
    return model


def evaluate_accuracy(model, device: str, precision: str = "fp32", limit: int | None = None) -> float:
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        seen = 0
        for inputs, targets in test_loader:
            if limit is not None and seen >= limit:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            if device == "cuda" and precision == "fp16":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            seen += targets.size(0)

    return 100.0 * correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="J4-04 FP16 latency benchmark for ResNet-18")
    parser.add_argument("--num-warmup", type=int, default=100)
    parser.add_argument("--num-runs", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.5, help="Max allowed accuracy drop (pp)")
    parser.add_argument("--limit", type=int, default=None, help="Optional test sample limit for quick accuracy check")
    parser.add_argument("--no-latency", action="store_true", help="Skip latency benchmark")
    parser.add_argument("--no-accuracy", action="store_true", help="Skip accuracy comparison")
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    if device == "cpu":
        print(" GPU not available; FP16 will behave like FP32 on CPU")

    model = load_teacher(device)
    # Latency
    if not args.no_latency:
        print("\n== ResNet-18 | FP32 benchmark ==")
        lat_fp32 = benchmark_latency(
            model, test_loader, device, precision="fp32",
            num_warmup=args.num_warmup, num_runs=args.num_runs,
        )
        stats_fp32 = summarize(lat_fp32, "LATENCY STATISTICS (batch=1, FP32)")

        print("\n== ResNet-18 | FP16 benchmark ==")
        lat_fp16 = benchmark_latency(
            model, test_loader, device, precision="fp16",
            num_warmup=args.num_warmup, num_runs=args.num_runs,
        )
        stats_fp16 = summarize(lat_fp16, "LATENCY STATISTICS (batch=1, FP16)")

        mean32, mean16 = stats_fp32["mean"], stats_fp16["mean"]
        p95_32, p95_16 = stats_fp32["p95"], stats_fp16["p95"]
        speedup_mean = mean32 / mean16 if mean16 > 0 else float("inf")
        speedup_p95 = p95_32 / p95_16 if p95_16 > 0 else float("inf")

        print("\n📊 Summary (ResNet-18, batch=1):")
        print(f"  FP32 mean: {mean32:.4f} ms | p95: {p95_32:.4f} ms")
        print(f"  FP16 mean: {mean16:.4f} ms | p95: {p95_16:.4f} ms")
        print(f"  Speedup (mean): ×{speedup_mean:.2f}")
        print(f"  Speedup (p95):  ×{speedup_p95:.2f}")

    # Accuracy
    if not args.no_accuracy:
        acc_fp32 = evaluate_accuracy(model, device, precision="fp32", limit=args.limit)
        acc_fp16 = evaluate_accuracy(model, device, precision="fp16", limit=args.limit)
        drop = acc_fp32 - acc_fp16
        verdict = "OK" if drop <= args.threshold else "FAIL"
        print("\n== Accuracy Comparison (ResNet-18) ==")
        print(f"FP32 accuracy: {acc_fp32:.2f}%")
        print(f"FP16 accuracy: {acc_fp16:.2f}%")
        print(f"Δ (FP32 - FP16): {drop:.2f} pp")
        print(f"Verdict (threshold {args.threshold:.2f} pp): {verdict}")


if __name__ == "__main__":
    main()
