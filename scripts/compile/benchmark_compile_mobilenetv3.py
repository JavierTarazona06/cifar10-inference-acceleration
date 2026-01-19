"""
J4-05/06/07 | torch.compile sur MobileNetV3-Small (batch=1)

Mesure latence avant/après torch.compile et temps de première exécution.
Supporte FP32 ou FP16 (autocast). Par défaut mode "reduce-overhead".

Usage principal (on wsl):
  python scripts/compile/benchmark_compile_mobilenetv3.py --mode fp32 --compile-mode reduce-overhead \
         --num-warmup 50 --num-runs 500

    python scripts/compile/benchmark_compile_mobilenetv3.py --mode fp32 --compile-mode reduce-overhead --num-warmup 50 --num-runs 500

Options utiles:
  --mode fp16            # compile + autocast FP16 (nécessite CUDA)
  --skip-baseline        # ne mesure pas le modèle non compilé
  --num-warmup 5 --num-runs 50   # smoke test rapide
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cifaracce.data import test_loader
from cifaracce.models import MobileNetV3Small
from cifaracce.utils.seed import set_seed


def _forward(model, x, device: str, precision: str):
    if device == "cuda" and precision == "fp16":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            return model(x)
    return model(x)


def benchmark_latency(model, loader, device: str, precision: str,
                      num_warmup: int, num_runs: int):
    model.eval()
    lat_ms = []
    with torch.no_grad():
        # Warm-up
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
                lat_ms.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = _forward(model, x, device, precision)
                t1 = time.perf_counter()
                lat_ms.append((t1 - t0) * 1000.0)
            runs += 1

    return np.asarray(lat_ms, dtype=np.float64)


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


def load_student(device: str):
    model = MobileNetV3Small(num_classes=10, device=device).to(device)
    ckpt = Path("checkpoints/mobilenetv3/mobilenetv3_best.pt")
    if ckpt.exists():
        print(f"Loading checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print("✓ Weights loaded")
    else:
        print("⚠ No checkpoint found, using random weights (latency only)")
    return model


def first_call_time(model, loader, device: str, precision: str):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(loader))
        x = x.to(device, non_blocking=True)
        if device == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = _forward(model, x, device, precision)
            end.record()
            torch.cuda.synchronize()
            return float(start.elapsed_time(end))
        t0 = time.perf_counter()
        _ = _forward(model, x, device, precision)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0


def main():
    parser = argparse.ArgumentParser(description="torch.compile benchmark for MobileNetV3-Small")
    parser.add_argument("--mode", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--compile-mode", default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode")
    parser.add_argument("--num-warmup", type=int, default=50)
    parser.add_argument("--num-runs", type=int, default=500)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    if args.mode == "fp16" and device == "cpu":
        print("⚠ FP16 requested but CUDA not available; results will behave like FP32 on CPU.")

    # Quick guard for triton/inductor availability (common failure on Windows)
    try:
        import triton  # type: ignore
        _ = triton.__version__
    except Exception:
        print("✗ torch.compile (inductor) requires Triton; not found or unsupported on this platform.")
        print("  • If on Windows, run under WSL/Linux for inductor+Triton, or skip compile.")
        print("  • Otherwise: pip install 'triton==2.1.0' (match your torch version).")
        return

    # Load model
    model = load_student(device)

    # Baseline (non-compiled)
    if not args.skip_baseline:
        print("\n== Baseline (uncompiled) ==")
        lat_base = benchmark_latency(
            model, test_loader, device, precision=args.mode,
            num_warmup=args.num_warmup, num_runs=args.num_runs,
        )
        stats_base = summarize(lat_base, "LATENCY (uncompiled)")
    else:
        stats_base = None

    # Compile
    print(f"\nCompiling model with torch.compile (mode={args.compile_mode}) ...")
    try:
        model_compiled = torch.compile(model, mode=args.compile_mode)
    except Exception as e:
        print(f"✗ torch.compile failed: {e}")
        print("  • Likely Triton/inductor unsupported or missing. Try running on Linux/WSL with CUDA")
        print("    and matching triton version, or rerun with --skip-baseline to keep eager only.")
        return

    # J4-06: Measure compilation time (first call, which includes compile overhead)
    print("\n" + "=" * 60)
    print("J4-06 | COMPILATION TIME MEASUREMENT")
    print("=" * 60)
    compile_first_ms = first_call_time(model_compiled, test_loader, device, precision=args.mode)
    print(f"First compiled call (includes compilation overhead): {compile_first_ms:.4f} ms")
    print("Note: This includes both JIT compilation and first inference.")
    print("=" * 60)

    # J4-07: Post-compile latency (after compilation is warm)
    print("\n== J4-07 | POST-COMPILATION LATENCY BENCHMARK ==")
    lat_comp = benchmark_latency(
        model_compiled, test_loader, device, precision=args.mode,
        num_warmup=args.num_warmup, num_runs=args.num_runs,
    )
    stats_comp = summarize(lat_comp, f"LATENCY (compiled, {args.mode.upper()})")

    # Summary comparison table
    if stats_base is not None:
        mean_base, mean_comp = stats_base["mean"], stats_comp["mean"]
        p95_base, p95_comp = stats_base["p95"], stats_comp["p95"]
        speedup_mean = mean_base / mean_comp if mean_comp > 0 else float("inf")
        speedup_p95 = p95_base / p95_comp if p95_comp > 0 else float("inf")

        # Comparison table
        print("\n" + "=" * 80)
        print("COMPARISON: Uncompiled vs Compiled (FP32/FP16)")
        print("=" * 80)
        comparison_rows = [
            ["Metric", "Baseline (eager)", "Compiled", "Speedup"],
            ["-" * 10, "-" * 15, "-" * 12, "-" * 10],
            [f"Mean latency (ms)", f"{mean_base:.4f}", f"{mean_comp:.4f}", f"×{speedup_mean:.2f}"],
            [f"p95 latency (ms)", f"{p95_base:.4f}", f"{p95_comp:.4f}", f"×{speedup_p95:.2f}"],
        ]
        for row in comparison_rows:
            print(f"  {row[0]:<20} {row[1]:<18} {row[2]:<15} {row[3]:<12}")
        print("=" * 80)


if __name__ == "__main__":
    main()
