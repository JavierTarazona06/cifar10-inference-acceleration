"""
Quick channels_last vs NCHW latency check.

Usage examples (short runs):
    python scripts/channels/quick_channels_last_bench.py --mode fp32 --warmup 10 --runs 50
    python scripts/channels/quick_channels_last_bench.py --mode fp16 --warmup 10 --runs 50
    python scripts/channels/quick_channels_last_bench.py --mode fp32 --device cpu --runs 20  # CPU sanity

Options:
    --mode fp32|fp16        Precision (fp16 uses autocast)
    --device cuda|cpu       Target device (default: cuda)
    --checkpoint PATH       Checkpoint to load (default: checkpoints/mobilenetv3/mobilenetv3_best.pt)
    --warmup N              Warmup iterations (default: 10)
    --runs N                Measured iterations (default: 50)
"""

import argparse
from pathlib import Path

import torch

from cifaracce.data import test_loader
from cifaracce.bench import benchmark_latency
from cifaracce.models.mobileNet import MobileNetV3Small


DEFAULT_CKPT = "checkpoints/mobilenetv3/mobilenetv3_best.pt"


def load_model(checkpoint_path: str, device: str) -> MobileNetV3Small:
    model = MobileNetV3Small(device=device)
    if checkpoint_path and Path(checkpoint_path).is_file():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("Checkpoint not found; using randomly initialized weights")
    model.eval()
    return model


def run_variant(label: str, channels_last: bool, use_amp: bool, args):
    model = load_model(args.checkpoint, args.device)
    stats, _ = benchmark_latency(
        model,
        test_loader,
        warmup_iters=args.warmup,
        measure_iters=args.runs,
        device=args.device,
        channels_last=channels_last,
        use_amp=use_amp,
        amp_dtype=torch.float16 if use_amp else torch.float32,
    )
    print(f"\n[{label}] Mean: {stats['mean']:.4f} ms | P95: {stats['p95']:.4f} ms | Std: {stats['std']:.4f} ms")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Quick latency check for channels_last vs NCHW")
    parser.add_argument("--device", default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument("--mode", choices=["fp32", "fp16"], default="fp32", help="Precision for inference")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Path to model checkpoint")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (short run)")
    parser.add_argument("--runs", type=int, default=50, help="Measured iterations (short run)")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available but device='cuda' was requested")

    use_amp = args.mode == "fp16"

    print("Running quick latency check")
    print(f"  device: {args.device}")
    print(f"  precision: {args.mode}")
    print(f"  warmup: {args.warmup}")
    print(f"  runs: {args.runs}")
    print(f"  checkpoint: {args.checkpoint}")

    baseline_stats = run_variant("baseline_nchw", channels_last=False, use_amp=use_amp, args=args)
    cl_stats = run_variant("channels_last", channels_last=True, use_amp=use_amp, args=args)

    def speedup(a, b):
        return a / b if b > 0 else float("nan")

    print("\nSummary (channels_last vs baseline)")
    print(f"  Mean speedup: x{speedup(baseline_stats['mean'], cl_stats['mean']):.2f}")
    print(f"  P95 speedup:  x{speedup(baseline_stats['p95'], cl_stats['p95']):.2f}")


if __name__ == "__main__":
    main()
