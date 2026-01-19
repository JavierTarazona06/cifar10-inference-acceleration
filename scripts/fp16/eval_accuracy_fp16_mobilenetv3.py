"""
J4-03 | Vérifier l'accuracy en FP16 (MobileNetV3-Small)

Compare l'accuracy FP32 vs FP16 (autocast) sur le test CIFAR-10.
Affiche le delta et vérifie que la chute < 0.5% (par défaut).

Usage:
  python eval_accuracy_fp16_mobilenetv3.py [--threshold 0.5]
  # Optionnel pour un smoke test rapide:
  python eval_accuracy_fp16_mobilenetv3.py --limit 200
"""

import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cifaracce.data import test_loader
from cifaracce.models import MobileNetV3Small
from cifaracce.utils.seed import set_seed


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


def load_student(device: str):
    model = MobileNetV3Small(num_classes=10, device=device).to(device)
    ckpt_dir = Path("checkpoints/mobilenetv3")
    ckpt_path = ckpt_dir / "mobilenetv3_best.pt"

    if ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])  # compatibility
        else:
            model.load_state_dict(state)
        print("✓ Weights loaded")
    else:
        print("⚠ No checkpoint found — evaluating random weights (for demo only)")

    return model


def main():
    parser = argparse.ArgumentParser(description="J4-03: Verify FP16 accuracy drop < threshold")
    parser.add_argument("--threshold", type=float, default=0.5, help="Max allowed drop in percentage points")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of test samples for a quick check")
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    if device == "cpu":
        print("⚠ Running on CPU — FP16 autocast will behave like FP32.")

    model = load_student(device)

    acc_fp32 = evaluate_accuracy(model, device, precision="fp32", limit=args.limit)
    acc_fp16 = evaluate_accuracy(model, device, precision="fp16", limit=args.limit)

    drop = acc_fp32 - acc_fp16

    print("\n== Accuracy Comparison (MobileNetV3-Small) ==")
    print(f"FP32 accuracy: {acc_fp32:.2f}%")
    print(f"FP16 accuracy: {acc_fp16:.2f}%")
    print(f"Δ (FP32 - FP16): {drop:.2f} pp")
    verdict = "OK" if drop <= args.threshold else "FAIL"
    print(f"Verdict (threshold {args.threshold:.2f} pp): {verdict}")


if __name__ == "__main__":
    main()
