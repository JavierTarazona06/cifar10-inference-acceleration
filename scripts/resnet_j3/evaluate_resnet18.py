"""
J3-09 | Évaluer l'accuracy finale sur test

Charger le best checkpoint (resnet18_cifar_best.pth) et calculer l'accuracy sur CIFAR-10 test.
Critère de succès : accuracy ≥85%

Usage:
    python evaluate_resnet18.py
"""

import sys
import torch
from pathlib import Path
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cifaracce.data import test_loader
from cifaracce.models.resnet18 import ResNet18
from cifaracce.utils.seed import set_seed


def evaluate_checkpoint(checkpoint_path, device):
    """Load checkpoint and evaluate on CIFAR-10 test set."""
    
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = ResNet18(num_classes=10).to(device)
    
    # Handle both formats: old (state_dict only) and new (full checkpoint dict)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "unknown")
        best_acc = checkpoint.get("best_acc", "unknown")
    else:
        # Old format: direct state_dict
        model.load_state_dict(checkpoint)
        epoch = "unknown"
        best_acc = "unknown"
    
    model.eval()
    
    if epoch != "unknown":
        print(f"  Checkpoint epoch: {epoch}")
    if best_acc != "unknown":
        print(f"  Checkpoint best_acc: {best_acc:.2f}%")
    print()
    
    # Evaluate on test set
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Progress
            if (batch_idx + 1) % 2000 == 0:
                print(f"  Evaluated {batch_idx + 1}/{len(test_loader)} batches...")
    
    test_acc = 100.0 * correct / total
    
    # Return tuple with flexible best_acc handling
    if isinstance(best_acc, str):
        training_best_acc = None
    else:
        training_best_acc = best_acc
    
    return test_acc, epoch, training_best_acc


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    checkpoint_path = Path("checkpoints/resnet18/resnet18_cifar_best.pth")
    
    # Fallback to old format if new one doesn't exist
    if not checkpoint_path.exists():
        old_checkpoint_path = Path("checkpoints/resnet18/resnet18_best.pt")
        if old_checkpoint_path.exists():
            print(f"⚠ New checkpoint not found, using old format: {old_checkpoint_path}\n")
            checkpoint_path = old_checkpoint_path
    
    print("=" * 60)
    print("J3-09 | Evaluation du Teacher ResNet-18 sur CIFAR-10 Test")
    print("=" * 60 + "\n")
    
    result = evaluate_checkpoint(checkpoint_path, device)
    
    if result is None:
        print("✗ Evaluation failed")
        return
    
    test_acc, epoch, training_best_acc = result
    
    # Create results table
    results = [
        ["Epoch (checkpoint)", epoch if epoch != "unknown" else "N/A"],
        ["Accuracy (training log)", f"{training_best_acc:.2f}%" if training_best_acc is not None else "N/A"],
        ["Accuracy (test eval)", f"{test_acc:.2f}%"],
        ["Target accuracy", "85.00%"],
        ["Status", "✓ PASS" if test_acc >= 85.0 else "✗ FAIL"],
    ]
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(tabulate(results, headers=["Metric", "Value"], tablefmt="grid"))
    print("=" * 60)
    
    # Detailed assessment
    print("\n📊 Assessment:")
    if test_acc >= 85.0:
        if training_best_acc is not None:
            gap = test_acc - training_best_acc
            print(f"  ✓ Target accuracy ≥85% ACHIEVED: {test_acc:.2f}%")
            if abs(gap) < 0.5:
                print(f"  ✓ No gap between training and test (gap: {gap:.2f}%)")
            elif gap > 0:
                print(f"  ⚠ Test accuracy slightly higher than training (gap: {gap:.2f}%)")
            else:
                print(f"  ⚠ Overfitting detected (gap: {gap:.2f}%)")
        else:
            print(f"  ✓ Target accuracy ≥85% ACHIEVED: {test_acc:.2f}%")
    else:
        gap = 85.0 - test_acc
        print(f"  ✗ Target accuracy NOT achieved: {test_acc:.2f}% (need {gap:.2f}% more)")
    
    print("\n" + "=" * 60)
    print("Teacher ResNet-18 ready for J4 (Distillation)" if test_acc >= 85.0 else "Teacher needs improvement")
    print("=" * 60)
    
    return test_acc >= 85.0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
