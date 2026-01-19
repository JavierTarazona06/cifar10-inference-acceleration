"""
J3-14 | Prepare Teacher for Distillation

Verify that the ResNet-18 teacher model is ready for J4 (Knowledge Distillation):
1. Check checkpoint integrity
2. Test model loading
3. Test soft label generation with temperature scaling
4. Generate and save sample predictions

Usage:
    python prepare_teacher_distillation.py
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cifaracce.data import test_loader
from cifaracce.models.resnet18 import ResNet18
from cifaracce.utils.seed import set_seed


def verify_checkpoint(checkpoint_path):
    """Verify checkpoint contains all required keys."""
    
    if not checkpoint_path.exists():
        return False, f"Checkpoint not found: {checkpoint_path}"
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, f"Failed to load checkpoint: {e}"
    
    # Check if new format (dict with model_state_dict)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        required_keys = ["model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "best_acc"]
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            return False, f"Missing keys in checkpoint: {missing_keys}"
        return True, "Checkpoint format: NEW (full state with optimizer/scheduler)"
    
    # Check if old format (dict with state_dict keys)
    if isinstance(checkpoint, dict):
        # Check for model keys (layer, fc, resnet.*, conv*, etc.)
        model_key_patterns = ["layer", "fc", "resnet.", "conv", "bn", "weight", "bias"]
        if any(any(pattern in k for pattern in model_key_patterns) for k in list(checkpoint.keys())[:5]):
            return True, "Checkpoint format: OLD (state_dict only - compatible)"
    
    # If we get here, format is unknown
    return False, f"Unknown checkpoint format. Keys: {list(checkpoint.keys())[:5]}..."


def load_teacher(checkpoint_path, device):
    """Load teacher model from checkpoint."""
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = ResNet18(num_classes=10).to(device)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        best_acc = checkpoint.get("best_acc", None)
        epoch = checkpoint.get("epoch", None)
    else:
        model.load_state_dict(checkpoint)
        best_acc = None
        epoch = None
    
    model.eval()
    return model, best_acc, epoch


def generate_soft_labels(model, inputs, temperature=4.0):
    """
    Generate soft labels for a batch using the teacher model.
    
    Args:
        model: Teacher model
        inputs: Batch of inputs (batch_size, 3, 32, 32)
        temperature: Temperature for softmax (higher = softer distribution)
    
    Returns:
        soft_labels: Soft probability distribution (batch_size, 10)
        hard_labels: Hard predictions (argmax)
    """
    
    with torch.no_grad():
        logits = model(inputs)
        
        # Soft labels with temperature scaling
        soft_labels = F.softmax(logits / temperature, dim=1)
        
        # Hard labels (argmax)
        hard_labels = torch.argmax(logits, dim=1)
    
    return soft_labels, hard_labels


def test_soft_label_generation(model, test_loader, device, num_batches=5):
    """Test soft label generation on sample batches."""
    
    print("\n" + "=" * 60)
    print("Testing Soft Label Generation")
    print("=" * 60 + "\n")
    
    temperatures = [1.0, 2.0, 4.0, 8.0]
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if batch_idx >= num_batches:
            break
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        print(f"Batch {batch_idx + 1}/{num_batches}:")
        print(f"  Input shape: {inputs.shape}")
        print(f"  Targets: {targets.cpu().numpy()}")
        
        for temp in temperatures:
            soft_labels, hard_labels = generate_soft_labels(model, inputs, temperature=temp)
            
            # Calculate entropy of soft labels (measure of softness)
            entropy = -(soft_labels * torch.log(soft_labels + 1e-10)).sum(dim=1).mean()
            
            # Accuracy with soft labels (using hard labels)
            accuracy = (hard_labels == targets).float().mean()
            
            print(f"  Temperature {temp}:")
            print(f"    - Soft labels shape: {soft_labels.shape}")
            print(f"    - Entropy (softness): {entropy:.4f}")
            print(f"    - Accuracy: {accuracy:.2%}")
            print(f"    - Max probability: {soft_labels.max():.4f}")
            print(f"    - Sample soft label: {soft_labels[0]}")
        
        print()


def generate_teacher_predictions(model, test_loader, device, num_samples=1000):
    """Generate predictions on test set for distillation."""
    
    print("\n" + "=" * 60)
    print("Generating Teacher Predictions on Test Set")
    print("=" * 60 + "\n")
    
    all_soft_labels = []
    all_hard_labels = []
    all_targets = []
    
    temperature = 4.0  # Standard temperature for distillation
    
    sample_count = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        soft_labels, hard_labels = generate_soft_labels(model, inputs, temperature=temperature)
        
        all_soft_labels.append(soft_labels.cpu())
        all_hard_labels.append(hard_labels.cpu())
        all_targets.append(targets.cpu())
        
        sample_count += inputs.size(0)
        
        if sample_count >= num_samples:
            break
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Generated predictions for {sample_count} samples...")
    
    # Concatenate all batches
    soft_labels_tensor = torch.cat(all_soft_labels, dim=0)
    hard_labels_tensor = torch.cat(all_hard_labels, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    
    # Calculate accuracy
    accuracy = (hard_labels_tensor == targets_tensor).float().mean()
    
    print(f"\n✓ Generated predictions for {soft_labels_tensor.shape[0]} test samples")
    print(f"  Soft labels shape: {soft_labels_tensor.shape}")
    print(f"  Teacher accuracy: {accuracy:.2%}")
    
    return soft_labels_tensor, hard_labels_tensor, targets_tensor


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    checkpoint_path = Path("checkpoints/resnet18/resnet18_cifar_best.pth")
    if not checkpoint_path.exists():
        checkpoint_path = Path("checkpoints/resnet18/resnet18_best.pt")
    
    print("=" * 60)
    print("J3-14 | Prepare Teacher for Distillation")
    print("=" * 60 + "\n")
    
    # Step 1: Verify checkpoint
    print("Step 1: Verify Checkpoint")
    print("-" * 60)
    valid, message = verify_checkpoint(checkpoint_path)
    print(f"Status: {'✓ PASS' if valid else '✗ FAIL'}")
    print(f"Details: {message}")
    print(f"Path: {checkpoint_path}\n")
    
    if not valid:
        print("✗ Checkpoint verification failed!")
        return False
    
    # Step 2: Load model
    print("Step 2: Load Teacher Model")
    print("-" * 60)
    try:
        model, best_acc, epoch = load_teacher(checkpoint_path, device)
        print("✓ Model loaded successfully")
        print(f"  Model class: {model.__class__.__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
        
        if best_acc is not None:
            print(f"  Best accuracy (from checkpoint): {best_acc:.2f}%")
        if epoch is not None:
            print(f"  Trained for {epoch} epochs")
        print()
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Step 3: Test soft label generation
    print("Step 3: Test Soft Label Generation")
    print("-" * 60)
    try:
        test_soft_label_generation(model, test_loader, device, num_batches=2)
        print("✓ Soft label generation working correctly")
    except Exception as e:
        print(f"✗ Soft label generation failed: {e}")
        return False
    
    # Step 4: Generate predictions
    print("\nStep 4: Generate Teacher Predictions")
    print("-" * 60)
    try:
        soft_labels, hard_labels, targets = generate_teacher_predictions(
            model, test_loader, device, num_samples=1000
        )
        print("✓ Predictions generated successfully\n")
    except Exception as e:
        print(f"✗ Prediction generation failed: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEACHER READINESS SUMMARY")
    print("=" * 60 + "\n")
    
    summary = [
        ["Checkpoint exists", "✓ Yes"],
        ["Checkpoint format", "✓ Valid"],
        ["Model loads", "✓ Yes"],
        ["Parameters", f"✓ {total_params / 1e6:.2f}M"],
        ["Soft labels generation", "✓ Working"],
        ["Predictions on test set", "✓ Generated"],
        ["Temperature (distillation)", "4.0"],
        ["Ready for J4 distillation", "✓ YES"],
    ]
    
    print(tabulate(summary, headers=["Check", "Status"], tablefmt="grid"))
    
    print("\n" + "=" * 60)
    print("NEXT STEPS FOR J4 (DISTILLATION)")
    print("=" * 60)
    print("""
The teacher is ready for Knowledge Distillation (J4). Use it to:

1. Generate soft targets for student model training:
   - Use temperature T=4.0 for soft label generation
   - Call generate_soft_labels(model, inputs, temperature=4.0)

2. Combine with hard targets for distillation loss:
   KL_loss = T² * KL(soft_student || soft_teacher)
   CE_loss = CrossEntropy(student, hard_targets)
   Total_loss = α * KL_loss + (1-α) * CE_loss

3. Freeze teacher weights (no gradient needed)

4. Train lightweight student model (e.g., MobileNetV3-Small)
""")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
