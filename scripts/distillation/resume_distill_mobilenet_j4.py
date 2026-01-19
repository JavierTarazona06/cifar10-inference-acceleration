"""
Resume distillation training for MobileNetV3-Small from checkpoint.

Load best checkpoint (epoch N, acc X%) and continue training for additional epochs.
Useful when initial training stalled below 85% and needs more epochs.

Usage:
  python scripts/distillation/resume_distill_mobilenet_j4.py \
    --checkpoint checkpoints/distill/mobilenetv3_best.pt \
    --additional-epochs 180 \
    --lr 0.001 \
    --prev-best-acc 79.43

This will:
1. Load best student checkpoint (epoch N, accuracy X%)
2. Load teacher checkpoint (ResNet-18)
3. Resume optimizer + scheduler from checkpoint (preserves momentum and LR schedule)
4. Train for additional_epochs more epochs (with lower LR for fine-tuning)
5. Save new best/last checkpoints

Note: Use --lr 0.001 (or lower) when resuming from epoch 120+ to avoid accuracy collapse.
      If optimizer/scheduler states are in checkpoint, they will be loaded automatically.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from cifaracce.data import train_loader, test_loader
from cifaracce.models.mobileNet import MobileNetV3Small
from cifaracce.models.resnet18 import ResNet18
from cifaracce.utils import set_seed, distillation_loss
from cifaracce import config as cfg


def load_teacher(ckpt_path: Path, device: str) -> torch.nn.Module:
    if not ckpt_path.exists():
        alt = ckpt_path.parent / "resnet18_cifar_best.pth"
        if alt.exists():
            ckpt_path = alt
        else:
            raise FileNotFoundError(f"Teacher checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ResNet18(num_classes=10).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"✓ Loaded teacher from {ckpt_path}")
    return model


def evaluate(model: torch.nn.Module, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def resume_training(args):
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    print(f"Using device: {device}\n")

    # Load best student checkpoint
    student_ckpt_path = Path(args.checkpoint)
    if not student_ckpt_path.exists():
        print(f"✗ Student checkpoint not found: {student_ckpt_path}")
        return

    print(f"Loading student checkpoint: {student_ckpt_path}")
    student_state = torch.load(student_ckpt_path, map_location=device, weights_only=False)
    
    student = MobileNetV3Small(num_classes=10, device=device).to(device)
    if isinstance(student_state, dict) and "model_state_dict" in student_state:
        student.load_state_dict(student_state["model_state_dict"])
        initial_epoch = student_state.get("epoch", 120)
        prev_best_acc = student_state.get("best_acc", 0.0)
    else:
        student.load_state_dict(student_state)
        initial_epoch = 120  # assume checkpoint from epoch 120
        prev_best_acc = args.prev_best_acc if hasattr(args, 'prev_best_acc') else 0.0

    print(f"✓ Student loaded (previous best acc: {prev_best_acc:.2f}%)\n")

    # Load teacher
    teacher_ckpt = cfg.CHECKPOINTS['resnet18_dir'] / "resnet18_best.pt"
    teacher = load_teacher(teacher_ckpt, device)
    print()

    # Optimizer: load from checkpoint or create fresh
    optimizer = torch.optim.SGD(
        student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    
    if isinstance(student_state, dict) and "optimizer_state_dict" in student_state:
        optimizer.load_state_dict(student_state["optimizer_state_dict"])
        print(f"✓ Optimizer state loaded from checkpoint\n")
    else:
        print(f"⚠ Optimizer state not in checkpoint, starting fresh with lr={args.lr}\n")

    # Scheduler: load from checkpoint or create fresh
    # If resuming, create scheduler for the additional epochs starting fresh
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.additional_epochs
    )
    
    if isinstance(student_state, dict) and "scheduler_state_dict" in student_state:
        try:
            scheduler.load_state_dict(student_state["scheduler_state_dict"])
            print(f"✓ Scheduler state loaded from checkpoint\n")
        except Exception as e:
            print(f"⚠ Could not load scheduler state ({e}), starting fresh\n")

    best_acc = prev_best_acc  # track best from this resume session
    best_epoch = initial_epoch
    ckpt_dir = cfg.CHECKPOINTS['distill_dir']
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Resume training: epoch {initial_epoch + 1} → {initial_epoch + args.additional_epochs}")
    print(f"Target: test_acc >= 85%\n")
    print("=" * 80)

    for epoch_offset in range(args.additional_epochs):
        current_epoch = initial_epoch + epoch_offset + 1

        student.train()
        running_loss = 0.0
        running_ce = 0.0
        running_kl = 0.0
        total = 0
        correct = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(inputs)
            loss, parts = distillation_loss(
                student_logits,
                teacher_logits,
                targets,
                alpha=args.alpha,
                temperature=args.temperature,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            running_ce += parts["ce_loss"] * targets.size(0)
            running_kl += parts["kl_loss"] * targets.size(0)

            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        ce_loss = running_ce / total
        kl_loss = running_kl / total

        test_acc = evaluate(student, device)

        print(
            f"Epoch {current_epoch:03d} | "
            f"TrainLoss {train_loss:.4f} (CE {ce_loss:.4f}, KL {kl_loss:.4f}) | "
            f"TrainAcc {train_acc:6.2f}% | TestAcc {test_acc:6.2f}% | "
            f"LR {scheduler.get_last_lr()[0]:.5f}", end="")

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = current_epoch
            checkpoint_dict = {
                "epoch": current_epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_acc": best_acc,
                "loss": train_loss,
            }
            torch.save(checkpoint_dict, ckpt_dir / "mobilenetv3_best.pt")
            print(" ← NEW BEST")
        else:
            print()

        # Early exit if target reached
        if test_acc >= 85.0:
            print(f"\n✓ Target accuracy {test_acc:.2f}% >= 85% REACHED at epoch {current_epoch}")
            break

    # Save last
    checkpoint_dict = {
        "epoch": current_epoch,
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc,
        "loss": train_loss,
    }
    torch.save(checkpoint_dict, ckpt_dir / "mobilenetv3_last.pt")

    print("\n" + "=" * 80)
    print("Training resumed and finished")
    print(f"Epochs trained: {initial_epoch + 1} → {current_epoch}")
    print(f"Best acc (in this resume): {best_acc:.2f}% (epoch {best_epoch})")
    print(f"Checkpoints: {ckpt_dir}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resume distillation training for MobileNetV3-Small (CIFAR-10)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/distill/mobilenetv3_best.pt",
        help="Path to best student checkpoint to resume from",
    )
    parser.add_argument(
        "--additional-epochs",
        type=int,
        default=180,
        help="Number of additional epochs to train (total will be initial_epoch + additional_epochs)",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (fresh start)")
    parser.add_argument("--alpha", type=float, default=0.7, help="Distillation loss weight")
    parser.add_argument(
        "--temperature", type=float, default=4.0, help="Temperature for distillation"
    )
    parser.add_argument(
        "--prev-best-acc",
        type=float,
        default=0.0,
        help="Previous best accuracy (informational)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resume_training(args)
