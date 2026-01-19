"""
Train MobileNetV3-Small student with distillation from ResNet-18 teacher (CIFAR-10).

Defaults (J4-15): T=4.0, alpha=0.7, lr=0.01, SGD mom=0.9, wd=5e-4.

Usage (short example):
  python scripts/distillation/train_distill_mobilenet_j4.py --epochs 120 --lr 0.01 --alpha 0.7 --temperature 4.0

Checkpoints:
  - Teacher expected at checkpoints/resnet18/resnet18_best.pt (fallback: resnet18_cifar_best.pth)
  - Student best saved to checkpoints/distill/mobilenetv3_best.pt
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
    print(f"Loaded teacher from {ckpt_path}")
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


def train(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    teacher = load_teacher(Path(args.teacher_ckpt), device)
    student = MobileNetV3Small(num_classes=10, device=device).to(device)

    optimizer = torch.optim.SGD(
        student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_epoch = 0
    ckpt_dir = cfg.CHECKPOINTS['distill_dir']
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
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
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"TrainLoss {train_loss:.4f} (CE {ce_loss:.4f}, KL {kl_loss:.4f}) | "
            f"TrainAcc {train_acc:6.2f}% | TestAcc {test_acc:6.2f}% | "
            f"LR {scheduler.get_last_lr()[0]:.5f}", end="")

        # Save best (with full checkpoint for resuming)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            checkpoint_dict = {
                "epoch": epoch,
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

    # Save last (with full checkpoint for resuming)
    checkpoint_dict = {
        "epoch": epoch,
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc,
        "loss": train_loss,
    }
    torch.save(checkpoint_dict, ckpt_dir / "mobilenetv3_last.pt")
    print("\nTraining finished")
    print(f"Best acc: {best_acc:.2f}% (epoch {best_epoch})")
    print(f"Checkpoints: {ckpt_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Distillation training for MobileNetV3-Small (CIFAR-10)")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument(
        "--teacher-ckpt",
        type=str,
        default="checkpoints/resnet18/resnet18_best.pt",
        help="Path to teacher checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
