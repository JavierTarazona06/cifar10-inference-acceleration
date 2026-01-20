"""
Resume training from a checkpoint (resnet18_cifar_best.pth or resnet18_cifar_last.pth)

Usage:
    python resume_training.py --checkpoint checkpoints/resnet18/resnet18_cifar_best.pth --epochs 50
"""

import sys
import csv
import torch
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cifaracce.data import train_loader, test_loader
from cifaracce.models.resnet18 import ResNet18
from cifaracce.utils.seed import set_seed


def load_checkpoint(checkpoint_path, device):
    """Load model, optimizer, and scheduler state from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = ResNet18(num_classes=10).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    start_epoch = checkpoint["epoch"] + 1
    best_acc = checkpoint["best_acc"]
    
    print(f" Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Best acc so far: {best_acc:.2f}%")
    print(f"  Resuming from epoch {start_epoch}")
    
    return model, optimizer, scheduler, start_epoch, best_acc


def main():
    parser = argparse.ArgumentParser(description="Resume ResNet-18 training from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--target-acc", type=float, default=85.0, help="Target accuracy")
    
    args = parser.parse_args()
    
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f" Checkpoint not found: {checkpoint_path}")
        return
    
    model, optimizer, scheduler, start_epoch, best_acc = load_checkpoint(checkpoint_path, device)
    
    num_epochs = start_epoch + args.epochs - 1
    target_acc = args.target_acc
    
    checkpoints_dir = Path("checkpoints/resnet18")
    log_path = Path("results_resnet18_training.csv")
    
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        test_acc = 100.0 * correct / total
        
        scheduler.step()
        
        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
            f"Test Acc: {test_acc:6.2f}% | LR: {scheduler.get_last_lr()[0]:.5f}"
        )
        
        with log_path.open("a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "train_loss", "train_acc", "test_acc", "lr", "timestamp"],
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "lr": scheduler.get_last_lr()[0],
                    "timestamp": datetime.now().isoformat(),
                }
            )
        
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint_best = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_acc": best_acc,
                "loss": train_loss,
            }
            torch.save(checkpoint_best, checkpoints_dir / "resnet18_cifar_best.pth")
            print(f"  -> Saved best checkpoint: {best_acc:.2f}%")
        
        if test_acc >= target_acc:
            print(f" Target accuracy {target_acc:.1f}% reached at epoch {epoch}!")
            break
    
    checkpoint_last = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc,
        "loss": train_loss,
    }
    torch.save(checkpoint_last, checkpoints_dir / "resnet18_cifar_last.pth")
    
    print("\n" + "=" * 60)
    print("Training resumed and finished")
    print(f"Best acc: {best_acc:.2f}%")
    print(f"Checkpoint: {checkpoints_dir}/resnet18_cifar_best.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()
