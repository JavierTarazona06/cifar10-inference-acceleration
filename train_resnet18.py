"""
Training ResNet-18 on CIFAR-10 (from scratch)
Hyperparameters:
- Optimizer: SGD(lr=0.1, momentum=0.9, weight_decay=5e-4)
- Scheduler: CosineAnnealingLR(T_max=200)
- Epochs: 200 (early stop if target acc reached after a floor)
- Batch size: defined in cifaracce.data (train=128)
"""

import sys
import csv
import torch
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cifaracce.data import train_loader, test_loader
from cifaracce.models.resnet18 import ResNet18
from cifaracce.utils.seed import set_seed


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    num_epochs = 200
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    target_acc = 85.0
    early_stop_floor = 120  # allow stopping once accuracy is stable after this epoch

    model = ResNet18(num_classes=10).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    # To use MultiStepLR instead, replace the scheduler above with:
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[100, 150], gamma=0.1
    # )

    best_acc = 0.0
    best_epoch = 0
    checkpoints_dir = Path("checkpoints/resnet18")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path("results_resnet18_training.csv")
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "test_acc", "lr", "timestamp"],
        )
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
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
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoints_dir / "resnet18_best.pt")

        if epoch >= early_stop_floor and test_acc >= target_acc:
            print(f"Early stop: test_acc {test_acc:.2f}% >= {target_acc:.1f}% at epoch {epoch}")
            break

    torch.save(model.state_dict(), checkpoints_dir / "resnet18_last.pt")

    print("\n" + "=" * 60)
    print("Training finished")
    print(f"Best acc: {best_acc:.2f}% (epoch {best_epoch})")
    print(f"Checkpoints: {checkpoints_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()