"""
Entrenamiento J2 - MobileNetV3-Small (CIFAR-10, from scratch)
Hyperparams fijados (J2-04):
- Optimizer: SGD(lr=0.1, momentum=0.9, weight_decay=5e-4)
- Scheduler: CosineAnnealingLR(T_max=num_epochs)
- Epochs: 60 (corta si test_acc >= 85% a partir de epoch 40)
- Batch size: 128 (definido en data.py)
- Device: auto cuda / cpu
- Seed: 42
"""

import sys
import csv
import torch
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from cifaracce.data import train_loader, test_loader
from cifaracce.models import MobileNetV3Small, get_model_info
from cifaracce.utils.seed import set_seed


def main():
    set_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    num_epochs = 60
    early_stop_epoch = 40  # evaluar corte si ya >=85%

    model = MobileNetV3Small(num_classes=10, device=device)
    model.to(device)

    # Hyperparams
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.12, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    get_model_info(model, device)

    best_acc = 0.0
    best_epoch = 0
    checkpoints_dir = Path('checkpoints/mobilenetv3')
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # CSV logging
    log_path = Path('results_j2_mobilenet_training.csv')
    with log_path.open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['epoch', 'train_loss', 'train_acc', 'test_acc', 'lr', 'timestamp']
        )
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        # Train
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

        # Eval
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

        print(f"Epoch {epoch:02d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
              f"Test Acc: {test_acc:6.2f}% | LR: {scheduler.get_last_lr()[0]:.5f}")

        # Log CSV
        with log_path.open('a', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['epoch', 'train_loss', 'train_acc', 'test_acc', 'lr', 'timestamp']
            )
            writer.writerow({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'lr': scheduler.get_last_lr()[0],
                'timestamp': datetime.now().isoformat()
            })

        # Checkpoint best
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoints_dir / 'mobilenetv3_best.pt')

        # Early stop opcional
        if epoch >= early_stop_epoch and test_acc >= 85.0:
            print(f"Early stop: test_acc {test_acc:.2f}% >= 85% en epoch {epoch}")
            break

    # Save last
    torch.save(model.state_dict(), checkpoints_dir / 'mobilenetv3_last.pt')

    # Resumen final
    print("\n" + "=" * 60)
    print("Training finished")
    print(f"Best acc: {best_acc:.2f}% (epoch {best_epoch})")
    print(f"Checkpoints: {checkpoints_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
