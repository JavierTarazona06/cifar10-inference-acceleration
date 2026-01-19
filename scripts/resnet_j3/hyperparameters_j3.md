# J3-13 | Reproducibility Sheet - ResNet-18 CIFAR-10

## 📋 Executive Summary

- **Model**: ResNet-18 adapted for CIFAR-10 (from-scratch)
- **Dataset**: CIFAR-10 (50,000 train / 10,000 test)
- **Objective**: Accuracy ≥85% on test set
- **Documentation date**: 2026-01-19

---

## 🏗️ Model Architecture

### Modifications vs standard ResNet-18 (ImageNet)

| Aspect | ImageNet | CIFAR-10 (J3-01) |
|--------|----------|------------------|
| First conv | 7×7, stride=2 | **3×3, stride=1** |
| First conv padding | 3 | **1** |
| MaxPool initial | Yes (2×2, stride=2) | **No (Identity)** |
| Input size | 224×224 | 32×32 |
| Output classes | 1000 | **10** |
| Weights initialization | `weights=None` | **`weights=None` (from-scratch)** |

### Architecture summary

```
ResNet-18(weights=None)
├── conv1: Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
├── maxpool: Identity() (REMOVED)
├── layer1/2/3/4: ResidualBlocks
├── avgpool: AdaptiveAvgPool2d((1, 1))
└── fc: Linear(512, 10)
```

**Total parameters**: ~11.17M (trainable)

---

## ⚙️ Training Hyperparameters

### Optimizer

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,                    # Initial learning rate
    momentum=0.9,              # Momentum
    weight_decay=5e-4          # L2 regularization
)
```

### Learning Rate Scheduler

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=200                  # Number of epochs
)
```

**Description**: Learning rate follows a cosine annealing schedule decreasing over 200 epochs.

### Loss Function

```python
loss = torch.nn.functional.cross_entropy(outputs, targets)
```

### General Training

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Epochs** | 200 | Typical for CIFAR-10 with cosine schedule |
| **Batch size (train)** | 128 | Balance between memory and convergence |
| **Batch size (test)** | 1 | Latency measurement (benchmarking) |
| **Early stopping floor** | 80 | Minimum epochs before early stopping |
| **Early stopping patience** | 15 | Epochs without improvement before stop |
| **Target accuracy** | ≥85% | Success criterion |
| **Seed** | 42 | Reproducibility |

---

## 📊 Data Configuration

### Data Augmentation (Training)

```python
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # Crop 32×32 with padding 4
    transforms.RandomHorizontalFlip(p=0.5),    # Horizontal flip 50%
    transforms.ToTensor(),                     # Conversion to tensor
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],        # CIFAR-10 mean
        std=[0.2023, 0.1994, 0.2010]          # CIFAR-10 std (J3-04)
    )
])
```

### Data Normalization (Test)

```python
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])
```

### Dataset

- **Source**: `torchvision.datasets.CIFAR10`
- **Train**: 50,000 images (10 classes × 5000)
- **Test**: 10,000 images (10 classes × 1000)
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

## 🚀 GPU Optimizations (J3-03 bis)

If using the optimized version of `train_resnet18.py`:

### Mixed Precision (FP16)

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast(dtype=torch.float16):
    outputs = model(inputs)
    loss = torch.nn.functional.cross_entropy(outputs, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefit**: ~30-50% speedup in training time

### torch.compile

```python
model = torch.compile(model, mode="reduce-overhead")
```

**Benefit**: ~10-20% additional speedup (torch >= 2.0)

### channels_last Memory Format

```python
model = model.to(memory_format=torch.channels_last)
inputs = inputs.to(memory_format=torch.channels_last)
```

**Benefit**: ~5-10% speedup (better GPU memory access pattern)

---

## 💾 Checkpoints and Outputs

### Generated Files

| File | Description |
|------|-------------|
| `checkpoints/resnet18/resnet18_cifar_best.pth` | Best checkpoint (epoch with highest test accuracy) |
| `checkpoints/resnet18/resnet18_cifar_last.pth` | Last checkpoint (last trained epoch) |
| `results_resnet18_training.csv` | Logs per epoch (loss, accuracy, LR) |

### Checkpoint Content

```python
checkpoint = {
    "epoch": int,                              # Epoch number
    "model_state_dict": model.state_dict(),   # Model weights
    "optimizer_state_dict": optimizer.state_dict(),  # Optimizer state
    "scheduler_state_dict": scheduler.state_dict(),  # Scheduler state
    "best_acc": float,                         # Best accuracy seen
    "loss": float,                             # Final loss
}
```

---

## 📈 Reproduction - Step by Step

### 1. Environment

```bash
# Create conda environment
conda create -n cifar10-resnet python=3.11
conda activate cifar10-resnet

# Install dependencies
pip install torch torchvision pytorch-cuda=12.1
pip install numpy scipy scikit-learn tabulate
```

### 2. Train from scratch

```bash
python train_resnet18.py
```

**Estimated duration**: 8-16 hours on modern GPU (RTX 3090/4090)
**Output**: Checkpoints in `checkpoints/resnet18/`

### 3. Evaluation

```bash
# Evaluate final accuracy
python evaluate_resnet18.py

# Measure GPU latency (FP32, batch=1)
python eval_latency_resnet18.py

# Document model size
python model_size_resnet18.py
```

### 4. Resume training (if needed)

```bash
python resume_training.py \
    --checkpoint checkpoints/resnet18/resnet18_cifar_best.pth \
    --epochs 50
```

---

## 🔬 Expected Metrics

Based on standard ResNet-18 CIFAR-10 configuration:

| Metric | Expected Value | Success Criterion |
|--------|---|---|
| **Test Accuracy** | 93-95% | ≥85% ✓ |
| **Mean Latency (FP32, B=1)** | 4-6 ms | < 10 ms |
| **Parameters** | 11.17M | ~11M |
| **Checkpoint size** | 42-45 MB | < 100 MB |

---

## 🖥️ Hardware and Software

### Minimum Requirements

- **GPU**: CUDA-capable (RTX, A100, V100, etc.)
- **VRAM**: 4 GB (8 GB recommended)
- **RAM**: 16 GB
- **Disk**: 5 GB (data + checkpoints)

### Tested Configuration

```python
Device: CUDA
PyTorch version: 2.0+
CUDA version: 12.1
Python: 3.11
```

---

## 📝 Reproducibility Notes

### Factors affecting reproducibility

1. **Seed**: Fixed to 42 in `set_seed(42)`
2. **Determinism**: `torch.manual_seed()`, `np.random.seed()`
3. **Data order**: `shuffle=True` in DataLoader (with seed)
4. **Numerical precision**: FP32 (FP16 may vary slightly)

### Expected variability

Even with fixed seed, small variations (<1%) may occur due to:
- Non-deterministic GPU operations
- Data parallelism
- Hardware differences

For maximum reproducibility:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 📚 References

- **Architecture**: He et al. (2015) - "Deep Residual Learning for Image Recognition"
- **Dataset**: Krizhevsky et al. (2009) - "Learning Multiple Layers of Features from Tiny Images"
- **CIFAR-10 adaptation**: Common in literature (papers on ResNet, MobileNet, etc.)
- **Hyperparameters**: Based on standard community practices

---

## ✅ Reproducibility Checklist

- [x] Model architecture documented
- [x] Training hyperparameters specified
- [x] Data configuration detailed
- [x] Loss function defined
- [x] Seed and determinism controlled
- [x] Checkpoint format documented
- [x] Training command clear
- [x] Expected metrics listed
- [x] Required hardware specified
- [x] Evaluation script provided

---

**Last update**: 2026-01-19  
**Version**: J3.0 (Day 3 - Precision Baseline)
