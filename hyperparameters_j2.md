# Hyperparamètres J2 — MobileNetV3-Small Baseline

**Date**: 2026-01-18  
**Tâche**: J2-12 — Documentation pour reproductibilité  
**Modèle**: MobileNetV3-Small (from-scratch)

---

## 📐 Architecture

| Paramètre | Valeur |
|-----------|--------|
| Modèle | `torchvision.models.mobilenet_v3_small` |
| Poids pré-entraînés | `None` (entraînement from-scratch) |
| Nombre de classes | 10 (CIFAR-10) |
| Nombre de paramètres | ~1.5M |

---

## 🎯 Optimizer

| Paramètre | Valeur |
|-----------|--------|
| Type | SGD with momentum |
| Learning rate (initial) | 0.1 |
| Momentum | 0.9 |
| Weight decay | 5e-4 |

---

## 📊 Scheduler

| Paramètre | Valeur |
|-----------|--------|
| Type | CosineAnnealingLR |
| T_max | 60 epochs |
| eta_min | 0 (default) |

---

## 🏋️ Entraînement

| Paramètre | Valeur |
|-----------|--------|
| Nombre d'epochs | 60 |
| Batch size (train) | 128 |
| Batch size (test) | 1 |
| Early stopping | epoch ≥40 AND test_acc ≥85% |
| Device | CUDA (GPU) |
| Seed | 42 |
| Loss function | CrossEntropyLoss |

---

## 🔄 Augmentations de données (train)

| Transformation | Paramètres |
|----------------|------------|
| RandomHorizontalFlip | p=0.5 (default) |
| RandomCrop | size=32, padding=4 |
| RandomRotation | degrees=15 |
| ToTensor | - |
| Normalize | mean=[0.4914, 0.4822, 0.4465]<br>std=[0.2470, 0.2435, 0.2616] |

**Note**: Les statistiques de normalisation sont calculées sur CIFAR-10 train set.

---

## 🧪 Augmentations de données (test)

| Transformation | Paramètres |
|----------------|------------|
| ToTensor | - |
| Normalize | mean=[0.4914, 0.4822, 0.4465]<br>std=[0.2470, 0.2435, 0.2616] |

---

## 💾 Checkpointing

| Paramètre | Valeur |
|-----------|--------|
| Répertoire | `checkpoints/mobilenetv3/` |
| Meilleur modèle | `mobilenetv3_best.pt` (best test accuracy) |
| Dernier modèle | `mobilenetv3_last.pt` (epoch 60 ou early stop) |
| Format | `torch.save(model.state_dict(), ...)` |

---

## 📈 Logging

| Paramètre | Valeur |
|-----------|--------|
| Fichier CSV | `results_j2_mobilenet_training.csv` |
| Colonnes | epoch, train_loss, train_acc, test_acc, lr, timestamp |
| Console | Affichage par epoch |

---

## 🎲 Reproductibilité

```python
# Seed fixé au début du script
from cifaracce.utils.seed import set_seed
set_seed(seed=42)
```

Détails de la fonction `set_seed()`:
- `random.seed(42)`
- `np.random.seed(42)`
- `torch.manual_seed(42)`
- `torch.cuda.manual_seed_all(42)`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

---

## 🔧 Environnement

| Composant | Version |
|-----------|---------|
| Python | 3.x |
| PyTorch | 2.5.1 |
| torchvision | 0.20.1 |
| CUDA | 12.1 |
| Device | NVIDIA GPU (detecté automatiquement) |

---

## 📝 Commandes d'exécution

### Entraînement
```bash
python train_mobilenet_j2.py
```

### Évaluation latence
```bash
python eval_latency_j2.py
```

---


## 📚 Références

- Architecture: [MobileNetV3 (Howard et al., 2019)](https://arxiv.org/abs/1905.02244)
- CIFAR-10: [Learning Multiple Layers of Features from Tiny Images (Krizhevsky, 2009)](https://www.cs.toronto.edu/~kriz/cifar.html)
