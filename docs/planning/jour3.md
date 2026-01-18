# Jour 3 — Baseline précision (teacher) + Itération I2

## 🎯 Objectif
Obtenir une référence robuste ≥85% et disposer d'un teacher pour la distillation.

---

## 📋 Tâches

### 🏗️ Architecture ResNet-18 adaptée CIFAR

- [ ] **J3-01** | Implémenter ResNet-18 adaptée CIFAR-10
  - **Description** :
    - Modifier la première couche conv : kernel 3×3, stride 1, padding 1 (au lieu de 7×7)
    - Supprimer le MaxPool initial (images 32×32 trop petites)
    - Tête de sortie à **10 classes**
    - `weights=None` (entraînement from-scratch)
  - **Labels** : `architecture`, `code`
  - **Priorité** : 🔴 Haute

- [ ] **J3-02** | Vérifier le forward pass
  - **Description** : Tester avec un batch fictif (1, 3, 32, 32) et vérifier la sortie (1, 10)
  - **Labels** : `test`, `validation`
  - **Priorité** : 🟡 Moyenne

---

### ⚙️ Configuration de l'entraînement

- [ ] **J3-03** | Définir les hyperparamètres
  - **Description** :
    - **LR initial** : 0.1 (typique pour SGD sur CIFAR)
    - **Optimizer** : SGD avec momentum 0.9, weight decay 5e-4
    - **Scheduler** : CosineAnnealingLR ou MultiStepLR (milestones 100, 150)
    - **Epochs** : 200 (ou moins si convergence rapide)
    - **Batch size** : 128
  - **Labels** : `hyperparamètres`, `configuration`
  - **Priorité** : 🔴 Haute

- [ ] **J3-04** | Configurer les augmentations robustes
  - **Description** :
    - RandomCrop(32, padding=4)
    - RandomHorizontalFlip(p=0.5)
    - Normalisation CIFAR-10 : mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
    - (Optionnel) Cutout, MixUp, ou AutoAugment si besoin de boost
  - **Labels** : `data`, `augmentation`
  - **Priorité** : 🟡 Moyenne

- [ ] **J3-05** | Ajouter régularisation si nécessaire
  - **Description** :
    - Weight decay (déjà dans optimizer)
    - Label smoothing (optionnel)
    - Dropout (optionnel, peu utilisé dans ResNet)
  - **Labels** : `régularisation`, `configuration`
  - **Priorité** : 🟢 Basse

---

### 🏋️ Entraînement du teacher

- [ ] **J3-06** | Lancer l'entraînement ResNet-18
  - **Description** : Entraîner le modèle sur CIFAR-10 train
  - **Critère de succès** : Atteindre **≥85% accuracy** sur test
  - **Labels** : `entraînement`, `exécution`
  - **Priorité** : 🔴 Haute

- [ ] **J3-07** | Monitorer l'entraînement
  - **Description** :
    - Logger loss train/val à chaque epoch
    - Logger accuracy train/val
    - Détecter overfitting (gap train/val)
  - **Labels** : `monitoring`, `logs`
  - **Priorité** : 🟡 Moyenne

- [ ] **J3-08** | Sauvegarder les checkpoints
  - **Description** :
    - `resnet18_cifar_best.pth` (meilleure accuracy val)
    - `resnet18_cifar_last.pth` (dernier epoch)
    - Sauvegarder aussi l'optimizer state (pour reprise)
  - **Labels** : `sauvegarde`, `checkpoints`
  - **Priorité** : 🔴 Haute

---

### 📊 Évaluation et mesures

- [ ] **J3-09** | Évaluer l'accuracy finale sur test
  - **Description** : Charger le best checkpoint et calculer l'accuracy sur CIFAR-10 test
  - **Critère de succès** : **≥85%**
  - **Labels** : `évaluation`, `métriques`
  - **Priorité** : 🔴 Haute

- [ ] **J3-10** | Mesurer la latence GPU (FP32)
  - **Description** :
    - Utiliser le benchmark J1
    - Batch = 1, entrée sur GPU
    - Warm-up + mesure (moyenne + p95)
  - **Labels** : `benchmark`, `latence`
  - **Priorité** : 🔴 Haute

- [ ] **J3-11** | Documenter la taille du modèle
  - **Description** :
    - Nombre de paramètres (~11M pour ResNet-18)
    - Taille du fichier checkpoint (MB)
  - **Labels** : `métriques`, `documentation`
  - **Priorité** : 🟢 Basse

---

### 📝 Documentation et suivi

- [ ] **J3-12** | Mettre à jour le tableau comparatif
  - **Description** : Ajouter les résultats dans le tableau (B2 : ResNet-18 CIFAR FP32)
    | ID | Variante | Acc. (%) | Lat. moy. (ms) | Lat. p95 (ms) | Taille (MB) |
    |----|----------|----------|----------------|---------------|-------------|
    | B2 | ResNet-18 CIFAR (FP32) | ... | ... | ... | ... |
  - **Labels** : `documentation`, `résultats`
  - **Priorité** : 🟡 Moyenne

- [ ] **J3-13** | Documenter les hyperparamètres
  - **Description** : Créer une fiche reproductibilité avec tous les paramètres utilisés
  - **Labels** : `documentation`, `reproductibilité`
  - **Priorité** : 🟡 Moyenne

- [ ] **J3-14** | Préparer le teacher pour distillation
  - **Description** :
    - Vérifier que le checkpoint est bien sauvegardé
    - Tester le chargement du modèle
    - S'assurer que le modèle peut générer des soft labels
  - **Labels** : `préparation`, `distillation`
  - **Priorité** : 🟡 Moyenne

---

## ✅ Critères d'acceptation J3

- [ ] ResNet-18 adaptée CIFAR implémentée
- [ ] **Accuracy ≥85%** sur CIFAR-10 test
- [ ] Mesures latence (moyenne + p95) documentées
- [ ] Tableau comparatif mis à jour (ligne B2)
- [ ] Checkpoint teacher prêt pour J4

---

## 📈 Sorties attendues

| Livrable | Statut |
|----------|--------|
| ResNet-18 CIFAR entraînée | ⬜ |
| Accuracy ≥85% atteinte | ⬜ |
| Latence GPU (FP32) mesurée | ⬜ |
| Tableau comparatif (ligne B2) | ⬜ |
| Teacher prêt pour distillation | ⬜ |

---

## 🚨 Points d'attention

- Si accuracy < 85% après 200 epochs :
  - Augmenter les epochs (300)
  - Ajouter augmentations (Cutout, MixUp)
  - Ajuster LR schedule
- ResNet-18 standard (ImageNet) ne convient pas directement à CIFAR (32×32)
- Le teacher doit être robuste car il guidera le student en J4

---

## 📚 Références utiles

- Architecture ResNet-18 CIFAR : première conv 3×3, pas de maxpool
- Hyperparamètres classiques : LR=0.1, SGD momentum=0.9, WD=5e-4
- Accuracy attendue ResNet-18 CIFAR : ~93-95% (avec bonne config)

