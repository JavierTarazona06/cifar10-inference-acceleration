# Jour 2 — Baseline vitesse + Itération I1

## 🎯 Objectif
Entraîner un modèle léger from-scratch et mesurer sa latence.

---

## 📋 Tâches

### 🔧 Configuration du modèle léger

- [X] **J2-01** | Choisir l'architecture légère
  - **Description** : Sélectionner entre MobileNetV3-Small et ShuffleNetV2 pour le baseline vitesse
  - **Labels** : `architecture`, `décision`
  - **Priorité** : 🔴 Haute

- [X] **J2-02** | Adapter le modèle pour CIFAR-10
  - **Description** : 
    - Instancier avec `weights=None` (pas de poids ImageNet)
    - Modifier la tête de sortie pour **10 classes**
    - Adapter la première couche conv si nécessaire (images 32×32)
  - **Labels** : `code`, `modèle`
  - **Priorité** : 🔴 Haute

- [X] **J2-03** | Configurer la gestion du device
  - **Description** : Assurer le transfert propre du modèle et des données sur GPU
  - **Labels** : `code`, `GPU`
  - **Priorité** : 🟡 Moyenne

---

### 🏋️ Entraînement

- [X] **J2-04** | Définir les hyperparamètres d'entraînement
  - **Description** :
    - Learning rate initial
    - Scheduler (CosineAnnealing, StepLR, etc.)
    - Optimizer (SGD+momentum ou AdamW)
    - Nombre d'epochs (objectif : ≥80% rapidement)
    - Batch size
  - **Labels** : `hyperparamètres`, `configuration`
  - **Priorité** : 🔴 Haute

- [X] **J2-05** | Configurer les augmentations de données
  - **Description** :
    - RandomCrop avec padding
    - RandomHorizontalFlip
    - Normalisation CIFAR-10
    - (Optionnel) Cutout, AutoAugment. They can help, not yet impelmented
  - **Labels** : `data`, `augmentation`
  - **Priorité** : 🟡 Moyenne

- [X] **J2-06** | Lancer l'entraînement du modèle léger
  - **Description** : Entraîner MobileNetV3-Small sur CIFAR-10 train
  - **Critère de succès** : Atteindre ≥80%, viser ≥85%
  - **Labels** : `entraînement`, `exécution`
  - **Priorité** : 🔴 Haute

- [X] **J2-07** | Sauvegarder les checkpoints
  - **Description** :
    - Sauvegarder le meilleur modèle (best accuracy)
    - Sauvegarder le dernier modèle
    - Logger les métriques (loss train/test, accuracy train/test)
  - **Labels** : `sauvegarde`, `logs`
  - **Priorité** : 🟡 Moyenne

---

### 📊 Évaluation et mesures

- [ ] **J2-08** | Évaluer l'accuracy sur le jeu de test
  - **Description** : Calculer l'accuracy finale sur CIFAR-10 test
  - **Critère de succès** : Documenter le résultat même si < 85%
  - **Labels** : `évaluation`, `métriques`
  - **Priorité** : 🔴 Haute

- [ ] **J2-09** | Mesurer la latence GPU (FP32)
  - **Description** :
    - Utiliser le benchmark développé en J1
    - Batch = 1, entrée sur GPU
    - Warm-up + mesure (moyenne + p95)
  - **Labels** : `benchmark`, `latence`
  - **Priorité** : 🔴 Haute

- [ ] **J2-10** | Documenter la taille du modèle
  - **Description** :
    - Nombre de paramètres
    - Taille du fichier checkpoint (MB)
  - **Labels** : `métriques`, `documentation`
  - **Priorité** : 🟢 Basse

---

### 📝 Documentation et suivi

- [ ] **J2-11** | Remplir la première ligne du tableau comparatif
  - **Description** : Ajouter les résultats dans le tableau (B1 : MobileNetV3/ShuffleNet FP32)
    | ID | Variante | Acc. (%) | Lat. moy. (ms) | Lat. p95 (ms) | Taille (MB) |
    |----|----------|----------|----------------|---------------|-------------|
    | B1 | ... | ... | ... | ... | ... |
  - **Labels** : `documentation`, `résultats`
  - **Priorité** : 🟡 Moyenne

- [ ] **J2-12** | Logger les hyperparamètres utilisés
  - **Description** : Documenter tous les choix pour reproductibilité
  - **Labels** : `documentation`, `reproductibilité`
  - **Priorité** : 🟡 Moyenne

---

## ✅ Critères d'acceptation J2

- [ ] Au moins un modèle léger entraîné from-scratch
- [ ] Résultats mesurés : accuracy + latence moyenne + latence p95
- [ ] Première ligne du tableau comparatif remplie
- [ ] Checkpoints et logs sauvegardés

---

## 📈 Sorties attendues

| Livrable | Statut |
|----------|--------|
| Modèle léger entraîné | ⬜ |
| Accuracy sur test documentée | ⬜ |
| Latence GPU (FP32) mesurée | ⬜ |
| Tableau comparatif (1ère ligne) | ⬜ |

---

## 🚨 Points d'attention

- Si accuracy < 80% : ajuster LR, augmentations, ou nombre d'epochs
- Si accuracy entre 80-85% : noter pour potentielle distillation J4
- Bien utiliser `model.eval()` et `torch.no_grad()` pour les mesures

