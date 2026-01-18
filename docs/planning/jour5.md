# Jour 5 — Consolidation + Itération I5

## 🎯 Objectif
Finaliser les livrables, assurer la reproductibilité et rédiger la narration de la démarche.

---

## 📋 Tâches

### 🧹 Nettoyage et organisation du code

- [ ] **J5-01** | Structurer les scripts Python
  - **Description** :
    - `train.py` — Entraînement des modèles
    - `eval.py` — Évaluation accuracy sur test
    - `bench.py` — Benchmark latence GPU
    - `distill.py` — Distillation (si utilisée)
  - **Labels** : `code`, `organisation`
  - **Priorité** : 🔴 Haute

- [ ] **J5-02** | Nettoyer les notebooks
  - **Description** :
    - Supprimer les cellules de debug/test
    - Ajouter des commentaires explicatifs
    - Structurer en sections claires
    - Vérifier que les notebooks s'exécutent de bout en bout
  - **Labels** : `code`, `documentation`
  - **Priorité** : 🟡 Moyenne

- [ ] **J5-03** | Créer un fichier de configuration
  - **Description** :
    - `config.py` ou `config.yaml` avec tous les hyperparamètres
    - Chemins des checkpoints
    - Paramètres du benchmark
  - **Labels** : `configuration`, `reproductibilité`
  - **Priorité** : 🟡 Moyenne

---

### 🔒 Reproductibilité

- [ ] **J5-04** | Figer les seeds
  - **Description** :
    ```python
    import torch
    import numpy as np
    import random
    
    def set_seed(seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    ```
  - **Labels** : `reproductibilité`, `code`
  - **Priorité** : 🔴 Haute

- [ ] **J5-05** | Logger les versions des dépendances
  - **Description** :
    - PyTorch version
    - torchvision version
    - CUDA version
    - Python version
    - GPU utilisé (nom, driver)
    - Créer `requirements.txt` ou `environment.yml`
  - **Labels** : `reproductibilité`, `documentation`
  - **Priorité** : 🔴 Haute

- [ ] **J5-06** | Documenter les chemins des checkpoints
  - **Description** :
    - Liste tous les checkpoints sauvegardés
    - Préciser lequel est le modèle final
    - Format : `checkpoints/model_name_epoch_accuracy.pth`
  - **Labels** : `documentation`, `organisation`
  - **Priorité** : 🟡 Moyenne

---

### 📏 Mesures finales

- [ ] **J5-07** | Effectuer 3 runs de benchmark pour le candidat final
  - **Description** :
    - Exécuter le benchmark 3 fois
    - Calculer moyenne et écart-type des latences
    - Vérifier la stabilité des mesures
  - **Labels** : `benchmark`, `validation`
  - **Priorité** : 🔴 Haute

- [ ] **J5-08** | Effectuer 3 runs de benchmark pour le baseline
  - **Description** :
    - Même protocole sur B1 (modèle léger FP32)
    - Permet de calculer le speedup final
  - **Labels** : `benchmark`, `validation`
  - **Priorité** : 🟡 Moyenne

- [ ] **J5-09** | Vérifier l'accuracy finale
  - **Description** :
    - Recharger le checkpoint final
    - Évaluer sur CIFAR-10 test
    - Confirmer ≥85%
  - **Labels** : `validation`, `accuracy`
  - **Priorité** : 🔴 Haute

---

### 📊 Tableau final et analyse

- [ ] **J5-10** | Compléter le tableau de résultats final
  - **Description** :
    | ID | Variante | Acc. (%) | Lat. moy. (ms) | Lat. p95 (ms) | Taille (MB) | Speedup |
    |----|----------|----------|----------------|---------------|-------------|---------|
    | B1 | MobileNetV3/ShuffleNet (FP32) | ... | ... | ... | ... | 1.0× |
    | B2 | ResNet-18 CIFAR (FP32) | ... | ... | ... | ... | ... |
    | O1 | B1 + FP16 | ... | ... | ... | ... | ... |
    | O2 | B1 + FP16 + compile | ... | ... | ... | ... | ... |
    | O3 | B1 + channels_last | ... | ... | ... | ... | ... |
    | D1 | Distillation student (FP16) | ... | ... | ... | ... | ... |
    | **F** | **Candidat final** | ... | ... | ... | ... | ... |
  - **Labels** : `documentation`, `résultats`
  - **Priorité** : 🔴 Haute

- [ ] **J5-11** | Calculer les speedups
  - **Description** :
    - Speedup = Latence_baseline / Latence_variante
    - Référence : B1 FP32 (baseline vitesse)
  - **Labels** : `analyse`, `métriques`
  - **Priorité** : 🟡 Moyenne

- [ ] **J5-12** | Rédiger l'analyse des résultats
  - **Description** :
    - Ce qui a été tenté
    - Ce qui a marché / pas marché
    - Justification du choix final
    - Limites et perspectives
  - **Labels** : `documentation`, `analyse`
  - **Priorité** : 🔴 Haute

---

### 📝 Rapport final

- [ ] **J5-13** | Rédiger l'introduction du rapport
  - **Description** :
    - Contexte du projet
    - Objectifs (accuracy ≥85%, latence minimale)
    - Contraintes (CIFAR-10 only, GPU batch=1)
  - **Labels** : `rapport`, `rédaction`
  - **Priorité** : 🟡 Moyenne

- [ ] **J5-14** | Décrire la méthodologie
  - **Description** :
    - Démarche incrémentale : baseline → optimisation → mesure
    - Protocole de benchmark (warm-up, sync, stats)
    - Architectures testées
  - **Labels** : `rapport`, `rédaction`
  - **Priorité** : 🔴 Haute

- [ ] **J5-15** | Présenter les résultats
  - **Description** :
    - Tableau comparatif
    - Graphiques (optionnel) : barplot latence, scatter accuracy vs latence
    - Analyse des gains
  - **Labels** : `rapport`, `visualisation`
  - **Priorité** : 🔴 Haute

- [ ] **J5-16** | Rédiger la conclusion
  - **Description** :
    - Résumé des résultats
    - Modèle final retenu et pourquoi
    - Améliorations possibles
  - **Labels** : `rapport`, `rédaction`
  - **Priorité** : 🟡 Moyenne

---

### 📦 Livrables

- [ ] **J5-17** | Préparer le package de livraison
  - **Description** :
    ```
    projet/
    ├── README.md           # Instructions d'utilisation
    ├── requirements.txt    # Dépendances
    ├── config.py           # Configuration
    ├── train.py            # Entraînement
    ├── eval.py             # Évaluation
    ├── bench.py            # Benchmark
    ├── checkpoints/
    │   └── final_model.pth # Modèle final
    ├── results/
    │   └── results.csv     # Résultats benchmark
    └── rapport.pdf         # Rapport final
    ```
  - **Labels** : `livrable`, `organisation`
  - **Priorité** : 🔴 Haute

- [ ] **J5-18** | Rédiger le README
  - **Description** :
    - Installation des dépendances
    - Comment entraîner un modèle
    - Comment évaluer l'accuracy
    - Comment lancer le benchmark
    - Comment reproduire les résultats
  - **Labels** : `documentation`, `livrable`
  - **Priorité** : 🔴 Haute

- [ ] **J5-19** | Exporter le modèle final
  - **Description** :
    - Sauvegarder `state_dict` propre
    - (Optionnel) Export TorchScript pour déploiement
    - Documenter le format et le chargement
  - **Labels** : `livrable`, `modèle`
  - **Priorité** : 🟡 Moyenne

- [ ] **J5-20** | Vérification finale
  - **Description** :
    - Tester le chargement du modèle final
    - Vérifier que le benchmark fonctionne
    - Relire le rapport
    - S'assurer que tout est reproductible
  - **Labels** : `validation`, `qualité`
  - **Priorité** : 🔴 Haute

---

## ✅ Critères d'acceptation J5

- [ ] Code nettoyé et organisé (scripts ou notebooks)
- [ ] Reproductibilité assurée (seeds, versions, configs)
- [ ] 3 runs de mesure finale effectués
- [ ] Tableau de résultats complet avec speedups
- [ ] Analyse écrite (ce qui a marché/pas marché)
- [ ] Rapport finalisé
- [ ] **Modèle final ≥85% accuracy + meilleure latence**
- [ ] Package de livraison prêt

---

## 📈 Sorties attendues

| Livrable | Statut |
|----------|--------|
| Code nettoyé | ⬜ |
| requirements.txt | ⬜ |
| Mesures finales (3 runs) | ⬜ |
| Tableau comparatif final | ⬜ |
| Analyse des résultats | ⬜ |
| Rapport complet | ⬜ |
| README.md | ⬜ |
| Modèle final (.pth) | ⬜ |
| results.csv | ⬜ |

---

## 🚨 Points d'attention

- **Deadline** : tout doit être finalisé aujourd'hui
- Ne pas oublier de **vérifier l'accuracy** après rechargement du modèle
- Les **3 runs** de benchmark sont essentiels pour la crédibilité
- Le **README** doit permettre à quelqu'un d'autre de reproduire les résultats

---

## 📋 Checklist de livraison

```
[ ] Le modèle final atteint ≥85% accuracy
[ ] La latence est la meilleure parmi les variantes
[ ] Le code s'exécute sans erreur
[ ] Les résultats sont reproductibles
[ ] Le rapport est complet et clair
[ ] Le README explique comment utiliser le projet
[ ] Tous les fichiers sont présents dans le package
```

