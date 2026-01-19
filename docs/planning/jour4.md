# Jour 4 — Optimisations GPU + Itération I3 (et I4 si besoin)

## 🎯 Objectif
Réduire la latence sans dégrader la précision en dessous de 85%.

---

## 📋 Tâches

### ⚡ I3.1 — Inférence FP16 (autocast)

- [X] **J4-01** | Implémenter l'inférence en FP16
  - **Description** :
    ```python
    with torch.cuda.amp.autocast(dtype=torch.float16):
        output = model(input)
    ```
  - **Labels** : `optimisation`, `FP16`
  - **Priorité** : 🔴 Haute

- [X] **J4-02** | Mesurer la latence FP16 (modèle léger)
  - **Description** :
    - Appliquer sur MobileNetV3/ShuffleNet (B1)
    - Benchmark : warm-up + mesure (moyenne + p95)
    - Comparer avec baseline FP32
  - **Labels** : `benchmark`, `latence`
  - **Priorité** : 🔴 Haute

- [X] **J4-03** | Vérifier l'accuracy en FP16
  - **Description** : S'assurer que l'accuracy ne chute pas significativement (< 0.5%)
  - **Labels** : `validation`, `accuracy`
  - **Priorité** : 🔴 Haute

- [X] **J4-04** | Mesurer la latence FP16 (ResNet-18)
  - **Description** : Appliquer le même test sur le teacher pour comparaison
  - **Labels** : `benchmark`, `latence`
  - **Priorité** : 🟡 Moyenne

---

### 🔧 I3.2 — torch.compile (si stable)

- [X] **J4-05** | Tester torch.compile sur le modèle léger
  - **Description** :
    ```python
    model_compiled = torch.compile(model, mode="reduce-overhead")
    ```
    - Mode recommandé pour batch=1 : `"reduce-overhead"` ou `"max-autotune"`
  - **Labels** : `optimisation`, `compile`
  - **Priorité** : 🟡 Moyenne

- [X] **J4-06** | Mesurer le temps de compilation
  - **Description** :
    - Noter le temps de première exécution (compilation)
    - Ce temps est **hors métrique** de latence
  - **Labels** : `benchmark`, `documentation`
  - **Priorité** : 🟢 Basse

- [X] **J4-07** | Mesurer la latence post-compilation
  - **Description** :
    - Benchmark après compilation complète
    - Comparer avec FP32 et FP16 sans compile
  - **Labels** : `benchmark`, `latence`
  - **Priorité** : 🟡 Moyenne

- [X] **J4-08** | Combiner FP16 + torch.compile
  - **Description** :
    - Tester la combinaison des deux optimisations
    - Mesurer latence (moyenne + p95)
  - **Labels** : `optimisation`, `benchmark`
  - **Priorité** : 🟡 Moyenne

- [X] **J4-09** | Gérer l'instabilité torch.compile
  - **Description** :
    - Si erreurs ou crashes : documenter et passer en fallback
    - Fallback : FP16 seul ou TorchScript (optionnel)
    - Statut : compilations FP32/FP16 stables sous WSL (chemin sans espaces) ; fallback FP16 seul prêt si régression ultérieure
  - **Labels** : `risque`, `fallback`
  - **Priorité** : 🟢 Basse

---

### 🔄 I3.3 — Format channels_last

- [X] **J4-10** | Convertir le modèle en channels_last
  - **Description** :
    ```python
    model = model.to(memory_format=torch.channels_last)
    input = input.to(memory_format=torch.channels_last)
    ```
  - **Labels** : `optimisation`, `mémoire`
  - **Priorité** : 🟡 Moyenne

- [X] **J4-11** | Mesurer la latence channels_last (modèle léger)
  - **Description** :
    - Tester sur MobileNetV3/ShuffleNet
    - Comparer avec baseline FP32
  - **Labels** : `benchmark`, `latence`
  - **Priorité** : 🟡 Moyenne
  - **Résultat** : MobileNetV3 FP32 et FP16 testés (warmup 10, runs 50) → aucun gain, légère régression en FP32, gain p95 marginal en FP16. Channels_last non retenu.

- [X] **J4-12** | Mesurer la latence channels_last (ResNet-18)
  - **Description** : Tester sur le teacher pour comparaison
  - **Labels** : `benchmark`, `latence`
  - **Priorité** : 🟢 Basse
  - **Résultat** : non poursuivi après constat d'absence de gain sur MobileNetV3; channels_last abandonné.

- [X] **J4-13** | Décider de garder channels_last
  - **Description** :
    - Garder uniquement si amélioration mesurable (>5%)
    - Sinon abandonner cette piste
  - **Labels** : `décision`, `analyse`
  - **Priorité** : 🟡 Moyenne
  - **Décision** : abandonner channels_last (gain < 5%, voire régression).

---

### 🎓 I4 — Distillation (optionnelle)

> ⚠️ **Condition** : Exécuter cette section uniquement si le modèle léger < 85% accuracy

- [X] **J4-14** | Implémenter la perte de distillation
  - **Description** :
    ```python
    # Loss = α * KL(soft_student, soft_teacher) + (1-α) * CE(student, labels)
    # Temperature T = 4-6 typiquement
    loss_kl = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    loss_ce = F.cross_entropy(student_logits, labels)
    loss = alpha * loss_kl + (1 - alpha) * loss_ce
    ```
  - **Labels** : `distillation`, `code`
  - **Priorité** : 🔴 Haute (si nécessaire)
  - **Statut** : Implémenté via util `distillation_loss` (combina KL + CE) dans src/cifaracce/utils/distillation.py (alpha=0.7, T=4 par défaut), prêt pour l'entraînement.

- [X] **J4-15** | Configurer l'entraînement distillation
  - **Description** :
    - Teacher : ResNet-18 (J3) en mode eval, frozen
    - Student : MobileNetV3/ShuffleNet (J2)
    - Hyperparamètres : T=4, α=0.7, LR=0.01
  - **Labels** : `distillation`, `configuration`
  - **Priorité** : 🔴 Haute (si nécessaire)
  - **Statut** : Script de distillation prêt (scripts/distillation/train_distill_mobilenet_j4.py) : teacher ResNet-18 gelé, student MobileNetV3, T=4, α=0.7, LR=0.01.

- [X] **J4-16** | Entraîner le student avec distillation
  - **Description** :
    - Epochs : 100-200
    - Objectif : Student ≥85% accuracy
  - **Labels** : `distillation`, `entraînement`
  - **Priorité** : 🔴 Haute (si nécessaire)

- [X] **J4-17** | Évaluer le student distillé
  - **Description** :
    - Accuracy sur test
    - Latence GPU (FP16)
  - **Labels** : `évaluation`, `métriques`
  - **Priorité** : 🔴 Haute (si nécessaire)

---

### 📊 Synthèse et décision

- [X] **J4-18** | Mettre à jour le tableau comparatif
  - **Description** : Ajouter toutes les variantes testées
    | ID | Variante | Acc. (%) | Lat. moy. (ms) | Lat. p95 (ms) | Taille (MB) |
    |----|----------|----------|----------------|---------------|-------------|
    | O1 | B1 + FP16 | ... | ... | ... | ... |
    | O2 | B1 + FP16 + compile | ... | ... | ... | ... |
    | O3 | B1 + channels_last | ... | ... | ... | ... |
    | D1 | Distillation student (FP16) | ... | ... | ... | ... |
  - **Labels** : `documentation`, `résultats`
  - **Priorité** : 🔴 Haute

- [ ] **J4-19** | Analyser les résultats
  - **Description** :
    - Identifier le gain de chaque optimisation
    - Calculer le speedup vs baseline FP32
    - Vérifier contrainte accuracy ≥85%
  - **Labels** : `analyse`, `décision`
  - **Priorité** : 🔴 Haute

- [ ] **J4-20** | Sélectionner le candidat final
  - **Description** :
    - Choisir le meilleur compromis accuracy/latence
    - Documenter la justification du choix
  - **Labels** : `décision`, `documentation`
  - **Priorité** : 🔴 Haute

---

## ✅ Critères d'acceptation J4

- [X] FP16 testé et mesuré sur les deux modèles
- [X] torch.compile testé (ou documenté si instable)
- [X] channels_last testé (décision prise)
- [X] Distillation réalisée si nécessaire
- [X] Tableau comparatif complet (toutes variantes)
- [ ] **Candidat final sélectionné** (meilleur compromis)

---

## 📈 Sorties attendues

| Livrable | Statut |
|----------|--------|
| Benchmark FP16 complété | ⬜ |
| Benchmark torch.compile complété | ⬜ |
| Benchmark channels_last complété | ⬜ |
| Distillation (si nécessaire) | ⬜ |
| Tableau comparatif mis à jour | ⬜ |
| Candidat final identifié | ⬜ |

---

## 🚨 Points d'attention

- **torch.compile** peut être instable : prévoir un fallback
- **FP16** : vérifier que l'accuracy ne chute pas
- **channels_last** : peut ne pas apporter de gain sur tous les modèles
- **Distillation** : seulement si modèle léger < 85%

---

## 📊 Matrice de décision

| Optimisation | Gain latence attendu | Risque | Priorité |
|--------------|---------------------|--------|----------|
| FP16 | 20-50% | Faible | 🔴 Haute |
| torch.compile | 10-30% | Moyen | 🟡 Moyenne |
| channels_last | 5-15% | Faible | 🟡 Moyenne |
| Distillation | N/A (accuracy) | Moyen | Conditionnelle |

---

## 🔀 Arbre de décision J4

```
Modèle léger (J2) accuracy ?
├── ≥85% → Optimisations I3 seulement
│   └── FP16 → compile → channels_last
└── <85% → Distillation I4 nécessaire
    └── Teacher (J3) → Student → puis I3
```

