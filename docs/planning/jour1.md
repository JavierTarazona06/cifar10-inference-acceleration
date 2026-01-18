# Jour 1 - Démarrage + Itération I0
## Objectif : Environnement propre + Benchmark latence GPU fiable

---

## 📋 Liste des tâches

### Card 1: Configuration de l'environnement
**Statut:** À faire  
**Priorité:** 🔴 Critique  
**Assigné à:** Javier  
**Due:** J1 matin

- [ ] Vérifier/installer PyTorch (dernière version stable)
- [ ] Vérifier/installer torchvision
- [ ] Vérifier CUDA et drivers GPU (version compatible)
- [ ] Vérifier disponibilité de `torch.compile` (PyTorch >= 2.0)
- [ ] Documenter versions utilisées (PyTorch, CUDA, GPU model)

**Notes:**
```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

### Card 2: Préparation des données CIFAR-10
**Statut:** À faire  
**Priorité:** 🔴 Critique  
**Assigné à:** Javier  
**Due:** J1 matin/midi

- [ ] Implémenter chargement CIFAR-10 (train + test)
- [ ] Définir transforms **train** (normalisation + augmentations)
- [ ] Définir transforms **test** (normalisation uniquement)
- [ ] Vérifier shapes et classes (10 classes, images 32×32)
- [ ] Préparer DataLoader test avec batch_size=1 (pour benchmark)

**Notes:**
- Normalisation : mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
- Augmentations recommandées : RandomCrop(32, padding=4), RandomHorizontalFlip, RandomRotation

---

### Card 3: Implémentation du benchmark de latence GPU
**Statut:** À faire  
**Priorité:** 🔴 Critique  
**Assigné à:** Javier  
**Due:** J1 midi/après-midi

- [ ] Créer fonction `benchmark_latency(model, dataloader, warmup_iters=50, measure_iters=500)`
- [ ] **Entrées précalculées sur GPU** (avant boucle, pas de transfert à chaque itération)
- [ ] **Warm-up** sans mesure (50–200 itérations)
- [ ] **Synchronisation GPU** (`torch.cuda.synchronize()`) avant et après
- [ ] Boucle de mesure avec `torch.cuda.Event` ou `time.perf_counter()`
- [ ] Calcul de : **moyenne**, **p95** (95e percentile), std, min, max
- [ ] Sauvegarde résultats dans **CSV** (colonnes : variant, accuracy, lat_mean, lat_p95, timestamp)

**Protocole détaillé:**
```python
# Pseudo-code
model.eval()
torch.no_grad()

# Warm-up
for i in range(warmup_iters):
    output = model(input)  # input déjà sur GPU
torch.cuda.synchronize()

# Mesure
times = []
for i in range(measure_iters):
    t0 = perf_counter()
    output = model(input)
    torch.cuda.synchronize()
    t1 = perf_counter()
    times.append((t1 - t0) * 1000)  # en ms

stats = {mean, p95, std}
```

---

### Card 4: Validation du benchmark (stabilité & reproductibilité)
**Statut:** À faire  
**Priorité:** 🟠 Haute  
**Assigné à:** Javier  
**Due:** J1 après-midi

- [ ] Exécuter benchmark **3 fois** sur un modèle dummy (ex. ResNet-18 random weights)
- [ ] Vérifier variance des résultats (p95 ne doit pas osciller > 10%)
- [ ] Documenter conditions fixes : GPU model, CUDA version, batch size, input shape
- [ ] Vérifier que `model.eval()` + `torch.no_grad()` sont bien appliqués
- [ ] Vérifier pas de transferts GPU/CPU dans la boucle
- [ ] Générer 1 fichier CSV de résultats test

**Critères d'acceptation :**
- Benchmark produit mean + p95 lisibles
- Variance acceptable (< 10% entre runs)
- CSV bien formé (colonnes header)

---

### Card 5: Documentation du protocole
**Statut:** À faire  
**Priorité:** 🟡 Moyenne  
**Assigné à:** Javier  
**Due:** J1 fin

- [ ] Écrire README ou section **"Protocole de mesure"** dans notebook
- [ ] Documenter :
  - Warm-up iterations = ?
  - Measure iterations = ?
  - GPU used = ?
  - CUDA synchronization method
  - Batch size = 1
  - Precision = FP32 (par défaut J1)
- [ ] Rendre reproductible : versions loggées, seed (optionnel)

---

## ✅ Critères d'acceptation J1

- [ ] Script/notebook exécutable avec toutes les étapes (env + data + benchmark)
- [ ] Benchmark latence produit **mean + p95** en ms
- [ ] Fichier **CSV résultats** généré
- [ ] **3 exécutions consécutives** montrent variance acceptable
- [ ] **Protocole documenté** (dans README ou notebook)

---

## 📊 Fichiers attendus en sortie

- `bench_latency.py` ou section notebook avec fonction benchmark
- `results_j1_validation.csv` (résultats des 3 runs de validation)
- `README_PROTOCOLE.md` ou équivalent (documentation)
