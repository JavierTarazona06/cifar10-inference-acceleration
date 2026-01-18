import csv
import numpy as np

import torch
import torchvision
from torchvision import datasets, transforms

from time import perf_counter
from datetime import datetime
from pathlib import Path

# ========================================================================
# BENCHMARK LATENCY MODULE
# ========================================================================
# 
# What does this file do?
# Measures how long the model takes to process one image (GPU latency).
#
# Step by step:
# 1. Preloads images on GPU (to avoid measuring CPU→GPU transfers)
# 2. Warm-up: Runs the model 50 times without measuring (heats up GPU)
# 3. Measurement: Runs 500 times timing each inference with precision
# 4. Calculates statistics: mean, P95, standard deviation, min, max
# 5. Saves results to CSV
#
# Key function: benchmark_latency() gives you precise inference times.
# ========================================================================

WARM_UP_ITERS = 50
MEASURE_ITERS = 500

# ------------------------------- #
# Benchmark Latency
# ------------------------------- #

def benchmark_latency(model, dataloader, warmup_iters=WARM_UP_ITERS, measure_iters=MEASURE_ITERS, device='cuda'):
    """
    Benchmark latency d'inférence GPU pour batch_size=1, suelement 1 image.

    Args:
        model: PyTorch model en mode eval
        dataloader: DataLoader avec batch_size=1
        warmup_iters: nombre d'itérations de warm-up (default 50)
        measure_iters: nombre d'itérations de mesure (default 500)
        device: 'cuda' (GPU) ou 'cpu'

    Returns:
        dict avec statistiques de latence (mean, p95, std, min, max en ms)
    """

    # Check eval mode and device
    model.eval()
    model.to(device)

    # Send data to be stored in GPU
    print(f"[Benchmark] Préparation des données sur {device}...")
    gpu_data = []
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        gpu_data.append((images, labels))
        if i >= max(warmup_iters, measure_iters) - 1:
            break

    print(f"[Benchmark] {len(gpu_data)} images préchargées sur GPU")

    # WARM-UP (no mesure) to avoid bias for overcharging
    print(f"[Benchmark] Warm-up: {warmup_iters} itérations...")
    with torch.no_grad():
        for i in range(min(warmup_iters, len(gpu_data))):
            images, _ = gpu_data[i]
            _ = model(images)

    torch.cuda.synchronize()  # GPU Synchronisation after warm-up
    print("[Benchmark] Warm-up terminé")

    # MESURE (avec chrono)
    print(f"[Benchmark] Mesure: {measure_iters} itérations...")
    times = []

    with torch.no_grad():
        for i in range(min(measure_iters, len(gpu_data))):
            images, _ = gpu_data[i]

            # Chrono précis
            torch.cuda.synchronize()  # Sync before
            t0 = perf_counter()

            output = model(images)

            torch.cuda.synchronize()  # Sync after
            t1 = perf_counter()

            # Temps en ms
            elapsed_ms = (t1 - t0) * 1000
            times.append(elapsed_ms)

    # STATISTIQUES
    times = np.array(times)
    stats = {
        'mean': float(np.mean(times)),
        'p95': float(np.percentile(times, 95)),
        'p50': float(np.percentile(times, 50)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'count': len(times)
    }

    print(f"[Benchmark] Résultats Latence:")
    print(f"  - Mean: {stats['mean']:.4f} ms")
    print(f"  - P95:  {stats['p95']:.4f} ms")
    print(f"  - P50:  {stats['p50']:.4f} ms")
    print(f"  - Std:  {stats['std']:.4f} ms")
    print(f"  - Min:  {stats['min']:.4f} ms")
    print(f"  - Max:  {stats['max']:.4f} ms")

    return stats, times

def save_benchmark_results(results_list, filename='benchmark_results.csv'):
    """
    Stocke les résultats de benchmark en CSV.

    Args:
        results_list: liste de dict avec {variant, accuracy, lat_mean, lat_p95, ...}
        filename: nom du fichier CSV
    """
    filepath = Path(filename)

    # Headers
    fieldnames = ['timestamp', 'variant', 'accuracy', 'lat_mean_ms', 'lat_p95_ms',
                  'lat_p50_ms', 'lat_std_ms', 'lat_min_ms', 'lat_max_ms', 'measure_iters']

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)

    print(f"\nRésultats stockées dans {filepath}")


# ========================
# Example
# ========================

def example_usage(test_loader):

    # Créer un modèle dummy pour tester
    dummy_model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AvgPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(16 * 16 * 16, 10)
    ).cuda()

    # Lancer 3 benchmarks (validation stabilité)
    print("="*60)
    print("VALIDATION STABILITÉ DU BENCHMARK")
    print("="*60)

    results_list = []
    for run in range(3):
        print(f"\n[Run {run+1}/3]")
        print("-" * 60)

        stats, times = benchmark_latency(
            dummy_model,
            test_loader,  # From data preparation
            warmup_iters= WARM_UP_ITERS,
            measure_iters= MEASURE_ITERS,
            device='cuda'
        )

        # Store results
        results_list.append({
            'timestamp': datetime.now().isoformat(),
            'variant': 'dummy_model_fp32',
            'accuracy': 0.0,  # Placeholder
            'lat_mean_ms': stats['mean'],
            'lat_p95_ms': stats['p95'],
            'lat_p50_ms': stats['p50'],
            'lat_std_ms': stats['std'],
            'lat_min_ms': stats['min'],
            'lat_max_ms': stats['max'],
            'measure_iters': stats['count']
        })

    # Store in CSV file
    save_benchmark_results(results_list, 'results_j1_validation.csv')

    # Variance
    print("\n" + "="*60)
    print("RÉSUMÉ VARIANCE (< 10% acceptable)")
    print("="*60)
    p95_values = [r['lat_p95_ms'] for r in results_list]
    p95_mean = np.mean(p95_values)
    p95_std = np.std(p95_values)
    p95_cv = (p95_std / p95_mean) * 100  # Coefficient de variation

    print(f"P95 Mean: {p95_mean:.4f} ms")
    print(f"P95 Std:  {p95_std:.4f} ms")
    print(f"P95 CV:   {p95_cv:.2f}% {'approuvé' if p95_cv < 20 else 'X'}")