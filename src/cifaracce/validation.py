import json
import numpy as np
import pandas as pd
import torch
import torchvision
from datetime import datetime
from pathlib import Path

from cifaracce.data import BATCH_SIZE_TEST, test_loader
from cifaracce.bench import WARM_UP_ITERS, MEASURE_ITERS, benchmark_latency, example_usage

# ========================================================================
# VALIDATION AND REPRODUCIBILITY MODULE
# ========================================================================
# 
# What does this file do?
# Documents benchmark conditions and verifies that measurements are stable.
#
# Step by step:
# 1. Documents the environment:
#    - PyTorch, CUDA versions
#    - GPU used, configurations
#    - Saves everything to benchmark_conditions.json
# 2. Runs 3 consecutive benchmarks of the same model
# 3. Validates stability:
#    - Calculates the coefficient of variation (CV) between the 3 runs
#    - If CV < 20%: ✓ Reliable measurements
#    - If CV ≥ 20%: ⚠️ Noise/instability present
# 4. Shows summary with results and statistics
#
# Objective: Ensure your measurements are reproducible and not affected
#            by random noise.
# ========================================================================

# -----------------------------
# Conditions for reproducing
# -----------------------------

def show_benchmark_conditions_and_run():
    """
    Documenta las condiciones del benchmark y ejecuta el ejemplo de validación.
    """
    
    print("\n" + "="*60)
    print("CONDITIONS FIXES DE MESURE (reproductibilité)")
    print("="*60)

    # Documenter l'environnement
    benchmark_conditions = {
        "timestamp": datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count(),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }

    # Paramètres de benchmark
    benchmark_params = {
        "batch_size": BATCH_SIZE_TEST,
        "input_shape": (1, 3, 32, 32),  # (batch, channels, height, width)
        "warmup_iterations": WARM_UP_ITERS,
        "measure_iterations": MEASURE_ITERS,
        "model_mode": "eval",
        "grad_disabled": "torch.no_grad()",
        "synchronization": "torch.cuda.synchronize()",
        "precision": "FP32 (float32)",
        "data_preloaded_gpu": True,  # Pas de transferts pendant mesure
    }

    print("\nEnvironnement:")
    for key, value in benchmark_conditions.items():
        print(f"  {key}: {value}")

    print("\nParamètres de benchmark:")
    for key, value in benchmark_params.items():
        print(f"  {key}: {value}")

    # Sauvegarder les conditions dans un fichier pour reproductibilité
    conditions_file = Path('benchmark_conditions.json')
    with open(conditions_file, 'w') as f:
        # Convertir torch.Version en string pour JSON
        benchmark_conditions_serializable = {
            k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
            for k, v in benchmark_conditions.items()
        }
        json.dump({
            "conditions": benchmark_conditions_serializable,
            "params": benchmark_params
        }, f, indent=2)

    print(f"\nConditions sauvegardées dans {conditions_file}")

    # Ejecutar el benchmark de ejemplo
    print("\n" + "="*60)
    print("EJECUCIÓN DEL BENCHMARK")
    print("="*60)
    
    # Ejecutar example_usage que hace el benchmark
    example_usage(test_loader)
    
    # Leer los resultados para el resumen
    print("\n" + "="*60)
    print("Résumé")
    print("="*60)

    # Leer CSV con los resultados
    df = pd.read_csv('results_j1_validation.csv')
    
    if len(df) > 0:
        # Calcular estadísticas de los runs
        stats_mean = df['lat_mean_ms'].iloc[-1]
        stats_p95 = df['lat_p95_ms'].iloc[-1]
        
        print("\nBenchmark produit mean + p95 lisibles")
        print(f"  - Mean: {stats_mean:.4f} ms")
        print(f"  - P95:  {stats_p95:.4f} ms")

        # Calcular varianza entre runs
        p95_values = df['lat_p95_ms'].values
        p95_mean = np.mean(p95_values)
        p95_std = np.std(p95_values)
        p95_cv = (p95_std / p95_mean) * 100  # Coefficient de variation

        print("\nVariance acceptable (< 20% entre runs)")
        if p95_cv < 20:
            print(f"  - CV: {p95_cv:.2f}% ACCEPTÉ")
        else:
            print(f"  - CV: {p95_cv:.2f}% ATTENTION (> 20%)")

    print("\nCSV bien formé (colonnes header)")
    print(f"  - Fichier: results_j1_validation.csv")
    print(f"  - Colonnes: {list(df.columns)}")
    print(f"  - Rows: {len(df)}")
    print(f"  - Aperçu:")
    print(df.to_string(index=False))

    print("\n" + "="*60)
    print("="*60)


if __name__ == "__main__":
    show_benchmark_conditions_and_run()