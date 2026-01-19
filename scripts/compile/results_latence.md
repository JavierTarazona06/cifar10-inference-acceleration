(.venv) javit@Javit06:~/cifar10-inference-acceleration$ python scripts/compile/benchmark_compile_mobilenetv3.py --mode fp32 --compile-mode reduce-overhead --num-warmup 50 --num-runs 500
Using device: cuda

Loading checkpoint: checkpoints/mobilenetv3/mobilenetv3_best.pt
✓ Weights loaded

== Baseline (uncompiled) ==

============================================================
LATENCY (uncompiled)
============================================================
+-----------+---------+
| Metric    |   Value |
+===========+=========+
| Mean (ms) |  4.0847 |
+-----------+---------+
| Std (ms)  |  0.8271 |
+-----------+---------+
| Min (ms)  |  3.2133 |
+-----------+---------+
| Max (ms)  |  8.2245 |
+-----------+---------+
| p50 (ms)  |  3.8013 |
+-----------+---------+
| p95 (ms)  |  5.5839 |
+-----------+---------+
| p99 (ms)  |  7.3032 |
+-----------+---------+
============================================================

Compiling model with torch.compile (mode=reduce-overhead) ...
First compiled call (includes compile) : 3117.9741 ms

== Compiled model benchmark ==

============================================================
LATENCY (compiled)
============================================================
+-----------+---------+
| Metric    |   Value |
+===========+=========+
| Mean (ms) |  0.6762 |
+-----------+---------+
| Std (ms)  |  0.3595 |
+-----------+---------+
| Min (ms)  |  0.4305 |
+-----------+---------+
| Max (ms)  |  2.6535 |
+-----------+---------+
| p50 (ms)  |  0.5731 |
+-----------+---------+
| p95 (ms)  |  1.8467 |
+-----------+---------+
| p99 (ms)  |  2.1843 |
+-----------+---------+
============================================================

📊 Speedup vs baseline:
  Mean: ×6.04
  p95 : ×3.02