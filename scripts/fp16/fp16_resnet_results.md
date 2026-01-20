(.venv) PS C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration> python scripts/fp16/eval_resnet18_fp16.py --num-warmup 100 --num-runs 1000
Using device: cuda

Loading checkpoint: checkpoints\resnet18\resnet18_best.pt
 Teacher weights loaded

== ResNet-18 | FP32 benchmark ==

============================================================
LATENCY STATISTICS (batch=1, FP32)
============================================================
+-----------+---------+
| Metric    |   Value |
+===========+=========+
| Mean (ms) |  2.0093 |
+-----------+---------+
| Std (ms)  |  0.5354 |
+-----------+---------+
| Min (ms)  |  1.5605 |
+-----------+---------+
| Max (ms)  |  5.305  |
+-----------+---------+
| p50 (ms)  |  1.8149 |
+-----------+---------+
| p95 (ms)  |  3.3518 |
+-----------+---------+
| p99 (ms)  |  4.0603 |
+-----------+---------+
============================================================

== ResNet-18 | FP16 benchmark ==
C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration\scripts\fp16\eval_resnet18_fp16.py:43: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(dtype=torch.float16):

============================================================
LATENCY STATISTICS (batch=1, FP16)
============================================================
+-----------+---------+
| Metric    |   Value |
+===========+=========+
| Mean (ms) |  2.5812 |
+-----------+---------+
| Std (ms)  |  0.7248 |
+-----------+---------+
| Min (ms)  |  1.8551 |
+-----------+---------+
| Max (ms)  |  6.2371 |
+-----------+---------+
| p50 (ms)  |  2.2833 |
+-----------+---------+
| p95 (ms)  |  4.6187 |
+-----------+---------+
| p99 (ms)  |  4.9594 |
+-----------+---------+
============================================================

📊 Summary (ResNet-18, batch=1):
  FP32 mean: 2.0093 ms | p95: 3.3518 ms
  FP16 mean: 2.5812 ms | p95: 4.6187 ms
  Speedup (mean): ×0.78
  Speedup (p95):  ×0.73
C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration\scripts\fp16\eval_resnet18_fp16.py:156: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(dtype=torch.float16):

== Accuracy Comparison (ResNet-18) ==
FP32 accuracy: 90.64%
FP16 accuracy: 90.64%
Δ (FP32 - FP16): 0.00 pp
Verdict (threshold 0.50 pp): OK