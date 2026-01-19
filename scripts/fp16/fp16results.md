(.venv) PS C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration> python eval_latency_mobilenetv3.py --num-warmup 100 --num-runs 1000
Using device: cuda

Loading checkpoint: checkpoints\mobilenetv3\mobilenetv3_best.pt
✓ Model weights loaded

== MobileNetV3-Small | FP32 benchmark ==

============================================================
LATENCY STATISTICS (batch=1, FP32)
============================================================
+-----------+---------+
| Metric    |   Value |
+===========+=========+
| Mean (ms) |  3.6177 |
+-----------+---------+
| Std (ms)  |  0.6045 |
+-----------+---------+
| Min (ms)  |  2.8452 |
+-----------+---------+
| Max (ms)  |  9.4268 |
+-----------+---------+
| p50 (ms)  |  3.4879 |
+-----------+---------+
| p95 (ms)  |  4.7224 |
+-----------+---------+
| p99 (ms)  |  5.6238 |
+-----------+---------+
============================================================

== MobileNetV3-Small | FP16 benchmark ==
C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration\eval_latency_mobilenetv3.py:31: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(dtype=torch.float16):

============================================================
LATENCY STATISTICS (batch=1, FP16)
============================================================
+-----------+---------+
| Metric    |   Value |
+===========+=========+
| Mean (ms) |  4.6091 |
+-----------+---------+
| Std (ms)  |  0.6121 |
+-----------+---------+
| Min (ms)  |  3.6152 |
+-----------+---------+
| Max (ms)  |  8.9468 |
+-----------+---------+
| p50 (ms)  |  4.4591 |
+-----------+---------+
| p95 (ms)  |  5.6189 |
+-----------+---------+
| p99 (ms)  |  7.0813 |
+-----------+---------+
============================================================

📊 Summary (MobileNetV3-Small, batch=1):
  FP32 mean: 3.6177 ms | p95: 4.7224 ms
  FP16 mean: 4.6091 ms | p95: 5.6189 ms
  Speedup (mean): ×0.78
  Speedup (p95):  ×0.84



  (.venv) PS C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration> python scripts\fp16\eval_accuracy_fp16_mobilenetv3.py
Using device: cuda

Loading checkpoint: checkpoints\mobilenetv3\mobilenetv3_best.pt
✓ Weights loaded
C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration\scripts\fp16\eval_accuracy_fp16_mobilenetv3.py:38: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(dtype=torch.float16):

== Accuracy Comparison (MobileNetV3-Small) ==
FP32 accuracy: 71.02%
FP16 accuracy: 71.05%
Δ (FP32 - FP16): -0.03 pp
Verdict (threshold 0.50 pp): OK