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
| Mean (ms) |  4.1922 |
+-----------+---------+
| Std (ms)  |  1.1102 |
+-----------+---------+
| Min (ms)  |  3.1892 |
+-----------+---------+
| Max (ms)  |  9.6606 |
+-----------+---------+
| p50 (ms)  |  3.7764 |
+-----------+---------+
| p95 (ms)  |  6.2514 |
+-----------+---------+
| p99 (ms)  |  9.1509 |
+-----------+---------+
============================================================

Compiling model with torch.compile (mode=reduce-overhead) ...

============================================================
J4-06 | COMPILATION TIME MEASUREMENT
============================================================
First compiled call (includes compilation overhead): 3601.9170 ms
Note: This includes both JIT compilation and first inference.
============================================================

== J4-07 | POST-COMPILATION LATENCY BENCHMARK ==

============================================================
LATENCY (compiled, FP32)
============================================================
+-----------+---------+
| Metric    |   Value |
+===========+=========+
| Mean (ms) |  0.9109 |
+-----------+---------+
| Std (ms)  |  0.3437 |
+-----------+---------+
| Min (ms)  |  0.5196 |
+-----------+---------+
| Max (ms)  |  2.5591 |
+-----------+---------+
| p50 (ms)  |  0.8565 |
+-----------+---------+
| p95 (ms)  |  1.6084 |
+-----------+---------+
| p99 (ms)  |  2.0752 |
+-----------+---------+
============================================================

================================================================================
COMPARISON: Uncompiled vs Compiled (FP32/FP16)
================================================================================
  Metric               Baseline (eager)   Compiled        Speedup     
  ----------           ---------------    ------------    ----------  
  Mean latency (ms)    4.1922             0.9109          ×4.60       
  p95 latency (ms)     6.2514             1.6084          ×3.89       
================================================================================

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------


(.venv) javit@Javit06:~/cifar10-inference-acceleration$ python scripts/compile/benchmark_compile_mobilenetv3.py --mode fp16 --compile-mode reduce-overhead --num-warmup 50 --num-runs 500
Using device: cuda

Loading checkpoint: checkpoints/mobilenetv3/mobilenetv3_best.pt
✓ Weights loaded

== Baseline (uncompiled) ==
/home/javit/cifar10-inference-acceleration/scripts/compile/benchmark_compile_mobilenetv3.py:37: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(dtype=torch.float16):

============================================================
LATENCY (uncompiled)
============================================================
+-----------+---------+
| Metric    |   Value |
+===========+=========+
| Mean (ms) |  4.874  |
+-----------+---------+
| Std (ms)  |  0.5902 |
+-----------+---------+
| Min (ms)  |  4.1088 |
+-----------+---------+
| Max (ms)  |  8.4955 |
+-----------+---------+
| p50 (ms)  |  4.7157 |
+-----------+---------+
| p95 (ms)  |  6.0173 |
+-----------+---------+
| p99 (ms)  |  7.0557 |
+-----------+---------+
============================================================

Compiling model with torch.compile (mode=reduce-overhead) ...

============================================================
J4-06 | COMPILATION TIME MEASUREMENT
============================================================
First compiled call (includes compilation overhead): 13669.8672 ms
Note: This includes both JIT compilation and first inference.
============================================================

== J4-07 | POST-COMPILATION LATENCY BENCHMARK ==
/home/javit/cifar10-inference-acceleration/scripts/compile/benchmark_compile_mobilenetv3.py:37: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(dtype=torch.float16):
/home/javit/cifar10-inference-acceleration/scripts/compile/benchmark_compile_mobilenetv3.py:37: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(dtype=torch.float16):

============================================================
LATENCY (compiled, FP16)
============================================================
+-----------+---------+
| Metric    |   Value |
+===========+=========+
| Mean (ms) |  0.56   |
+-----------+---------+
| Std (ms)  |  0.0728 |
+-----------+---------+
| Min (ms)  |  0.4531 |
+-----------+---------+
| Max (ms)  |  1.0529 |
+-----------+---------+
| p50 (ms)  |  0.5334 |
+-----------+---------+
| p95 (ms)  |  0.7128 |
+-----------+---------+
| p99 (ms)  |  0.779  |
+-----------+---------+
============================================================

================================================================================
COMPARISON: Uncompiled vs Compiled (FP32/FP16)
================================================================================
  Metric               Baseline (eager)   Compiled        Speedup     
  ----------           ---------------    ------------    ----------  
  Mean latency (ms)    4.8740             0.5600          ×8.70       
  p95 latency (ms)     6.0173             0.7128          ×8.44       
================================================================================