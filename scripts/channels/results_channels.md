Quick check: channels_last vs NCHW (MobileNetV3-Small, batch=1, warmup=10, runs=50).

FP32 (eager, CUDA):
- Baseline mean 4.2220 ms, p95 5.1492 ms
- channels_last mean 4.5356 ms, p95 5.7790 ms
- Outcome: slight regression (speedup x0.93 mean, x0.89 p95)

FP16 (autocast, CUDA):
- Baseline mean 4.6073 ms, p95 6.7300 ms
- channels_last mean 4.5919 ms, p95 5.8274 ms
- Outcome: essentially flat on mean, modest p95 improvement (x1.15) but under 5% target on mean

Conclusion: channels_last does not provide a measurable gain on this GPU/model; keep NCHW as default and skip further channels_last work unless hardware changes.
