(.venv) PS C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration> python scripts/distillation/train_distill_mobilenet_j4.py --epochs 120 --lr 0.01 --alpha 0.7 --temperature 4.0
Using device: cuda
C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration\scripts\distillation\train_distill_mobilenet_j4.py:38: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(ckpt_path, map_location=device)
Loaded teacher from checkpoints\resnet18\resnet18_best.pt
Epoch 001/120 | TrainLoss 6.3767 (CE 2.0252, KL 8.2417) | TrainAcc  27.24% | TestAcc  11.82% | LR 0.01000
Epoch 002/120 | TrainLoss 4.9209 (CE 1.8224, KL 6.2489) | TrainAcc  44.29% | TestAcc  46.37% | LR 0.00999
Epoch 003/120 | TrainLoss 4.4245 (CE 1.7129, KL 5.5866) | TrainAcc  49.59% | TestAcc  51.98% | LR 0.00998
Epoch 004/120 | TrainLoss 4.1371 (CE 1.6371, KL 5.2085) | TrainAcc  52.28% | TestAcc  54.29% | LR 0.00997
Epoch 005/120 | TrainLoss 3.8814 (CE 1.5693, KL 4.8723) | TrainAcc  54.97% | TestAcc  54.04% | LR 0.00996
Epoch 006/120 | TrainLoss 3.7125 (CE 1.5147, KL 4.6544) | TrainAcc  56.56% | TestAcc  57.70% | LR 0.00994
Epoch 007/120 | TrainLoss 3.5427 (CE 1.4619, KL 4.4345) | TrainAcc  58.20% | TestAcc  57.51% | LR 0.00992
Epoch 008/120 | TrainLoss 3.3781 (CE 1.4126, KL 4.2205) | TrainAcc  59.97% | TestAcc  57.91% | LR 0.00989
Epoch 009/120 | TrainLoss 3.2754 (CE 1.3790, KL 4.0881) | TrainAcc  61.08% | TestAcc  60.55% | LR 0.00986
Epoch 010/120 | TrainLoss 3.1579 (CE 1.3360, KL 3.9388) | TrainAcc  62.25% | TestAcc  63.79% | LR 0.00983
Epoch 011/120 | TrainLoss 3.0484 (CE 1.2997, KL 3.7979) | TrainAcc  63.37% | TestAcc  63.98% | LR 0.00979
Epoch 012/120 | TrainLoss 3.0082 (CE 1.2909, KL 3.7443) | TrainAcc  63.71% | TestAcc  64.75% | LR 0.00976
Epoch 013/120 | TrainLoss 2.9343 (CE 1.2572, KL 3.6531) | TrainAcc  64.61% | TestAcc  64.61% | LR 0.00971
Epoch 014/120 | TrainLoss 2.8843 (CE 1.2424, KL 3.5879) | TrainAcc  64.88% | TestAcc  64.06% | LR 0.00967
Epoch 015/120 | TrainLoss 2.8227 (CE 1.2213, KL 3.5090) | TrainAcc  65.69% | TestAcc  66.31% | LR 0.00962
Epoch 016/120 | TrainLoss 2.7765 (CE 1.1971, KL 3.4533) | TrainAcc  66.08% | TestAcc  62.67% | LR 0.00957
Epoch 017/120 | TrainLoss 2.6965 (CE 1.1735, KL 3.3491) | TrainAcc  66.77% | TestAcc  65.89% | LR 0.00951
Epoch 018/120 | TrainLoss 2.6700 (CE 1.1607, KL 3.3168) | TrainAcc  66.99% | TestAcc  66.42% | LR 0.00946
Epoch 019/120 | TrainLoss 2.6289 (CE 1.1460, KL 3.2645) | TrainAcc  67.34% | TestAcc  68.07% | LR 0.00939
Epoch 020/120 | TrainLoss 2.5986 (CE 1.1378, KL 3.2247) | TrainAcc  67.71% | TestAcc  67.33% | LR 0.00933
Epoch 021/120 | TrainLoss 2.5611 (CE 1.1208, KL 3.1783) | TrainAcc  68.18% | TestAcc  62.30% | LR 0.00926
Epoch 022/120 | TrainLoss 2.5153 (CE 1.1013, KL 3.1213) | TrainAcc  68.61% | TestAcc  70.27% | LR 0.00919
Epoch 023/120 | TrainLoss 2.5104 (CE 1.1025, KL 3.1137) | TrainAcc  68.56% | TestAcc  67.90% | LR 0.00912
Epoch 024/120 | TrainLoss 2.4722 (CE 1.0867, KL 3.0661) | TrainAcc  69.28% | TestAcc  66.33% | LR 0.00905
Epoch 025/120 | TrainLoss 2.4415 (CE 1.0747, KL 3.0273) | TrainAcc  69.30% | TestAcc  65.72% | LR 0.00897
Epoch 026/120 | TrainLoss 2.4176 (CE 1.0703, KL 2.9950) | TrainAcc  69.68% | TestAcc  64.99% | LR 0.00889
Epoch 027/120 | TrainLoss 2.3883 (CE 1.0532, KL 2.9604) | TrainAcc  70.03% | TestAcc  70.82% | LR 0.00880
Epoch 028/120 | TrainLoss 2.3746 (CE 1.0517, KL 2.9415) | TrainAcc  70.09% | TestAcc  68.47% | LR 0.00872
Epoch 029/120 | TrainLoss 2.3502 (CE 1.0382, KL 2.9125) | TrainAcc  70.32% | TestAcc  65.58% | LR 0.00863
Epoch 030/120 | TrainLoss 2.3150 (CE 1.0271, KL 2.8669) | TrainAcc  70.68% | TestAcc  68.78% | LR 0.00854
Epoch 031/120 | TrainLoss 2.2981 (CE 1.0183, KL 2.8466) | TrainAcc  70.69% | TestAcc  71.17% | LR 0.00844
Epoch 032/120 | TrainLoss 2.2757 (CE 1.0140, KL 2.8164) | TrainAcc  71.11% | TestAcc  69.57% | LR 0.00835
Epoch 033/120 | TrainLoss 2.2487 (CE 1.0038, KL 2.7822) | TrainAcc  71.28% | TestAcc  70.96% | LR 0.00825
Epoch 034/120 | TrainLoss 2.2332 (CE 0.9956, KL 2.7636) | TrainAcc  71.57% | TestAcc  70.50% | LR 0.00815
Epoch 035/120 | TrainLoss 2.2179 (CE 0.9891, KL 2.7445) | TrainAcc  71.70% | TestAcc  70.12% | LR 0.00804
Epoch 036/120 | TrainLoss 2.2083 (CE 0.9863, KL 2.7321) | TrainAcc  71.77% | TestAcc  72.43% | LR 0.00794
Epoch 037/120 | TrainLoss 2.1761 (CE 0.9749, KL 2.6909) | TrainAcc  72.18% | TestAcc  70.93% | LR 0.00783
Epoch 038/120 | TrainLoss 2.1555 (CE 0.9710, KL 2.6631) | TrainAcc  72.21% | TestAcc  70.59% | LR 0.00772
Epoch 039/120 | TrainLoss 2.1492 (CE 0.9657, KL 2.6565) | TrainAcc  72.28% | TestAcc  71.12% | LR 0.00761
Epoch 040/120 | TrainLoss 2.1239 (CE 0.9542, KL 2.6251) | TrainAcc  72.70% | TestAcc  67.82% | LR 0.00750
Epoch 041/120 | TrainLoss 2.1113 (CE 0.9494, KL 2.6092) | TrainAcc  72.89% | TestAcc  72.28% | LR 0.00739
Epoch 042/120 | TrainLoss 2.0886 (CE 0.9415, KL 2.5802) | TrainAcc  73.07% | TestAcc  72.84% | LR 0.00727
Epoch 043/120 | TrainLoss 2.0984 (CE 0.9437, KL 2.5932) | TrainAcc  73.16% | TestAcc  69.85% | LR 0.00715
Epoch 044/120 | TrainLoss 2.0908 (CE 0.9459, KL 2.5815) | TrainAcc  72.84% | TestAcc  72.25% | LR 0.00703
Epoch 045/120 | TrainLoss 2.0578 (CE 0.9305, KL 2.5410) | TrainAcc  73.38% | TestAcc  70.40% | LR 0.00691
Epoch 046/120 | TrainLoss 2.0512 (CE 0.9277, KL 2.5326) | TrainAcc  73.24% | TestAcc  71.67% | LR 0.00679
Epoch 047/120 | TrainLoss 2.0472 (CE 0.9290, KL 2.5264) | TrainAcc  73.55% | TestAcc  72.98% | LR 0.00667
Epoch 048/120 | TrainLoss 2.0230 (CE 0.9150, KL 2.4978) | TrainAcc  73.83% | TestAcc  72.97% | LR 0.00655
Epoch 049/120 | TrainLoss 2.0025 (CE 0.9050, KL 2.4729) | TrainAcc  73.97% | TestAcc  73.41% | LR 0.00642
Epoch 050/120 | TrainLoss 2.0169 (CE 0.9150, KL 2.4891) | TrainAcc  73.76% | TestAcc  73.73% | LR 0.00629
Epoch 051/120 | TrainLoss 1.9891 (CE 0.9066, KL 2.4530) | TrainAcc  74.04% | TestAcc  73.55% | LR 0.00617
Epoch 052/120 | TrainLoss 1.9642 (CE 0.8911, KL 2.4241) | TrainAcc  74.39% | TestAcc  73.19% | LR 0.00604
Epoch 053/120 | TrainLoss 1.9663 (CE 0.8898, KL 2.4276) | TrainAcc  74.33% | TestAcc  71.18% | LR 0.00591
Epoch 054/120 | TrainLoss 1.9430 (CE 0.8875, KL 2.3954) | TrainAcc  74.67% | TestAcc  72.47% | LR 0.00578
Epoch 055/120 | TrainLoss 1.9411 (CE 0.8819, KL 2.3950) | TrainAcc  74.74% | TestAcc  74.34% | LR 0.00565
Epoch 056/120 | TrainLoss 1.9401 (CE 0.8875, KL 2.3912) | TrainAcc  74.52% | TestAcc  74.77% | LR 0.00552
Epoch 057/120 | TrainLoss 1.9074 (CE 0.8705, KL 2.3518) | TrainAcc  74.74% | TestAcc  73.06% | LR 0.00539
Epoch 058/120 | TrainLoss 1.9067 (CE 0.8710, KL 2.3506) | TrainAcc  74.96% | TestAcc  73.16% | LR 0.00526
Epoch 059/120 | TrainLoss 1.9046 (CE 0.8757, KL 2.3456) | TrainAcc  75.12% | TestAcc  74.59% | LR 0.00513
Epoch 060/120 | TrainLoss 1.8628 (CE 0.8526, KL 2.2957) | TrainAcc  75.58% | TestAcc  73.76% | LR 0.00500
Epoch 061/120 | TrainLoss 1.8641 (CE 0.8609, KL 2.2941) | TrainAcc  75.16% | TestAcc  72.71% | LR 0.00487
Epoch 062/120 | TrainLoss 1.8624 (CE 0.8525, KL 2.2952) | TrainAcc  75.48% | TestAcc  74.09% | LR 0.00474
Epoch 063/120 | TrainLoss 1.8492 (CE 0.8496, KL 2.2777) | TrainAcc  75.85% | TestAcc  75.68% | LR 0.00461
Epoch 064/120 | TrainLoss 1.8357 (CE 0.8440, KL 2.2607) | TrainAcc  75.77% | TestAcc  75.71% | LR 0.00448
Epoch 065/120 | TrainLoss 1.8110 (CE 0.8341, KL 2.2297) | TrainAcc  76.08% | TestAcc  73.69% | LR 0.00435
Epoch 066/120 | TrainLoss 1.8072 (CE 0.8366, KL 2.2232) | TrainAcc  76.18% | TestAcc  75.11% | LR 0.00422
Epoch 067/120 | TrainLoss 1.7963 (CE 0.8265, KL 2.2119) | TrainAcc  76.10% | TestAcc  74.08% | LR 0.00409
Epoch 068/120 | TrainLoss 1.7866 (CE 0.8242, KL 2.1990) | TrainAcc  76.41% | TestAcc  74.20% | LR 0.00396
Epoch 069/120 | TrainLoss 1.7904 (CE 0.8284, KL 2.2026) | TrainAcc  76.27% | TestAcc  75.30% | LR 0.00383
Epoch 070/120 | TrainLoss 1.7711 (CE 0.8209, KL 2.1783) | TrainAcc  76.53% | TestAcc  75.38% | LR 0.00371
Epoch 071/120 | TrainLoss 1.7712 (CE 0.8194, KL 2.1791) | TrainAcc  76.34% | TestAcc  75.11% | LR 0.00358
Epoch 072/120 | TrainLoss 1.7595 (CE 0.8137, KL 2.1649) | TrainAcc  76.62% | TestAcc  74.70% | LR 0.00345
Epoch 073/120 | TrainLoss 1.7420 (CE 0.8072, KL 2.1427) | TrainAcc  76.71% | TestAcc  75.31% | LR 0.00333
Epoch 074/120 | TrainLoss 1.7177 (CE 0.7937, KL 2.1137) | TrainAcc  77.05% | TestAcc  75.93% | LR 0.00321
Epoch 075/120 | TrainLoss 1.7122 (CE 0.7951, KL 2.1053) | TrainAcc  77.06% | TestAcc  75.82% | LR 0.00309
Epoch 076/120 | TrainLoss 1.6983 (CE 0.7876, KL 2.0886) | TrainAcc  77.32% | TestAcc  74.78% | LR 0.00297
Epoch 077/120 | TrainLoss 1.7015 (CE 0.7858, KL 2.0939) | TrainAcc  77.19% | TestAcc  76.52% | LR 0.00285
Epoch 078/120 | TrainLoss 1.6918 (CE 0.7856, KL 2.0802) | TrainAcc  77.35% | TestAcc  76.33% | LR 0.00273
Epoch 079/120 | TrainLoss 1.6748 (CE 0.7725, KL 2.0615) | TrainAcc  77.48% | TestAcc  76.71% | LR 0.00261
Epoch 080/120 | TrainLoss 1.6618 (CE 0.7750, KL 2.0418) | TrainAcc  77.53% | TestAcc  77.31% | LR 0.00250
Epoch 081/120 | TrainLoss 1.6531 (CE 0.7677, KL 2.0326) | TrainAcc  77.86% | TestAcc  76.30% | LR 0.00239
Epoch 082/120 | TrainLoss 1.6273 (CE 0.7579, KL 1.9999) | TrainAcc  78.19% | TestAcc  77.22% | LR 0.00228
Epoch 083/120 | TrainLoss 1.6259 (CE 0.7570, KL 1.9983) | TrainAcc  78.23% | TestAcc  75.53% | LR 0.00217
Epoch 084/120 | TrainLoss 1.6217 (CE 0.7552, KL 1.9931) | TrainAcc  78.22% | TestAcc  76.63% | LR 0.00206
Epoch 085/120 | TrainLoss 1.6036 (CE 0.7485, KL 1.9701) | TrainAcc  78.62% | TestAcc  77.49% | LR 0.00196
Epoch 086/120 | TrainLoss 1.6068 (CE 0.7458, KL 1.9759) | TrainAcc  78.26% | TestAcc  76.79% | LR 0.00185
Epoch 087/120 | TrainLoss 1.5869 (CE 0.7444, KL 1.9479) | TrainAcc  78.42% | TestAcc  77.65% | LR 0.00175
Epoch 088/120 | TrainLoss 1.5667 (CE 0.7301, KL 1.9253) | TrainAcc  78.79% | TestAcc  78.11% | LR 0.00165
Epoch 089/120 | TrainLoss 1.5699 (CE 0.7287, KL 1.9304) | TrainAcc  78.75% | TestAcc  77.37% | LR 0.00156
Epoch 090/120 | TrainLoss 1.5498 (CE 0.7248, KL 1.9034) | TrainAcc  79.03% | TestAcc  77.25% | LR 0.00146
Epoch 091/120 | TrainLoss 1.5453 (CE 0.7212, KL 1.8985) | TrainAcc  79.16% | TestAcc  77.96% | LR 0.00137
Epoch 092/120 | TrainLoss 1.5491 (CE 0.7244, KL 1.9025) | TrainAcc  79.13% | TestAcc  77.54% | LR 0.00128
Epoch 093/120 | TrainLoss 1.5314 (CE 0.7139, KL 1.8818) | TrainAcc  79.27% | TestAcc  78.61% | LR 0.00120
Epoch 094/120 | TrainLoss 1.5085 (CE 0.7063, KL 1.8523) | TrainAcc  79.46% | TestAcc  78.35% | LR 0.00111
Epoch 095/120 | TrainLoss 1.5049 (CE 0.7070, KL 1.8468) | TrainAcc  79.41% | TestAcc  78.16% | LR 0.00103
Epoch 096/120 | TrainLoss 1.4956 (CE 0.6975, KL 1.8376) | TrainAcc  79.68% | TestAcc  78.58% | LR 0.00095
Epoch 097/120 | TrainLoss 1.4926 (CE 0.6967, KL 1.8337) | TrainAcc  79.77% | TestAcc  78.04% | LR 0.00088
Epoch 098/120 | TrainLoss 1.4822 (CE 0.6955, KL 1.8193) | TrainAcc  79.71% | TestAcc  77.94% | LR 0.00081
Epoch 099/120 | TrainLoss 1.4708 (CE 0.6877, KL 1.8064) | TrainAcc  79.91% | TestAcc  78.62% | LR 0.00074
Epoch 100/120 | TrainLoss 1.4823 (CE 0.6970, KL 1.8188) | TrainAcc  79.89% | TestAcc  78.81% | LR 0.00067
Epoch 101/120 | TrainLoss 1.4527 (CE 0.6798, KL 1.7839) | TrainAcc  80.22% | TestAcc  78.81% | LR 0.00061
Epoch 102/120 | TrainLoss 1.4415 (CE 0.6762, KL 1.7694) | TrainAcc  80.44% | TestAcc  78.65% | LR 0.00054
Epoch 103/120 | TrainLoss 1.4439 (CE 0.6779, KL 1.7722) | TrainAcc  80.28% | TestAcc  78.56% | LR 0.00049
Epoch 104/120 | TrainLoss 1.4306 (CE 0.6738, KL 1.7549) | TrainAcc  80.35% | TestAcc  79.04% | LR 0.00043
Epoch 105/120 | TrainLoss 1.4351 (CE 0.6735, KL 1.7615) | TrainAcc  80.44% | TestAcc  78.94% | LR 0.00038
Epoch 106/120 | TrainLoss 1.4248 (CE 0.6695, KL 1.7485) | TrainAcc  80.57% | TestAcc  79.16% | LR 0.00033
Epoch 107/120 | TrainLoss 1.4139 (CE 0.6623, KL 1.7361) | TrainAcc  80.64% | TestAcc  79.39% | LR 0.00029
Epoch 108/120 | TrainLoss 1.3998 (CE 0.6540, KL 1.7195) | TrainAcc  80.85% | TestAcc  79.23% | LR 0.00024
Epoch 109/120 | TrainLoss 1.4044 (CE 0.6565, KL 1.7250) | TrainAcc  80.80% | TestAcc  79.23% | LR 0.00021
Epoch 110/120 | TrainLoss 1.3958 (CE 0.6536, KL 1.7138) | TrainAcc  81.00% | TestAcc  79.30% | LR 0.00017
Epoch 111/120 | TrainLoss 1.3938 (CE 0.6481, KL 1.7134) | TrainAcc  81.06% | TestAcc  79.54% | LR 0.00014
Epoch 112/120 | TrainLoss 1.3988 (CE 0.6517, KL 1.7189) | TrainAcc  81.03% | TestAcc  79.37% | LR 0.00011
Epoch 113/120 | TrainLoss 1.3958 (CE 0.6489, KL 1.7159) | TrainAcc  81.03% | TestAcc  79.35% | LR 0.00008
Epoch 114/120 | TrainLoss 1.4098 (CE 0.6608, KL 1.7309) | TrainAcc  80.68% | TestAcc  79.48% | LR 0.00006
Epoch 115/120 | TrainLoss 1.3892 (CE 0.6494, KL 1.7063) | TrainAcc  81.20% | TestAcc  79.52% | LR 0.00004
Epoch 116/120 | TrainLoss 1.3883 (CE 0.6523, KL 1.7037) | TrainAcc  80.94% | TestAcc  79.46% | LR 0.00003
Epoch 117/120 | TrainLoss 1.3798 (CE 0.6451, KL 1.6946) | TrainAcc  80.99% | TestAcc  79.34% | LR 0.00002
Epoch 118/120 | TrainLoss 1.3734 (CE 0.6427, KL 1.6865) | TrainAcc  80.96% | TestAcc  79.37% | LR 0.00001
Epoch 119/120 | TrainLoss 1.3733 (CE 0.6395, KL 1.6877) | TrainAcc  81.21% | TestAcc  79.41% | LR 0.00000
Epoch 120/120 | TrainLoss 1.3835 (CE 0.6461, KL 1.6995) | TrainAcc  81.00% | TestAcc  79.43% | LR 0.00000

Training finished
Best acc: 79.54% (epoch 111)
Checkpoints: checkpoints\distill
(.venv) PS C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration> python scripts/distillation/eval_latency_distil.py
Using device: cuda

Loading checkpoint: checkpoints\distill\mobilenetv3_best.pt
✓ Model loaded

============================================================
J4 | Benchmark Latence GPU MobileNetV3 (Distilled, FP32)
============================================================

Warming up (100 iterations)...
Measuring latency (1000 runs)...

============================================================
LATENCY STATISTICS (batch_size=1, FP32)
============================================================
+-----------+---------+
| Metric    |   Value |
+===========+=========+
| Mean (ms) |  3.9982 |
+-----------+---------+
| Std (ms)  |  0.8006 |
+-----------+---------+
| Min (ms)  |  3.1774 |
+-----------+---------+
| Max (ms)  |  9.0714 |
+-----------+---------+
| p50 (ms)  |  3.8012 |
+-----------+---------+
| p95 (ms)  |  5.5826 |
+-----------+---------+
| p99 (ms)  |  7.3416 |
+-----------+---------+
============================================================

📊 Summary:
  Device: cuda
  Batch size: 1
  Precision: FP32
  Number of runs: 1000
  Mean latency: 3.9982 ms
  p95 latency: 5.5826 ms