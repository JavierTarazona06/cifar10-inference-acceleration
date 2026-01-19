# CIFAR-10 — GPU Inference Acceleration (batch=1)
Optimize GPU inference latency for CIFAR-10 while keeping strong accuracy.

## Project goals (requirements)
- **Accuracy:** reach **≥ 85%** on **CIFAR-10 test**.
- **Primary metric:** **GPU inference latency** with **batch = 1**.
- **Data constraint:** training uses **only CIFAR-10 train** (augmentations allowed). No external data or external pretraining.
- **Reporting:** at least **mean latency** and **p95 latency** for each variant; track accuracy and (optionally) model size / params.


## Repository structure (suggested)
.

├── src/

│   ├── data.py                # CIFAR-10 dataloaders + transforms

│   ├── train.py               # training entrypoint

│   ├── eval.py                # accuracy evaluation on test

│   ├── bench.py               # GPU latency benchmark (batch=1)

│   ├── models/

│   │   ├── resnet18_cifar.py

│   │   └── mobilenetv3_cifar.py

│   └── utils/

│       ├── seed.py            # reproducibility helpers

│       └── logging.py         # save metrics/metadata to CSV

├── checkpoints/               # saved weights (gitignored)

└── README.md


## Notes / guardrails
- No ImageNet weights: instantiate models with `weights=None` (or equivalent) and train on CIFAR-10.
- Keep benchmarking fair: same preprocessing, same precision mode, same GPU, same batch size, same protocol.

## Deliverables
- Training + evaluation + benchmarking code
- Results CSV + final comparison table (accuracy vs latency)
- Final model weights + config to reproduce the reported numbers

## Configuration
- Centralized settings live in [src/cifaracce/config.py](src/cifaracce/config.py): device, seeds, training hyperparameters, checkpoint paths, and benchmark defaults.
- Scripts import this module, e.g., the MobileNet J2 training at [scripts/mobilenet_j2/train_mobilenet_j2.py](scripts/mobilenet_j2/train_mobilenet_j2.py) uses `cfg.DEVICE`, `cfg.TRAIN_EPOCHS`, optimizer params, and paths.
- To change defaults, edit the values in `config.py`. This keeps experiments consistent and reproducible.
