# CIFAR-10 — GPU Inference Acceleration (batch=1)
Optimize GPU inference latency for CIFAR-10 while keeping strong accuracy.

## Project goals (requirements)
- **Accuracy:** reach **≥ 85%** on **CIFAR-10 test**.
- **Primary metric:** **GPU inference latency** with **batch = 1**.
- **Data constraint:** training uses **only CIFAR-10 train** (augmentations allowed). No external data or external pretraining.
- **Reporting:** at least **mean latency** and **p95 latency** for each variant; track accuracy and (optionally) model size / params.

# Installation and Execution

## Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA 12.1 support
- pip or conda package manager

## Installation Instructions

### For Linux and Windows

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JavierTarazona06/cifar10-inference-acceleration.git
   cd cifar10-inference-acceleration
   ```

2. **Create a virtual environment:**

   **Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   **Windows (PowerShell):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   **Windows (Command Prompt):**
   ```cmd
   python -m venv venv
   venv\Scripts\activate.bat
   ```

3. **Install PyTorch with CUDA 12.1 support:**
   ```bash
   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

## Running Scripts

To execute any training, evaluation, or benchmarking script, use:

```bash
python scripts/<path_to_script>/<script_name>.py [arguments]
```

### Examples:

- Train MobileNetV3-Small (J2):
  ```bash
  python scripts/mobilenet_j2/train_mobilenet_j2.py
  ```

- Evaluate ResNet-18 (J3):
  ```bash
  python scripts/resnet_j3/evaluate_resnet18.py
  ```

- Benchmark latency (J4):
  ```bash
  python scripts/distillation/eval_latency_distil.py
  ```

- Run FP16 evaluation:
  ```bash
  python scripts/fp16/eval_latency_mobilenetv3.py
  ```

- Run torch.compile benchmark:
  ```bash
  python scripts/compile/benchmark_compile_mobilenetv3.py
  ```

For detailed arguments and options, see individual script files or run:
```bash
python scripts/<path_to_script>/<script_name>.py --help
```

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
