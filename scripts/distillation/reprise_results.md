(.venv) PS C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration> python scripts/distillation/resume_distill_mobilenet_j4.py --checkpoint checkpoints/distill/mobilenetv3_best.pt --additional-epochs 180 --lr 0.001 --prev-best-acc 79.43
Using device: cuda

Loading student checkpoint: checkpoints\distill\mobilenetv3_best.pt
 Student loaded (previous best acc: 79.43%)

 Loaded teacher from checkpoints\resnet18\resnet18_best.pt

 Optimizer state not in checkpoint, starting fresh with lr=0.001

Resume training: epoch 121 → 300
Target: test_acc >= 85%

================================================================================
Epoch 121 | TrainLoss 1.4614 (CE 0.6816, KL 1.7956) | TrainAcc  80.12% | TestAcc  78.33% | LR 0.00100
Epoch 122 | TrainLoss 1.4801 (CE 0.6912, KL 1.8182) | TrainAcc  80.06% | TestAcc  78.64% | LR 0.00100
Epoch 123 | TrainLoss 1.4718 (CE 0.6887, KL 1.8074) | TrainAcc  79.95% | TestAcc  78.85% | LR 0.00100
Epoch 124 | TrainLoss 1.4713 (CE 0.6843, KL 1.8086) | TrainAcc  79.88% | TestAcc  78.45% | LR 0.00100
Epoch 125 | TrainLoss 1.4670 (CE 0.6870, KL 1.8013) | TrainAcc  79.91% | TestAcc  78.45% | LR 0.00100
Epoch 126 | TrainLoss 1.4752 (CE 0.6951, KL 1.8096) | TrainAcc  79.74% | TestAcc  78.38% | LR 0.00100
Epoch 127 | TrainLoss 1.4741 (CE 0.6900, KL 1.8101) | TrainAcc  79.82% | TestAcc  78.43% | LR 0.00100
Epoch 128 | TrainLoss 1.4591 (CE 0.6817, KL 1.7923) | TrainAcc  80.11% | TestAcc  77.98% | LR 0.00100
Epoch 129 | TrainLoss 1.4601 (CE 0.6801, KL 1.7944) | TrainAcc  80.17% | TestAcc  78.39% | LR 0.00099
Epoch 130 | TrainLoss 1.4679 (CE 0.6850, KL 1.8035) | TrainAcc  80.14% | TestAcc  77.83% | LR 0.00099
Epoch 131 | TrainLoss 1.4577 (CE 0.6836, KL 1.7895) | TrainAcc  80.18% | TestAcc  78.70% | LR 0.00099
Epoch 132 | TrainLoss 1.4593 (CE 0.6782, KL 1.7941) | TrainAcc  80.24% | TestAcc  78.26% | LR 0.00099
Epoch 133 | TrainLoss 1.4691 (CE 0.6885, KL 1.8036) | TrainAcc  79.96% | TestAcc  78.87% | LR 0.00099
Epoch 134 | TrainLoss 1.4733 (CE 0.6912, KL 1.8085) | TrainAcc  79.91% | TestAcc  78.76% | LR 0.00099
Epoch 135 | TrainLoss 1.4589 (CE 0.6837, KL 1.7912) | TrainAcc  80.34% | TestAcc  78.73% | LR 0.00098
Epoch 136 | TrainLoss 1.4664 (CE 0.6840, KL 1.8017) | TrainAcc  80.12% | TestAcc  78.68% | LR 0.00098
Epoch 137 | TrainLoss 1.4626 (CE 0.6778, KL 1.7990) | TrainAcc  80.22% | TestAcc  78.51% | LR 0.00098
Epoch 138 | TrainLoss 1.4495 (CE 0.6760, KL 1.7810) | TrainAcc  80.29% | TestAcc  78.77% | LR 0.00098
Epoch 139 | TrainLoss 1.4463 (CE 0.6784, KL 1.7754) | TrainAcc  80.26% | TestAcc  78.79% | LR 0.00097
Epoch 140 | TrainLoss 1.4623 (CE 0.6815, KL 1.7969) | TrainAcc  80.07% | TestAcc  78.40% | LR 0.00097
Epoch 141 | TrainLoss 1.4555 (CE 0.6835, KL 1.7863) | TrainAcc  80.06% | TestAcc  78.80% | LR 0.00097
Epoch 142 | TrainLoss 1.4450 (CE 0.6760, KL 1.7746) | TrainAcc  80.35% | TestAcc  78.11% | LR 0.00096
Epoch 143 | TrainLoss 1.4417 (CE 0.6740, KL 1.7707) | TrainAcc  80.29% | TestAcc  78.47% | LR 0.00096
Epoch 144 | TrainLoss 1.4495 (CE 0.6780, KL 1.7802) | TrainAcc  80.36% | TestAcc  79.01% | LR 0.00096
Epoch 145 | TrainLoss 1.4454 (CE 0.6754, KL 1.7754) | TrainAcc  80.46% | TestAcc  78.29% | LR 0.00095
Epoch 146 | TrainLoss 1.4421 (CE 0.6794, KL 1.7689) | TrainAcc  80.29% | TestAcc  78.61% | LR 0.00095
Epoch 147 | TrainLoss 1.4354 (CE 0.6710, KL 1.7631) | TrainAcc  80.35% | TestAcc  79.12% | LR 0.00095
Epoch 148 | TrainLoss 1.4458 (CE 0.6742, KL 1.7766) | TrainAcc  80.34% | TestAcc  78.56% | LR 0.00094
Epoch 149 | TrainLoss 1.4357 (CE 0.6724, KL 1.7628) | TrainAcc  80.46% | TestAcc  78.83% | LR 0.00094
Epoch 150 | TrainLoss 1.4318 (CE 0.6714, KL 1.7577) | TrainAcc  80.43% | TestAcc  78.20% | LR 0.00093
Epoch 151 | TrainLoss 1.4415 (CE 0.6760, KL 1.7695) | TrainAcc  80.35% | TestAcc  78.24% | LR 0.00093
Epoch 151 | TrainLoss 1.4415 (CE 0.6760, KL 1.7695) | TrainAcc  80.35% | TestAcc  78.24% | LR 0.00093
Epoch 152 | TrainLoss 1.4352 (CE 0.6718, KL 1.7624) | TrainAcc  80.38% | TestAcc  79.04% | LR 0.00092
Epoch 153 | TrainLoss 1.4193 (CE 0.6661, KL 1.7421) | TrainAcc  80.58% | TestAcc  78.90% | LR 0.00092
Epoch 154 | TrainLoss 1.4286 (CE 0.6643, KL 1.7562) | TrainAcc  80.57% | TestAcc  78.61% | LR 0.00091
Epoch 155 | TrainLoss 1.4233 (CE 0.6616, KL 1.7497) | TrainAcc  80.67% | TestAcc  78.86% | LR 0.00091
Epoch 156 | TrainLoss 1.4256 (CE 0.6661, KL 1.7512) | TrainAcc  80.79% | TestAcc  78.70% | LR 0.00090
Epoch 157 | TrainLoss 1.4190 (CE 0.6573, KL 1.7454) | TrainAcc  80.77% | TestAcc  78.84% | LR 0.00090
Epoch 158 | TrainLoss 1.4245 (CE 0.6656, KL 1.7498) | TrainAcc  80.52% | TestAcc  78.75% | LR 0.00089
Epoch 159 | TrainLoss 1.4271 (CE 0.6682, KL 1.7523) | TrainAcc  80.72% | TestAcc  78.68% | LR 0.00089
Epoch 160 | TrainLoss 1.4151 (CE 0.6597, KL 1.7389) | TrainAcc  80.79% | TestAcc  78.55% | LR 0.00088
Epoch 161 | TrainLoss 1.4094 (CE 0.6530, KL 1.7336) | TrainAcc  80.89% | TestAcc  78.30% | LR 0.00088
Epoch 162 | TrainLoss 1.4194 (CE 0.6616, KL 1.7441) | TrainAcc  80.72% | TestAcc  79.06% | LR 0.00087
Epoch 163 | TrainLoss 1.4159 (CE 0.6590, KL 1.7403) | TrainAcc  80.85% | TestAcc  78.80% | LR 0.00087
Epoch 164 | TrainLoss 1.4174 (CE 0.6625, KL 1.7409) | TrainAcc  80.76% | TestAcc  78.52% | LR 0.00086
Epoch 165 | TrainLoss 1.4170 (CE 0.6559, KL 1.7431) | TrainAcc  80.96% | TestAcc  78.57% | LR 0.00085
Epoch 166 | TrainLoss 1.4142 (CE 0.6628, KL 1.7362) | TrainAcc  80.70% | TestAcc  78.66% | LR 0.00085
Epoch 167 | TrainLoss 1.3982 (CE 0.6503, KL 1.7188) | TrainAcc  80.99% | TestAcc  78.64% | LR 0.00084
Epoch 168 | TrainLoss 1.4057 (CE 0.6538, KL 1.7280) | TrainAcc  80.93% | TestAcc  78.41% | LR 0.00083
Epoch 169 | TrainLoss 1.3984 (CE 0.6492, KL 1.7195) | TrainAcc  81.03% | TestAcc  78.85% | LR 0.00083
Epoch 170 | TrainLoss 1.4106 (CE 0.6581, KL 1.7331) | TrainAcc  80.83% | TestAcc  78.56% | LR 0.00082
 KL 1.6913) | TrainAcc  81.36% | TestAcc  78.69% | LR 0.00068
Epoch 190 | TrainLoss 1.3631 (CE 0.6369, KL 1.6744) | TrainAcc  81.43% | TestAcc  78.74% | LR 0.00067
Epoch 191 | TrainLoss 1.3707 (CE 0.6427, KL 1.6828) | TrainAcc  81.35% | TestAcc  79.19% | LR 0.00066
Epoch 192 | TrainLoss 1.3613 (CE 0.6335, KL 1.6733) | TrainAcc  81.52% | TestAcc  78.84% | LR 0.00065
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:\Users\javit\AppData\Local\Programs\Python\Python312\Lib\multiprocessing\spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\javit\AppData\Local\Programs\Python\Python312\Lib\multiprocessing\spawn.py", line 131, in _main
    prepare(preparation_data)
  File "C:\Users\javit\AppData\Local\Programs\Python\Python312\Lib\multiprocessing\spawn.py", line 246, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
  File "C:\Users\javit\AppData\Local\Programs\Python\Python312\Lib\multiprocessing\spawn.py", line 297, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen runpy>", line 286, in run_path
  File "<frozen runpy>", line 98, in _run_module_code
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration\scripts\distillation\resume_distill_mobilenet_j4.py", line 30, in <module>
    import torch
  File "C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration\.venv\Lib\site-packages\torch\__init__.py", line 2016, in <module>
    from torch import _VF as _VF, functional as functional  # usort: skip
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration\.venv\Lib\site-packages\torch\functional.py", line 7, in <module>
    import torch.nn.functional as F
  File "C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration\.venv\Lib\site-packages\torch\nn\__init__.py", line 8, in <module>
    from torch.nn.modules import *  # usort: skip # noqa: F403
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration\.venv\Lib\site-packages\torch\nn\modules\__init__.py", line 1, in <module>
    from .module import Module  # usort: skip
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration\.venv\Lib\site-packages\torch\nn\modules\module.py", line 29, in <module>
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass
  File "C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration\.venv\Lib\site-packages\torch\utils\__init__.py", line 8, in <module>
    from torch.utils import (
  File "C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration\.venv\Lib\site-packages\torch\utils\data\__init__.py", line 1, in <module>
    from torch.utils.data.dataloader import (
", line 1, in <module>
    from torch.utils.data.dataloader import (
  File "C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration\.venv\Lib\site-packages\torch\utils\data\dataloader.py", line 20, in <module>
    from torch.utils.data.dataloader import (
  File "C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration\.venv\Lib\site-packages\torch\utils\data\dataloader.py", line 20, in <module>
  File "C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration\.venv\Lib\site-packages\torch\utils\data\dataloader.py", line 20, in <module>
    import torch.distributed as dist
  File "C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration\.venv\Lib\site-packages\torch\distributed\__init__.py", line 122, in <module>
    from .device_mesh import DeviceMesh, init_device_mesh
y", line 122, in <module>
    from .device_mesh import DeviceMesh, init_device_mesh
    from .device_mesh import DeviceMesh, init_device_mesh
  File "C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration\.venv\Lib\site-packages\torch\distributed\device_mesh.py", line 39, in <module>
    from torch.distributed.distributed_c10d import (
  File "C:\Users\javit\Documents\ENSTA\Module 2\Apprentissage Automatique\cifar10-inference-acceleration\.venv\Lib\site-packages\torch\distributed\distributed_c10d.py", line 49, in <module>
    from .c10d_logger import _exception_logger, _time_logger
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 991, in exec_module
  File "<frozen importlib._bootstrap_external>", line 1087, in get_code
  File "<frozen importlib._bootstrap_external>", line 1186, in get_data
KeyboardInterrupt
