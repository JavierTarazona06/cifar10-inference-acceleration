import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v3_small


class MobileNetV3Small(nn.Module):
	"""MobileNetV3-Small adapté à CIFAR-10 (weights=None, 10 classes)."""

	def __init__(self, num_classes: int = 10, device: str = "cuda") -> None:
		super().__init__()
		self.device = device
		self.backbone = mobilenet_v3_small(weights=None, num_classes=num_classes)
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
		self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
		self.num_classes = num_classes

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.backbone(x)

	def fit_epoch(self, trainloader, device: str | None = None):
		if device is None:
			device = self.device
		self.train()
		self.to(device)

		correct, total, running_loss = 0, 0, 0.0
		for inputs, targets in trainloader:
			inputs, targets = inputs.to(device), targets.to(device)
			self.optimizer.zero_grad()
			outputs = self(inputs)
			loss = self.criterion(outputs, targets)
			loss.backward()
			self.optimizer.step()

			running_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

		self.scheduler.step()
		avg_loss = running_loss / len(trainloader)
		acc = 100.0 * correct / total
		return avg_loss, acc

	def evaluate(self, testloader, device: str | None = None):
		if device is None:
			device = self.device
		self.eval()
		self.to(device)

		correct, total = 0, 0
		with torch.no_grad():
			for inputs, targets in testloader:
				inputs, targets = inputs.to(device), targets.to(device)
				outputs = self(inputs)
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()

		acc = 100.0 * correct / total
		return acc


def count_parameters(model: nn.Module) -> int:
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module, device: str = "cuda") -> int:
	model.to(device)
	num_params = count_parameters(model)
	print("=" * 50)
	print("Model: MobileNetV3-Small (CIFAR-10)")
	print(f"Device: {device}")
	print(f"Trainable params: {num_params:,}")
	print("=" * 50)
	return num_params
