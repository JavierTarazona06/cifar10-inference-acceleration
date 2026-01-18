import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # Load the pre-trained ResNet model
        self.resnet = models.resnet18(weights=None)  # Initialize from scratch
        # Modify the first layer
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove the initial max pooling layer
        self.resnet.maxpool = nn.Identity()
        # Change the output layer to match CIFAR-10 classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Example usage
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Initialize model
    model = ResNet18(num_classes=10)
    model = model.to(device)
    model.eval()
    
    # Create a dummy batch: (batch_size=1, channels=3, height=32, width=32)
    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    
    # Validation checks
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    print("✓ Output shape is correct: (1, 10)")
    
    # Test with multiple batch sizes
    for batch_size in [1, 4, 8, 16, 32]:
        dummy_input = torch.randn(batch_size, 3, 32, 32, device=device)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (batch_size, 10), f"Batch {batch_size}: Expected shape ({batch_size}, 10), got {output.shape}"
        print(f"✓ Batch size {batch_size}: output shape {output.shape}")
    
    print("\n:) All forward pass tests passed!")
    print(f"\nModel architecture:\n{model}")