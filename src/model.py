import torch
import torch.nn as nn

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # First conv layer: 1->16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # Second conv layer: 16->32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # Third conv layer: 32->32
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        # Fully connected layer: 32*3*3->10
        self.fc = nn.Linear(32 * 3 * 3, 10)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 28x28 -> 14x14 -> 7x7 -> 3x3
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 32 * 3 * 3)
        x = self.fc(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def print_model_summary(self):
        """Print layer-wise parameter count"""
        total_params = 0
        for name, parameter in self.named_parameters():
            param_count = parameter.numel()
            print(f'{name}: {param_count:,} parameters')
            total_params += param_count
        print(f'\nTotal Parameters: {total_params:,}')
        return total_params