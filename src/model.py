import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

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