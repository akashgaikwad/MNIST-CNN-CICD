import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTNet
import datetime
import os

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
            transforms.RandomPerspective(distortion_scale=0.05, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

def train(save_model=True):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    
    # MNIST Dataset with augmentation
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        transform=get_transforms(train=True),
        download=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Model
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training statistics
    total_step = len(train_loader)
    total_correct = 0
    total_samples = 0
    
    # Training loop - one epoch
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        
        if (i+1) % 100 == 0:
            print(f'Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
    
    accuracy = 100 * total_correct / total_samples
    print(f'Final Accuracy: {accuracy:.2f}%')
    
    if save_model:
        # Save model with timestamp and accuracy
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'model_acc{accuracy:.2f}_{timestamp}.pth'
        torch.save(model.state_dict(), model_path)
    
    return accuracy

if __name__ == '__main__':
    model = MNISTNet()
    model.print_model_summary()
    train()