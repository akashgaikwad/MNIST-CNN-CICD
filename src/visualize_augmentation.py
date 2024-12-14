import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from train import get_transforms
import numpy as np

def show_augmented_samples(num_samples=5):
    # Get the dataset with augmentation
    dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=get_transforms(train=True),
        download=True
    )
    
    # Get the original dataset without augmentation
    original_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=get_transforms(train=False),
        download=True
    )

    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    for i in range(num_samples):
        idx = torch.randint(len(dataset), (1,)).item()
        
        # Get original image
        orig_img, label = original_dataset[idx]
        orig_img = orig_img.squeeze().numpy()
        
        # Get augmented image
        aug_img, _ = dataset[idx]
        aug_img = aug_img.squeeze().numpy()
        
        # Plot original image
        axes[0, i].imshow(orig_img, cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original\nLabel: {label}')
        
        # Plot augmented image
        axes[1, i].imshow(aug_img, cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Augmented')
    
    plt.tight_layout()
    plt.savefig('augmented_samples.png')
    plt.close()

if __name__ == '__main__':
    show_augmented_samples() 