
# MNIST CNN with CI/CD Pipeline

[![Build Status](https://github.com/akashgaikwad/MNISTCNNCICD/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/akashgaikwad/MNISTCNNCICD/actions)

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification using PyTorch, with an integrated CI/CD pipeline using GitHub Actions.

## Project Overview

The project consists of a CNN model designed to classify handwritten digits from the MNIST dataset. It features:
- A lightweight CNN architecture optimized for MNIST
- Data augmentation for improved training
- Comprehensive automated testing
- Continuous Integration pipeline with GitHub Actions

## Model Architecture

The CNN architecture includes:
- 2 convolutional layers (16 and 32 filters)
- Max pooling layers after each convolution
- ReLU activation functions
- A fully connected output layer
- Input shape: 28x28 grayscale images
- Output: 10 classes (digits 0-9)
- Total parameters: <25,000

## Project Structure

```
├── .github/
│ └── workflows/
│ └── ci_cd.yml
├── src/
│ ├── model.py # CNN model definition
│ ├── train.py # Training script with data augmentation
│ └── test_model.py # Comprehensive model tests
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch >= 2.1.0
- torchvision >= 0.16.0
- pytest == 7.4.0
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MNISTCNNCICD.git
cd MNISTCNNCICD
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

python src/train.py

The training script includes:
- Data augmentation (rotation, affine transforms, perspective)
- Adam optimizer
- CrossEntropy loss
- Automatic model saving with timestamp and accuracy

### Running Tests

```bash
pytest src/test_model.py -v
```

The test suite verifies:
- Input/output shapes
- Model accuracy (>95%)
- Parameter count (<25,000)
- Feature map shapes
- Model deterministic behavior
- Training functionality

## CI/CD Pipeline

The GitHub Actions workflow:
1. Triggers on push events
2. Sets up Python 3.8 environment
3. Installs project dependencies
4. Runs the test suite
5. Uploads trained model artifacts on successful tests


## Contact

Akash Gaikwad - [@akashgaikwad](https://github.com/akashgaikwad)

Project Link: [https://github.com/akashgaikwad/MNISTCNNCICD](https://github.com/akashgaikwad/MNISTCNNCICD)
