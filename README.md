# MNIST CNN with CI/CD Pipeline

[![Build Status](https://github.com/akashgaikwad/MNISTCNNCICD/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/akashgaikwad/MNISTCNNCICD/actions)

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification using PyTorch, with an integrated CI/CD pipeline using GitHub Actions.

## Project Overview

The project consists of a CNN model designed to classify handwritten digits from the MNIST dataset. It features:
- A custom CNN architecture with batch normalization
- Automated testing and model artifact generation
- Continuous Integration/Continuous Deployment pipeline

## Model Architecture

The CNN architecture includes:
- 3 convolutional layers with batch normalization
- Max pooling layers
- ReLU activation functions
- A fully connected output layer
- Input shape: 28x28 grayscale images
- Output: 10 classes (digits 0-9)

## Project Structure

```
├── .github/
│   └── workflows/
│       └── ci_cd.yml
├── src/
│   ├── model.py
│   └── test_model.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch
- pytest
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

### Running Tests

```bash
pytest src/test_model.py -v
```

## CI/CD Pipeline

The project includes a GitHub Actions workflow that:
1. Sets up Python environment
2. Installs dependencies
3. Runs tests
4. Uploads model artifacts on successful test completion


## Contact

Akash Gaikwad - [@akashgaikwad](https://github.com/akashgaikwad)

Project Link: [https://github.com/akashgaikwad/MNISTCNNCICD](https://github.com/akashgaikwad/MNISTCNNCICD)
