import torch
import pytest
from model import MNISTNet
from train import train

def test_parameter_count():
    model = MNISTNet()
    param_count = model.count_parameters()
    assert param_count < 100000, f"Model has {param_count} parameters (should be <100000)"
    print(f"Parameter test passed: {param_count} parameters")

def test_input_shape():
    model = MNISTNet()
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
        print("Input shape test passed")
    except Exception as e:
        pytest.fail(f"Model failed to process 28x28 input: {str(e)}")

def test_output_shape():
    model = MNISTNet()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"
    print("Output shape test passed")

def test_model_accuracy():
    accuracy = train(save_model=False)
    assert accuracy > 80.0, f"Model accuracy {accuracy:.2f}% is below 80%"
    print(f"Accuracy test passed: {accuracy:.2f}%")