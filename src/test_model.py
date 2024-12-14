import torch
import pytest
from model import MNISTNet
from train import train



def test_input_shape():
    model = MNISTNet()
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
        print("Input shape test passed")
    except Exception as e:
        pytest.fail(f"Model failed to process 28x28 input: {str(e)}")

def test_model_accuracy():
    accuracy = train(save_model=False)
    assert accuracy > 95.0, f"Model accuracy {accuracy:.2f}% is below 95%"
    print(f"Accuracy test passed: {accuracy:.2f}%")

def test_output_shape():
    model = MNISTNet()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"
    print("Output shape test passed")


def test_parameter_count():
    model = MNISTNet()
    print("\nModel Parameter Breakdown:")
    param_count = model.print_model_summary()
    assert param_count < 25000, f"Model has {param_count:,} parameters (should be <25000)"
    print(f"\nParameter test passed: {param_count:,} parameters")


def test_feature_map_shapes():
    model = MNISTNet()
    x = torch.randn(1, 1, 28, 28)
    
    # Get intermediate feature maps
    conv1_out = model.pool(model.relu(model.conv1(x)))
    conv2_out = model.pool(model.relu(model.conv2(conv1_out)))
    
    assert conv1_out.shape == (1, 16, 14, 14), f"Conv1 output shape incorrect: {conv1_out.shape}"
    assert conv2_out.shape == (1, 32, 7, 7), f"Conv2 output shape incorrect: {conv2_out.shape}"

def test_model_deterministic():
    model = MNISTNet()
    x = torch.randn(1, 1, 28, 28)
    
    # Test if model gives same output for same input
    output1 = model(x)
    output2 = model(x)
    
    assert torch.allclose(output1, output2), "Model not deterministic"
    print("Deterministic behavior test passed")
