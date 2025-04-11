import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Import functions from nn-from-scratch
from nn_from_scratch import (
    Layer, 
    ReLU, 
    Dense, 
    softmax_crossentropy_with_logits,
    forward,
    predict,
    train
)

def test_relu_layer():
    """Test the ReLU layer implementation"""
    # Create a ReLU layer
    relu = ReLU()

    # Test input with positive and negative values
    input = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])

    # Forward pass
    output = relu.forward(input)
    print("Input:", input)
    print("ReLU output:", output)

    # Expected output: [0, 0, 0, 1, 2]
    expected = np.array([[0.0, 0.0, 0.0, 1.0, 2.0]])
    assert np.array_equal(output, expected), f"ReLU forward pass failed. Expected {expected}, got {output}"

    # Backward pass
    grad_output = np.ones_like(output)
    grad_input = relu.backward(input, grad_output)
    print("Gradient output (dL/doutput):", grad_output)
    print("Gradient input (dL/dinput):", grad_input)

    # Expected gradient: [0, 0, 0, 1, 1]
    expected_grad = np.array([[0.0, 0.0, 0.0, 1.0, 1.0]])
    assert np.array_equal(grad_input, expected_grad), f"ReLU backward pass failed. Expected {expected_grad}, got {grad_input}"
    
    return True


def test_dense_layer():
    """Test the Dense layer implementation"""
    # Create a small dense layer: 2 inputs, 3 outputs
    dense = Dense(2, 3, learning_rate=0.1)

    # Set weights and biases for predictable testing
    dense.weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    dense.biases = np.array([0.1, 0.2, 0.3])

    # Test input: batch of 2 examples
    input = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Forward pass
    output = dense.forward(input)
    print("Input shape:", input.shape)
    print("Input:\n", input)
    print("\nWeights shape:", dense.weights.shape)
    print("Weights:\n", dense.weights)
    print("\nBiases:", dense.biases)
    print("\nOutput shape:", output.shape)
    print("Output:\n", output)

    # Verify output with manual calculation
    expected_output = np.array(
        [
            [
                1.0 * 0.1 + 2.0 * 0.4 + 0.1,
                1.0 * 0.2 + 2.0 * 0.5 + 0.2,
                1.0 * 0.3 + 2.0 * 0.6 + 0.3,
            ],
            [
                3.0 * 0.1 + 4.0 * 0.4 + 0.1,
                3.0 * 0.2 + 4.0 * 0.5 + 0.2,
                3.0 * 0.3 + 4.0 * 0.6 + 0.3,
            ],
        ]
    )
    assert np.allclose(output, expected_output), "Dense forward pass failed"
    print("\nCorrect forward pass:", np.allclose(output, expected_output))

    # Test backward pass
    grad_output = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    # Save original weights
    original_weights = dense.weights.copy()
    original_biases = dense.biases.copy()

    grad_input = dense.backward(input, grad_output)

    print("\nGradient output (dL/doutput):\n", grad_output)
    print("Gradient input (dL/dinput):\n", grad_input)

    # Check parameter updates (weights and biases should be updated)
    assert not np.array_equal(original_weights, dense.weights), "Weights were not updated"
    assert not np.array_equal(original_biases, dense.biases), "Biases were not updated"
    
    print("\nWeights before update:\n", original_weights)
    print("Weights after update:\n", dense.weights)
    print("\nBiases before update:", original_biases)
    print("Biases after update:", dense.biases)
    
    return True


def run_tests():
    """Run all tests and report results"""
    tests = [
        ("ReLU Layer Test", test_relu_layer),
        ("Dense Layer Test", test_dense_layer),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {name}")
        print(f"{'='*50}")
        try:
            success = test_func()
            results.append((name, success, None))
            print(f"{name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"{name}: FAILED - {str(e)}")
    
    print(f"\n{'='*50}")
    print("Test Summary")
    print(f"{'='*50}")
    for name, success, error in results:
        status = "PASSED" if success else f"FAILED - {error}"
        print(f"{name}: {status}")


if __name__ == "__main__":
    run_tests() 