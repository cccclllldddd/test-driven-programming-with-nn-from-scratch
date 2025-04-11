# Test-Driven Programming with Neural Networks from Scratch

This repository is for educational purposes to learn test-driven development while building a neural network from scratch using NumPy.

## Project Structure

- **nn_from_scratch.py**: The main implementation of neural network components from scratch
- **code-for-test.py**: Helper functions for testing the neural network components
- **tests/**: Directory containing test files
  - **test.py**: Basic test for model accuracy
  - **test_all.py**: Comprehensive test suite for all components
- **.github/workflows/test.yml**: GitHub Actions workflow for automated testing
- **mnist_train.csv, mnist_test.csv**: MNIST dataset for training and testing

## Getting Started

### Prerequisites

- Python 3.9+
- Required packages (install via `pip install -r requirements.txt`):
  - numpy
  - pandas
  - matplotlib
  - pytest

## Running Tests

### Run Individual Component Tests

To test only the ReLU and Dense layer implementations:

```bash
python code-for-test.py
```

### Run Complete Test Suite

To run all tests including model accuracy:

```bash
python -m pytest tests/test_all.py -v
```

### Run Model Accuracy Test Only

```bash
python -m pytest tests/test.py -v
```

## Testing Requirements

The tests verify:

1. ReLU Layer: Correct forward and backward pass implementations
2. Dense Layer: Correct forward and backward pass implementations with weight updates
3. Model Accuracy: The neural network achieves at least 70% validation accuracy on MNIST

## GitHub Actions

This repository includes a GitHub Actions workflow that automatically runs tests on push. It:

1. Sets up Python 3.9
2. Installs dependencies
3. Runs component tests
4. Tests model accuracy

## License

This project is provided for educational purposes only. 