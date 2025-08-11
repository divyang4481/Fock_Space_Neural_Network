# Fock Space Neural Network (FSNN)

A research repository exploring dynamic neural network architectures using creation and annihilation operators.

## Installation

```bash
pip install -e .
```

For development, install with extra dependencies:

```bash
pip install -e ".[dev]"
```

## Example Usage

Download datasets:

```bash
python scripts/download_data.py
```

Run training experiments (YAML configs in `configs/`):

```bash
make train-tabular  # UCI Adult
make train-vision   # MNIST
```

Evaluate a trained model:

```bash
python scripts/eval.py --config configs/vision/mnist.yaml --checkpoint runs/mnist/model.pt
```

Run tests:

```bash
make test
```
