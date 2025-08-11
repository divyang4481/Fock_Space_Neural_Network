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

Run training experiments:

```bash
make train-tabular
make train-vision
```

Run tests:

```bash
make test
```
