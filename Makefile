PYTHON := python

.PHONY: setup lint test format train-tabular train-vision

setup:
$(PYTHON) -m pip install -e ".[dev]"

lint:
pre-commit run --files $(shell git ls-files '*.py')

test:
pytest

format:
black .

train-tabular:
$(PYTHON) scripts/train.py --config configs/tabular/fsnn.yaml

train-vision:
$(PYTHON) scripts/train.py --config configs/vision/fsnn.yaml
