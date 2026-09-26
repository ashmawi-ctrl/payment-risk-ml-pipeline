.PHONY: install lint test quality demo

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest

quality: lint test

demo:
	python scripts/generate_demo_data.py --rows 2000 --output data/demo_transactions.csv
	payment-risk train data/demo_transactions.csv --target risky --output-dir artifacts/demo
