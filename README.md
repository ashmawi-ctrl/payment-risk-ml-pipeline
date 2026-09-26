# Payment Risk ML Pipeline

[![quality](https://github.com/ashmawi-ctrl/payment-risk-ml-pipeline/actions/workflows/quality.yml/badge.svg)](https://github.com/ashmawi-ctrl/payment-risk-ml-pipeline/actions/workflows/quality.yml)

A small machine-learning project for estimating transaction risk from tabular payment data.

The project grew out of the kinds of patterns I investigate in payment support work: failed transactions, unusual amounts, repeated attempts, channel differences, and response-code behavior. It does **not** use employer or customer data. The included demo data is synthetic, and the pipeline is designed so a public dataset can be plugged in later.

The first model is deliberately simple. The goal is to establish a reproducible baseline, avoid leakage, and measure the right things before trying more complex models.

## What the baseline does

- reads a CSV with a configurable binary target
- allows identifier or leakage-prone columns to be explicitly excluded
- validates target and schema assumptions before training
- splits data before fitting preprocessing
- uses median imputation + scaling for numeric features
- uses most-frequent imputation + one-hot encoding for categorical features
- trains a class-balanced logistic regression baseline
- reports ROC-AUC and average precision
- reports threshold-dependent precision, recall, F1, and confusion matrix
- persists the complete sklearn pipeline with `joblib`
- writes a machine-readable metrics file
- keeps synthetic demo generation separate from model code

## Why start with logistic regression?

Payment-risk data is usually imbalanced, and a complex model can look impressive while hiding basic data problems.

A logistic baseline is useful because it is:

- quick to train
- easy to reproduce
- easy to compare against later models
- strong enough to expose whether the feature set contains useful signal
- less likely to distract from leakage, split, and metric mistakes

This repository is not claiming that logistic regression is the best fraud or risk model.

## Project structure

```text
src/payment_risk/
  cli.py
  data.py
  metrics.py
  modeling.py
  training.py

scripts/
  generate_demo_data.py

tests/
  test_data.py
  test_metrics.py
  test_training.py
```

## Install

Python 3.11+:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

## Generate local demo data

```bash
python scripts/generate_demo_data.py \
  --rows 5000 \
  --output data/demo_transactions.csv
```

The generator creates intentionally simple synthetic signals such as:

- transaction amount
- recent attempt count
- account age
- transaction hour
- channel
- card-country match
- 3DS state

It is only for exercising the pipeline. It is not intended to simulate real-world fraud distributions accurately.

## Train the baseline

The synthetic data contains a transaction identifier. It should not be used as a model feature, so the CLI makes that exclusion explicit:

```bash
payment-risk train data/demo_transactions.csv \
  --target risky \
  --drop-column transaction_id \
  --output-dir artifacts/demo
```

Artifacts:

```text
artifacts/demo/
  model.joblib
  metrics.json
```

## Why average precision matters here

Accuracy can be misleading when the positive class is uncommon.

The baseline reports ROC-AUC, but it also reports **average precision**, which is usually more informative when the goal is identifying a relatively small positive class.

Threshold metrics are reported separately because the right operating threshold depends on the real cost of false positives vs false negatives.

## Leakage rules

The target is removed before preprocessing.

Columns such as transaction IDs or post-decision fields should be removed with repeated `--drop-column` arguments:

```bash
payment-risk train transactions.csv \
  --target risky \
  --drop-column transaction_id \
  --drop-column final_review_decision \
  --output-dir artifacts/run-01
```

The tool does not guess which business columns are leakage. That decision should be explicit and documented for each dataset.

## Quality checks

```bash
make quality
```

GitHub Actions runs the same lint and test checks on pushes and pull requests.

## Docker

```bash
docker build -t payment-risk .

docker run --rm \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/artifacts:/artifacts" \
  payment-risk train /data/demo_transactions.csv \
  --target risky \
  --drop-column transaction_id \
  --output-dir /artifacts/demo
```

## Engineering workflow

The first baseline is tracked through:

- [Issue #1](https://github.com/ashmawi-ctrl/payment-risk-ml-pipeline/issues/1)
- feature branch: `feat/baseline-risk-model`
- schema and training regression tests
- GitHub Actions
- reviewable pull request

## Current limitations

- one random stratified train/test split
- no cross-validation yet
- no threshold selection strategy
- no probability calibration
- no temporal split
- no model explainability report
- no dataset-specific feature engineering
- no Kaggle adapter yet

Those are intentional. The first version is a baseline to compare later work against.

## Next steps

- add a temporal split option for datasets with transaction timestamps
- compare tree-based models against the baseline
- add cross-validation
- add calibration plots / Brier score
- add precision-recall threshold analysis
- create a Kaggle notebook that reuses the same package code instead of duplicating training logic
- document a public dataset experiment with reproducible metrics

## Data and privacy

No private payment, merchant, or customer records are included in this repository.

Use only datasets you have permission to process and publish.

## License

MIT
