import json
from pathlib import Path

import numpy as np
import pandas as pd

from payment_risk.data import Dataset
from payment_risk.training import save_artifacts, train_baseline


def sample_dataset(rows: int = 200) -> Dataset:
    rng = np.random.default_rng(10)
    amount = rng.normal(300, 100, size=rows)
    attempts = rng.integers(1, 6, size=rows)
    channel = rng.choice(["web", "mobile", "pos"], size=rows)

    risky = (
        (amount > 360)
        | (attempts >= 4)
        | ((channel == "web") & (amount > 300))
    ).astype(int)

    return Dataset(
        features=pd.DataFrame(
            {
                "amount": amount,
                "attempts": attempts,
                "channel": channel,
            }
        ),
        target=pd.Series(risky),
    )


def test_training_returns_expected_metrics() -> None:
    model, result = train_baseline(sample_dataset())

    assert hasattr(model, "predict_proba")
    assert result.train_rows == 150
    assert result.test_rows == 50
    assert 0 <= result.metrics["average_precision"] <= 1
    assert 0 <= result.metrics["roc_auc"] <= 1
    assert result.numeric_features == ("amount", "attempts")
    assert result.categorical_features == ("channel",)


def test_artifacts_are_written(tmp_path: Path) -> None:
    model, result = train_baseline(sample_dataset())

    save_artifacts(model, result, tmp_path)

    assert (tmp_path / "model.joblib").is_file()
    metrics = json.loads(
        (tmp_path / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["train_rows"] == 150
    assert "average_precision" in metrics["metrics"]
