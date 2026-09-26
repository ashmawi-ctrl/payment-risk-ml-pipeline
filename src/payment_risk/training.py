import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.model_selection import train_test_split

from .data import Dataset
from .metrics import classification_metrics
from .modeling import build_pipeline, infer_feature_groups


@dataclass(frozen=True)
class TrainingResult:
    metrics: dict[str, Any]
    train_rows: int
    test_rows: int
    positive_rate_train: float
    positive_rate_test: float
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]


def train_baseline(
    dataset: Dataset,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    threshold: float = 0.5,
):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    x_train, x_test, y_train, y_test = train_test_split(
        dataset.features,
        dataset.target,
        test_size=test_size,
        stratify=dataset.target,
        random_state=random_state,
    )

    model = build_pipeline(x_train)
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]

    groups = infer_feature_groups(x_train)
    result = TrainingResult(
        metrics=classification_metrics(
            y_test,
            probability,
            threshold=threshold,
        ),
        train_rows=len(x_train),
        test_rows=len(x_test),
        positive_rate_train=float(y_train.mean()),
        positive_rate_test=float(y_test.mean()),
        numeric_features=groups.numeric,
        categorical_features=groups.categorical,
    )
    return model, result


def save_artifacts(
    model,
    result: TrainingResult,
    output_dir: str | Path,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, destination / "model.joblib")
    (destination / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": result.metrics,
                "train_rows": result.train_rows,
                "test_rows": result.test_rows,
                "positive_rate_train": result.positive_rate_train,
                "positive_rate_test": result.positive_rate_test,
                "numeric_features": list(result.numeric_features),
                "categorical_features": list(
                    result.categorical_features
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
