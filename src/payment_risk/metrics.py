from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true,
    probability,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")

    probability = np.asarray(probability)
    predicted = (probability >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])

    return {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(
            average_precision_score(y_true, probability)
        ),
        "precision": float(
            precision_score(y_true, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predicted, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
    }
