import pytest

from payment_risk.metrics import classification_metrics


def test_threshold_must_be_probability() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        classification_metrics(
            [0, 1],
            [0.1, 0.9],
            threshold=1.0,
        )
