from pathlib import Path

import pandas as pd
import pytest

from payment_risk.data import DatasetError, load_dataset


def write_csv(tmp_path: Path, frame: pd.DataFrame) -> Path:
    path = tmp_path / "transactions.csv"
    frame.to_csv(path, index=False)
    return path


def test_load_dataset_separates_target_and_dropped_identifier(
    tmp_path: Path,
) -> None:
    path = write_csv(
        tmp_path,
        pd.DataFrame(
            {
                "transaction_id": ["a", "b", "c", "d"],
                "amount": [10, 20, 30, 40],
                "risky": [0, 1, 0, 1],
            }
        ),
    )

    dataset = load_dataset(
        path,
        target_column="risky",
        drop_columns=("transaction_id",),
    )

    assert list(dataset.features.columns) == ["amount"]
    assert dataset.target.tolist() == [0, 1, 0, 1]


def test_target_must_be_binary(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        pd.DataFrame(
            {
                "amount": [10, 20, 30],
                "risky": [0, 1, 2],
            }
        ),
    )

    with pytest.raises(DatasetError, match="two binary classes"):
        load_dataset(path, target_column="risky")


def test_missing_drop_column_is_reported(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        pd.DataFrame(
            {
                "amount": [10, 20, 30, 40],
                "risky": [0, 1, 0, 1],
            }
        ),
    )

    with pytest.raises(DatasetError, match="drop columns not found"):
        load_dataset(
            path,
            target_column="risky",
            drop_columns=("transaction_id",),
        )
