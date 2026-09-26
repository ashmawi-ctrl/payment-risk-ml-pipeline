from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class Dataset:
    features: pd.DataFrame
    target: pd.Series


def load_dataset(
    path: str | Path,
    *,
    target_column: str,
    drop_columns: tuple[str, ...] = (),
) -> Dataset:
    frame = pd.read_csv(path)

    if frame.empty:
        raise DatasetError("dataset is empty")
    if target_column not in frame.columns:
        raise DatasetError(f"target column not found: {target_column}")
    if frame[target_column].isna().any():
        raise DatasetError("target column contains missing values")

    target = frame[target_column]
    classes = set(target.unique())
    if not classes.issubset({0, 1, False, True}) or len(classes) != 2:
        raise DatasetError(
            "target must contain exactly two binary classes encoded as 0/1"
        )

    missing_drop = sorted(set(drop_columns) - set(frame.columns))
    if missing_drop:
        raise DatasetError(
            "drop columns not found: " + ", ".join(missing_drop)
        )

    excluded = {target_column, *drop_columns}
    features = frame.drop(columns=list(excluded))

    if features.shape[1] == 0:
        raise DatasetError("no feature columns remain after exclusions")

    return Dataset(
        features=features,
        target=target.astype(int),
    )
