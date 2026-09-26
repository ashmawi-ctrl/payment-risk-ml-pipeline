from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureGroups:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]


def infer_feature_groups(features: pd.DataFrame) -> FeatureGroups:
    numeric = tuple(
        features.select_dtypes(include="number").columns.astype(str)
    )
    categorical = tuple(
        column
        for column in features.columns.astype(str)
        if column not in numeric
    )

    if not numeric and not categorical:
        raise ValueError("no supported feature columns found")

    return FeatureGroups(
        numeric=numeric,
        categorical=categorical,
    )


def build_pipeline(features: pd.DataFrame) -> Pipeline:
    groups = infer_feature_groups(features)
    transformers = []

    if groups.numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, list(groups.numeric)))

    if groups.categorical:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "one_hot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=2,
                    ),
                ),
            ]
        )
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                list(groups.categorical),
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )
