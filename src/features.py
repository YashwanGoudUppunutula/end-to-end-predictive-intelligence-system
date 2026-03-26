from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class DateFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create recency features from datetime columns and drop raw date columns."""

    def __init__(self, reference_date: str = "2024-12-31") -> None:
        self.reference_date = pd.Timestamp(reference_date)
        self.date_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DateFeatureEngineer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DateFeatureEngineer expects pandas DataFrame input.")
        self.date_columns_ = [
            col
            for col in [
                "signup_date",
                "last_transaction_date",
                "last_login_feature_date",
                "last_login_date",
            ]
            if col in X.columns
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        for col in self.date_columns_:
            date_series = pd.to_datetime(X_out[col], errors="coerce")
            X_out[f"{col}_days_from_ref"] = (self.reference_date - date_series).dt.days

        X_out = X_out.drop(columns=self.date_columns_, errors="ignore")
        X_out = X_out.drop(columns=["customer_id"], errors="ignore")
        return X_out


def build_preprocessing_pipeline(reference_date: str = "2024-12-31") -> Pipeline:
    """Build leakage-safe preprocessing pipeline for mixed tabular features."""
    numeric_features = [
        "age",
        "txn_count",
        "total_amount",
        "avg_amount",
        "max_amount",
        "min_amount",
        "days_since_last_transaction",
        "login_count",
        "total_seconds_active",
        "avg_seconds_active",
        "max_seconds_active",
        "days_since_last_login_feature",
        "days_since_last_login",
        "customer_tenure_days",
        "signup_date_days_from_ref",
        "last_transaction_date_days_from_ref",
        "last_login_feature_date_days_from_ref",
        "last_login_date_days_from_ref",
    ]
    categorical_features = ["region"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    col_transform = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("date_engineering", DateFeatureEngineer(reference_date=reference_date)),
            ("preprocess", col_transform),
        ]
    )
