from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class CustomerTenureTransformer(BaseEstimator, TransformerMixin):
    """Create customer_tenure_days from signup_date and a reference date."""

    def __init__(self, reference_date: str = "2024-12-31") -> None:
        self.reference_date = pd.Timestamp(reference_date)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CustomerTenureTransformer":
        if "signup_date" not in X.columns:
            raise ValueError("signup_date is required for CustomerTenureTransformer.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out["signup_date"] = pd.to_datetime(out["signup_date"], errors="coerce")
        out["customer_tenure_days"] = (self.reference_date - out["signup_date"]).dt.days.clip(lower=0)
        return out


class DateRecencyTransformer(BaseEstimator, TransformerMixin):
    """Create recency features from datetime columns and drop raw date columns."""

    def __init__(self, reference_date: str = "2024-12-31") -> None:
        self.reference_date = pd.Timestamp(reference_date)
        self.date_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DateRecencyTransformer":
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
        out = X.copy()
        for col in self.date_columns_:
            date_series = pd.to_datetime(out[col], errors="coerce")
            out[f"{col}_days_from_ref"] = (self.reference_date - date_series).dt.days

        out = out.drop(columns=self.date_columns_, errors="ignore")
        out = out.drop(columns=["customer_id"], errors="ignore")
        return out


class BehavioralInteractionTransformer(BaseEstimator, TransformerMixin):
    """Add domain-driven interaction features from spend and engagement behavior."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BehavioralInteractionTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if "total_amount" in out.columns and "login_count" in out.columns:
            out["spend_per_login"] = out["total_amount"] / (out["login_count"].fillna(0) + 1.0)
        else:
            out["spend_per_login"] = pd.NA

        if "total_amount" in out.columns and "total_seconds_active" in out.columns:
            out["spend_per_second_active"] = out["total_amount"] / (
                out["total_seconds_active"].fillna(0) + 1.0
            )
        else:
            out["spend_per_second_active"] = pd.NA

        return out


def build_preprocessing_pipeline(reference_date: str = "2024-12-31") -> Pipeline:
    numeric_features = [
        "age",
        "txn_count",
        "total_amount",
        "avg_amount",
        "avg_transaction_value",
        "max_amount",
        "min_amount",
        "total_spend_last_30d",
        "txn_count_last_30d",
        "days_since_last_transaction",
        "login_count",
        "total_seconds_active",
        "avg_seconds_active",
        "max_seconds_active",
        "customer_tenure_days",
        "spend_per_login",
        "spend_per_second_active",
        "signup_date_days_from_ref",
        "last_transaction_date_days_from_ref",
    ]
    categorical_features = ["region"]

    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
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
            ("tenure", CustomerTenureTransformer(reference_date=reference_date)),
            ("date_recency", DateRecencyTransformer(reference_date=reference_date)),
            ("behavior_interactions", BehavioralInteractionTransformer()),
            ("preprocess", col_transform),
        ]
    )
