from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.pipeline import CustomerTenureTransformer
from src.train import build_model_pipeline


def _dummy_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_id": 1,
                "signup_date": "2024-01-01",
                "region": "north",
                "age": 30.0,
                "txn_count": 2.0,
                "total_amount": 180.0,
                "avg_amount": 90.0,
                "avg_transaction_value": 90.0,
                "max_amount": 100.0,
                "min_amount": 80.0,
                "total_spend_last_30d": 40.0,
                "txn_count_last_30d": 1.0,
                "last_transaction_date": "2024-12-20",
                "days_since_last_transaction": 11.0,
                "txn_cat_count_addon": 0.0,
                "txn_cat_count_discount": 0.0,
                "txn_cat_count_hardware": 0.0,
                "txn_cat_count_subscription": 1.0,
                "txn_cat_count_support": 1.0,
                "login_count": 9.0,
                "total_seconds_active": 3000.0,
                "avg_seconds_active": 333.0,
                "max_seconds_active": 600.0,
                "last_login_feature_date": "2024-12-25",
                "days_since_last_login_feature": 6.0,
                "last_login_date": "2024-12-25",
                "days_since_last_login": 6.0,
                "churned": 0,
            },
            {
                "customer_id": 2,
                "signup_date": "2024-01-10",
                "region": "south",
                "age": 48.0,
                "txn_count": 1.0,
                "total_amount": 30.0,
                "avg_amount": 30.0,
                "avg_transaction_value": 30.0,
                "max_amount": 30.0,
                "min_amount": 30.0,
                "total_spend_last_30d": 0.0,
                "txn_count_last_30d": 0.0,
                "last_transaction_date": "2024-10-15",
                "days_since_last_transaction": 77.0,
                "txn_cat_count_addon": 0.0,
                "txn_cat_count_discount": 1.0,
                "txn_cat_count_hardware": 0.0,
                "txn_cat_count_subscription": 0.0,
                "txn_cat_count_support": 0.0,
                "login_count": 1.0,
                "total_seconds_active": 120.0,
                "avg_seconds_active": 120.0,
                "max_seconds_active": 120.0,
                "last_login_feature_date": "2024-10-01",
                "days_since_last_login_feature": 91.0,
                "last_login_date": "2024-10-01",
                "days_since_last_login": 91.0,
                "churned": 1,
            },
        ]
    )


def test_customer_tenure_transformer_creates_expected_days() -> None:
    transformer = CustomerTenureTransformer(reference_date="2024-12-31")
    inp = pd.DataFrame({"signup_date": ["2024-12-01", "2024-01-01"]})
    out = transformer.fit_transform(inp)
    assert "customer_tenure_days" in out.columns
    assert int(out.loc[0, "customer_tenure_days"]) == 30
    assert int(out.loc[1, "customer_tenure_days"]) == 365


def test_saved_pipeline_can_be_loaded_and_predict(tmp_path: Path) -> None:
    df = _dummy_rows()
    X = df.drop(columns=["churned"])
    y = df["churned"]

    pipeline = build_model_pipeline(reference_date="2024-12-31")
    pipeline.fit(X, y)

    model_path = tmp_path / "churn_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    loaded = joblib.load(model_path)
    preds = loaded.predict(X.iloc[[0]])
    probas = loaded.predict_proba(X.iloc[[0]])

    assert preds.shape == (1,)
    assert probas.shape == (1, 2)
