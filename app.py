from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path("models/churn_pipeline.joblib")
app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")


class PredictRequest(BaseModel):
    customer_id: Optional[int] = Field(default=None, examples=[1001])
    signup_date: Optional[str] = Field(default=None, examples=["2024-01-15"])
    region: Optional[str] = Field(default=None, examples=["north"])
    age: Optional[float] = Field(default=None, examples=[39])
    txn_count: Optional[float] = None
    total_amount: Optional[float] = None
    avg_amount: Optional[float] = None
    max_amount: Optional[float] = None
    min_amount: Optional[float] = None
    last_transaction_date: Optional[str] = Field(default=None, examples=["2024-12-10"])
    days_since_last_transaction: Optional[float] = None
    txn_cat_count_addon: Optional[float] = None
    txn_cat_count_discount: Optional[float] = None
    txn_cat_count_hardware: Optional[float] = None
    txn_cat_count_subscription: Optional[float] = None
    txn_cat_count_support: Optional[float] = None
    login_count: Optional[float] = None
    total_seconds_active: Optional[float] = None
    avg_seconds_active: Optional[float] = None
    max_seconds_active: Optional[float] = None
    last_login_feature_date: Optional[str] = Field(default=None, examples=["2024-12-20"])
    days_since_last_login_feature: Optional[float] = None
    last_login_date: Optional[str] = Field(default=None, examples=["2024-12-20"])
    days_since_last_login: Optional[float] = None
    customer_tenure_days: Optional[float] = None


class PredictResponse(BaseModel):
    churn_probability: float
    churn_prediction: int


def _load_model() -> object:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. Run scripts/serialize_pipeline.py first."
        )
    return joblib.load(MODEL_PATH)


@app.on_event("startup")
def startup() -> None:
    app.state.model = _load_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        model = app.state.model
        row = payload.model_dump()
        X = pd.DataFrame([row])
        for col in ["signup_date", "last_transaction_date", "last_login_feature_date", "last_login_date"]:
            if col in X.columns:
                X[col] = pd.to_datetime(X[col], errors="coerce")

        proba = float(model.predict_proba(X)[0, 1])
        pred = int(proba >= 0.5)
        return PredictResponse(churn_probability=proba, churn_prediction=pred)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
