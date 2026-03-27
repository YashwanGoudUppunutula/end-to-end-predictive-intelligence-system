from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path("models/churn_pipeline.joblib")
app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")
logger = logging.getLogger("churn_api")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


class CustomerFeatures(BaseModel):
    customer_id: Optional[int] = Field(default=None, examples=[1001])
    signup_date: str = Field(examples=["2024-01-15"])
    region: str = Field(examples=["north"])
    age: float = Field(examples=[39])
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


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float


class ExplainResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    top_contributors: list[FeatureContribution]


def _load_model() -> object:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. Run scripts/serialize_pipeline.py first."
        )
    return joblib.load(MODEL_PATH)


def _extract_positive_class_shap(shap_values, n_features: int) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    if arr.ndim == 2 and arr.shape[1] == n_features:
        return arr
    raise ValueError(f"Unexpected SHAP shape: {arr.shape}")


@app.on_event("startup")
def startup() -> None:
    app.state.model = _load_model()
    logger.info("Service started and model loaded from %s", MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(features: CustomerFeatures) -> PredictResponse:
    try:
        logger.info("Received prediction request")
        model = app.state.model
        row = features.model_dump()
        X = pd.DataFrame([row])
        for col in ["signup_date", "last_transaction_date", "last_login_feature_date", "last_login_date"]:
            if col in X.columns:
                X[col] = pd.to_datetime(X[col], errors="coerce")

        proba = float(model.predict_proba(X)[0, 1])
        pred = int(proba >= 0.5)
        logger.info("Prediction generated churn_probability=%.4f churn_prediction=%d", proba, pred)
        return PredictResponse(churn_probability=proba, churn_prediction=pred)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/explain", response_model=ExplainResponse)
def explain(features: CustomerFeatures, top_k: int = 10) -> ExplainResponse:
    try:
        model_pipeline = app.state.model
        row = features.model_dump()
        X = pd.DataFrame([row])
        for col in ["signup_date", "last_transaction_date", "last_login_feature_date", "last_login_date"]:
            if col in X.columns:
                X[col] = pd.to_datetime(X[col], errors="coerce")

        proba = float(model_pipeline.predict_proba(X)[0, 1])
        pred = int(proba >= 0.5)

        preprocess = model_pipeline.named_steps["preprocess"]
        model = model_pipeline.named_steps["model"]

        transformed = preprocess.transform(X)
        dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
        feature_names = list(preprocess.named_steps["preprocess"].get_feature_names_out())

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(dense)
        shap_pos = _extract_positive_class_shap(shap_values, len(feature_names))
        one_row = shap_pos[0]

        top_k = max(1, min(top_k, len(feature_names)))
        order = np.argsort(np.abs(one_row))[::-1][:top_k]
        contributors = [
            FeatureContribution(feature=feature_names[idx], shap_value=float(one_row[idx]))
            for idx in order
        ]

        logger.info("Explanation generated churn_probability=%.4f top_k=%d", proba, top_k)
        return ExplainResponse(
            churn_probability=proba,
            churn_prediction=pred,
            top_contributors=contributors,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}") from exc
