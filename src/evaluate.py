from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data import DataConfig, ingest_training_data
from src.train import split_data

F1_THRESHOLD = 0.70
MODEL_PATH = Path("models/churn_pipeline.joblib")
METRICS_PATH = Path("model_metrics.json")


def evaluate() -> int:
    """Evaluate trained model and enforce an F1 quality gate."""
    if not MODEL_PATH.exists():
        print(f"Error: {MODEL_PATH} not found. Did training run?")
        return 1

    pipeline = joblib.load(MODEL_PATH)

    data_cfg = DataConfig(n_customers=2500, seed=7, reference_date="2024-12-31")
    abt_df = ingest_training_data(data_cfg)
    _, X_test, _, y_test = split_data(
        abt_df=abt_df,
        target_col="churned",
        test_size=0.2,
        random_state=42,
    )

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Model metrics:", metrics)

    if metrics["f1"] < F1_THRESHOLD:
        print(
            f"Quality gate failed. F1 {metrics['f1']:.4f} < threshold {F1_THRESHOLD:.2f}. "
            "Stopping pipeline."
        )
        return 1

    print(f"Quality gate passed. F1 {metrics['f1']:.4f} >= {F1_THRESHOLD:.2f}.")
    return 0


if __name__ == "__main__":
    sys.exit(evaluate())
