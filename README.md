# End-to-End Predictive Intelligence System

Production-focused customer churn prediction system built with Python, scikit-learn, SHAP, and FastAPI.

## At-a-Glance

- Modular ML codebase (`src/data.py`, `src/features.py`, `src/train.py`)
- Leakage-safe train/test workflow with deterministic preprocessing pipeline
- Hyperparameter tuning optimized for F1 (imbalanced churn scenario)
- Explainability with SHAP (global and customer-level)
- Deployable inference endpoint with FastAPI (`POST /predict`)

## System Architecture

![System Architecture](images/system_architecture.png)

Flow: relational data -> ABT -> preprocessing pipeline -> tuned model -> serialized artifact -> FastAPI scoring.

## Explainability Proof (SHAP)

### Global Feature Impact

![SHAP Summary Plot](images/shap_summary_test_set.png)

Business interpretation: the model is primarily driven by activity and recency behavior (for example, lower engagement and longer inactivity push predictions toward churn risk). This gives retention teams clear levers: prioritize users with declining recency/activity signals.

### High-Risk Customer Explanation

![High-Risk Customer SHAP](images/shap_high_risk_customer.png)

Business interpretation: this chart explains one specific high-risk prediction by ranking features that increased/decreased churn probability. It can directly support targeted intervention playbooks for account managers.

---

## Project Structure

```text
.
|-- app.py
|-- simulate_messy_data.py
|-- eda_feature_pipeline.py
|-- model_training_interpretation.py
|-- requirements.txt
|-- images
|   |-- system_architecture.png
|   |-- shap_summary_test_set.png
|   `-- shap_high_risk_customer.png
|-- src
|   |-- __init__.py
|   |-- data.py
|   |-- features.py
|   `-- train.py
|-- scripts
|   `-- serialize_pipeline.py
`-- docs
    `-- MAINTAINER_CONTEXT.md
```

## Quick Start

### 1) Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Train and serialize model

```bash
python scripts/serialize_pipeline.py
```

Output artifact:

- `models/churn_pipeline.joblib`

### 3) Run API

```bash
uvicorn app:app --reload
```

Endpoints:

- `GET /health`
- `POST /predict`

---

## Reproducibility and Engineering Rigor

- exact package versions pinned in `requirements.txt`
- preprocessing and model tightly coupled in one sklearn pipeline
- no leakage: split before fit, transform test only with fitted training pipeline
- runtime artifacts excluded with `.gitignore`

---

## API Example

### Request

```json
{
  "customer_id": 101,
  "signup_date": "2024-01-15",
  "region": "north",
  "age": 38,
  "txn_count": 5,
  "total_amount": 420.5,
  "avg_amount": 84.1,
  "max_amount": 150,
  "min_amount": 20,
  "last_transaction_date": "2024-12-10",
  "days_since_last_transaction": 21,
  "login_count": 14,
  "total_seconds_active": 9000,
  "avg_seconds_active": 642,
  "max_seconds_active": 1200,
  "last_login_feature_date": "2024-12-20",
  "days_since_last_login_feature": 11,
  "last_login_date": "2024-12-20",
  "days_since_last_login": 11,
  "customer_tenure_days": 351
}
```

### Response

```json
{
  "churn_probability": 0.8731,
  "churn_prediction": 1
}
```

---

## Deep Technical Context

For full implementation rationale, maintenance runbook, and future roadmap:

- `docs/MAINTAINER_CONTEXT.md`
