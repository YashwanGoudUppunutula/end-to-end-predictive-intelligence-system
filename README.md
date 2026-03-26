# End-to-End Predictive Intelligence System

A production-oriented customer churn prediction system that demonstrates the full ML lifecycle:

- relational data simulation and analytical base table (ABT) construction
- EDA and leakage-safe preprocessing
- model training and hyperparameter tuning
- SHAP-based explainability (global + customer-level)
- pipeline serialization for deployment
- FastAPI inference endpoint

## Why this project exists

This repository is built as a complete reference implementation for turning data science work into deployable software. It is designed to balance:

- **Analytical rigor**: interpretable features, robust validation, F1 optimization for imbalanced churn
- **Engineering rigor**: modular Python package layout, reproducible scripts, clear API serving path

---

## Project Structure

```text
.
|-- app.py
|-- simulate_messy_data.py
|-- eda_feature_pipeline.py
|-- model_training_interpretation.py
|-- requirements.txt
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

### Core modules

- `src/data.py`
  - training data ingestion entrypoint (`ingest_training_data`)
  - currently sources ABT from synthetic relational simulator
- `src/features.py`
  - reusable preprocessing pipeline (`build_preprocessing_pipeline`)
  - custom date transformer (`DateFeatureEngineer`)
- `src/train.py`
  - model pipeline assembly, split logic, randomized hyperparameter tuning, evaluation, model save
- `scripts/serialize_pipeline.py`
  - script to train, evaluate, and serialize best pipeline to `models/churn_pipeline.joblib`
- `app.py`
  - FastAPI service with `/health` and `/predict`

---

## End-to-End Workflow

### 1) Generate relational churn data and ABT

```bash
python simulate_messy_data.py
```

Outputs:

- `customers_df`: demographic table with missing ages
- `transactions_df`: event-level transactions
- `logs_df`: event-level app activity
- ABT with one row per customer, engineered aggregates, churn label

### 2) Run EDA and preprocessing pipeline checks

```bash
python eda_feature_pipeline.py
```

Creates visual outputs under `eda_outputs/`:

- churn rate by region
- login count distribution by churn
- numeric correlation heatmap

### 3) Train tuned model and SHAP explanations

```bash
python model_training_interpretation.py
```

Creates model explainability outputs under `model_outputs/`:

- `shap_summary_test_set.png`
- `shap_high_risk_customer.png`

### 4) Serialize deployment artifact

```bash
python scripts/serialize_pipeline.py
```

Creates:

- `models/churn_pipeline.joblib`

### 5) Serve model with FastAPI

```bash
uvicorn app:app --reload
```

Endpoints:

- `GET /health`
- `POST /predict`

---

## Installation

### Prerequisites

- Python 3.10+ (tested in Python 3.13 environment)
- pip

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## API Contract

### POST `/predict`

Accepts raw customer feature payload (ABT-style fields expected by the pipeline).

Example request:

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

Example response:

```json
{
  "churn_probability": 0.8731,
  "churn_prediction": 1
}
```

---

## Leakage Prevention Strategy

The project explicitly avoids common leakage pitfalls:

- data split (`train_test_split`) occurs before any fitting
- preprocessing pipeline is fit on training only, then applied to test/inference
- date features are transformed deterministically via reference-date recency logic
- target is not used in feature transformations

---

## Modeling Notes

- classifier: `RandomForestClassifier`
- tuning: `RandomizedSearchCV`
- objective: `f1` (appropriate for imbalanced churn labels)
- interpretability:
  - global SHAP summary for population-level feature influence
  - local SHAP analysis for one highest-risk customer

---

## Repository Hygiene

- runtime artifacts are ignored via `.gitignore`:
  - `__pycache__/`
  - `eda_outputs/`
  - `model_outputs/`
  - `models/`
  - `*.joblib`, `*.pkl`

This keeps Git history clean and reproducible.

---

## What to improve next

- add unit and integration tests for `src/features.py`, `src/train.py`, `app.py`
- add request/response schema validation tests for API
- add Dockerfile and CI pipeline (lint, test, build)
- replace synthetic ingestion with production data source
- add threshold optimization and calibration for business-specific cost curves

---

## Maintainer Notes

For detailed implementation history, architecture rationale, and maintenance guidance, see:

- `docs/MAINTAINER_CONTEXT.md`
