# Maintainer Context and Technical Notes

This document is intended to preserve implementation context so future updates can be made safely and quickly.

## 1. Project objective

Build a production-grade customer churn prediction system that is:

- analytically strong (clear churn definition, robust validation, interpretable outputs)
- engineering-ready (modular package, reproducible scripts, API serving contract)

The repository was developed iteratively from relational data simulation to deployable inference.

---

## 2. Scope completed

### Data simulation and ABT generation

Implemented in `simulate_messy_data.py`.

Main behaviors:

- simulates three relational tables:
  - customer demographics (`customers_df`)
  - transaction logs (`transactions_df`)
  - app usage logs (`logs_df`)
- injects realistic messiness:
  - missing values in `age`
  - skewed transaction amounts
  - irregular activity cadence
- defines churn:
  - `churned = 1` when no login in last 30 days from `reference_date`
- builds one-row-per-customer ABT using aggregated SQL-style joins.

Key anti-footgun choices:

- feature aggregations happen before joining (prevents row explosion)
- ABT uniqueness on `customer_id` enforced
- recency nulls filled with tenure-driven defaults for consistency.

### EDA and preprocessing

Implemented in `eda_feature_pipeline.py` and productionized in `src/features.py`.

EDA outputs:

- churn rate by region
- login count by churn class
- correlation heatmap

Preprocessing design:

- custom date feature engineering (`DateFeatureEngineer`)
- median imputation for numerical columns
- one-hot encoding for categorical region with unknown-safe handling
- explicit train/test split before fit to prevent leakage.

### Modeling and interpretation

Implemented in `model_training_interpretation.py` and productionized in `src/train.py`.

Details:

- model: `RandomForestClassifier(class_weight="balanced")`
- tuning: `RandomizedSearchCV`
- objective: `f1`
- explainability: SHAP
  - global summary on test set
  - local feature contribution chart for highest-risk customer.

### Deployment scaffolding

- training artifact serialization script: `scripts/serialize_pipeline.py`
- FastAPI app: `app.py`
  - startup model load from `models/churn_pipeline.joblib`
  - `POST /predict` returns probability and class label.

---

## 3. Architecture and module responsibilities

## `src/data.py`

Primary ingestion boundary for model development.

- `ingest_training_data()` currently returns synthetic ABT
- designed to be replaced by DB/warehouse ingestion without changing training interfaces
- `load_abt_csv()` for local ABT experiments.

## `src/features.py`

All feature transformations required before model.

- owns deterministic date-derived features
- owns categorical and numeric preprocessing policy
- keeps pipeline portable for train + inference parity.

## `src/train.py`

Training orchestration and model persistence.

- split function
- randomized hyperparameter search
- holdout evaluation
- serialization helper (`joblib`).

## `scripts/serialize_pipeline.py`

Convenience script for one-command model build artifact creation.

Use this script before starting API serving if `models/churn_pipeline.joblib` does not exist.

## `app.py`

Inference API only.

- no training logic inside API process
- loads already-trained artifact
- converts incoming date strings to datetime with coercion.

---

## 4. Data leakage controls in place

1. Train/test split is performed before fitting transformers/models.
2. Preprocessing is encapsulated in sklearn pipelines.
3. `fit_transform` on train and `transform` on test only.
4. SHAP is computed on held-out test features from trained pipeline.

Potential future leakage risks:

- if future features include post-outcome events
- if reference-date logic diverges between offline and online systems
- if ABT generation is modified without maintaining strict temporal boundaries.

---

## 5. API payload assumptions and implications

The current `/predict` endpoint expects ABT-style flattened features, not raw event streams.

This is acceptable for:

- batch scoring pipelines that precompute ABT features
- systems where feature store already materializes customer aggregates.

Not yet implemented:

- real-time event ingestion and aggregation inside API
- online feature store lookups
- temporal feature freshness checks.

---

## 6. Reproducibility and environment

Dependencies are listed in `requirements.txt`.

Reproducibility controls:

- deterministic seeds in simulation and model search
- immutable module boundaries for preprocessing and training
- script-driven artifact generation.

Non-deterministic caveats:

- thread scheduling in parallel jobs can create slight run-to-run timing differences
- synthetic dataset quality can change if simulation parameters are edited.

---

## 7. Operational runbook

### Generate model artifact

```bash
python scripts/serialize_pipeline.py
```

Expected:

- console prints best params, CV F1, test F1
- file `models/churn_pipeline.joblib` created.

### Start API

```bash
uvicorn app:app --reload
```

Health check:

```bash
GET /health
```

Prediction:

```bash
POST /predict
```

Failure mode to expect:

- if model file missing, startup fails with a clear message asking to run serialization script.

---

## 8. Maintenance checklist (when changing code)

When editing preprocessing:

- keep train/inference schema parity
- confirm transformed feature names still align
- rerun `scripts/serialize_pipeline.py`
- retest API with representative payloads.

When editing model:

- ensure scoring objective remains business-appropriate
- evaluate confusion tradeoffs, not just overall metrics
- confirm SHAP still works with chosen estimator.

When editing API:

- preserve backward compatibility or version endpoint
- validate date parsing behavior
- verify response schema stability.

---

## 9. Recommended next enhancements

1. Add automated tests:
   - unit tests for `DateFeatureEngineer`
   - integration test for training + artifact load
   - API tests with FastAPI `TestClient`.
2. Add Docker support:
   - reproducible runtime for local and cloud deploy.
3. Add CI workflow:
   - lint, test, and artifact validation checks.
4. Add threshold tuning and model calibration:
   - optimize for retention campaign economics.
5. Replace synthetic ingest with production connectors:
   - SQL extraction, data contracts, and schema checks.

---

## 10. Quick file map

- `simulate_messy_data.py`: simulation + ABT construction
- `eda_feature_pipeline.py`: EDA visuals + preprocessing demo
- `model_training_interpretation.py`: model search + SHAP demo
- `src/data.py`: ingestion interface
- `src/features.py`: preprocessing components
- `src/train.py`: training utilities
- `scripts/serialize_pipeline.py`: artifact creation script
- `app.py`: FastAPI inference service
- `README.md`: public-facing project guide
- `docs/MAINTAINER_CONTEXT.md`: internal technical context
