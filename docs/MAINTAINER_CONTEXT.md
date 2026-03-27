# Maintainer Context and Technical Notes

This document captures current implementation state, design rationale, and safe extension points.

## 0) Full Project Approach and Build Story

This section is intentionally long and narrative so a new maintainer can understand not only
what exists, but why the project was built this way.

### 0.1 Why this project was built the way it was

The project did not start as a pure API or pure notebook. It was deliberately built to bridge
the common gap between:

- data science experimentation (feature ideas, model explainability, evaluation)
- production engineering (modular code, stable interfaces, reproducibility, serving)

The goal was to avoid a "great notebook, weak system" outcome. We treated the project as a
small but realistic production template for churn prediction, where decisions were made to
maximize maintainability and onboarding clarity.

### 0.2 Core problem framing

The business framing is customer churn risk estimation. The system needs to:

1. build customer-level training rows from relational activity data,
2. produce robust, leakage-aware model features,
3. train and tune a classifier under churn imbalance constraints,
4. provide explanation outputs usable by stakeholders,
5. expose an inference endpoint that can be integrated by applications.

Because real data was not available, we used synthetic relational simulation with intentional
messiness to mimic realistic operational conditions.

### 0.3 Development strategy used

The approach followed a layered progression:

1. **Data realism first**:
   - create noisy relational sources (customers, transactions, logs),
   - define churn event with clear temporal logic,
   - aggregate to one-row-per-customer ABT.
2. **Feature and pipeline rigor**:
   - move transformations into sklearn-compatible components,
   - preserve train/inference parity with reusable pipeline objects,
   - minimize leakage by construction.
3. **Model + interpretation**:
   - tune with F1 objective,
   - add SHAP so "why predicted churn?" is first-class.
4. **Serving and software hardening**:
   - serialize pipeline artifact,
   - expose FastAPI endpoint with schema validation and logging,
   - add tests and containerization.
5. **Documentation and handoff quality**:
   - maintain recruiter-facing README,
   - maintain maintainer-facing deep technical context (this file).

This sequence was intentional: a robust model API is only meaningful if upstream feature logic
and leakage controls are reliable.

### 0.4 Data modeling philosophy

The simulator includes three relational tables to reflect common SaaS/e-commerce telemetry:

- static/semi-static profile table (`customers_df`),
- monetary event stream (`transactions_df`),
- behavior event stream (`logs_df`).

We intentionally added common data quality characteristics:

- missing demographic values (age),
- skewed monetary distributions (lognormal transaction values),
- sparse and uneven user activity patterns.

This avoids building a model against unrealistically clean data and forces handling of nulls,
recency behavior, and customer heterogeneity.

### 0.5 Churn definition philosophy

Churn was defined as **no login in the previous 30 days relative to a reference date**.

Why this matters:

- easy to reason about,
- realistic for product engagement contexts,
- directly translatable into features and interventions.

Potential production adaptation:

- domain-specific windows (e.g., 14/60/90 days),
- subscription cancellation integration,
- multi-stage churn states (at-risk vs churned).

### 0.6 ABT construction approach

ABT design principle: **one row per customer**.

Operationally:

1. Aggregate each event stream first,
2. then join aggregates to customer grain.

This prevents row explosion and creates a stable schema for both modeling and API inference.

Key feature classes:

- value/volume: total spend, average transaction value, transaction counts,
- recent intensity: spend/count in last 30 days,
- recency: days since last transaction / login,
- engagement: login counts and active seconds,
- demographic/context: region, age, tenure.

### 0.7 Pipeline and feature engineering philosophy

A major project choice was moving preprocessing into reusable sklearn components.

Why:

- train and inference see identical transformations,
- easier serialization and deployment,
- lower chance of train/serving skew.

Custom transformer responsibilities are explicit:

- `CustomerTenureTransformer`: creates tenure from dates and reference date,
- `DateRecencyTransformer`: converts date columns into model-friendly numeric recency signals.

Both were implemented as `BaseEstimator` + `TransformerMixin` so they integrate naturally with
`Pipeline`/`ColumnTransformer` and can be unit-tested.

### 0.8 Leakage prevention doctrine

Leakage risks were considered early because churn datasets are especially vulnerable to
time-window mistakes.

Enforced controls:

1. Split before fitting.
2. Fit transforms on train only.
3. Transform test/inference with fitted train objects.
4. Keep label logic and feature windows conceptually aligned with prediction time.

Any future feature addition should be reviewed with one question:
"Could this value be known at prediction time for the target horizon?"

### 0.9 Model selection and tuning rationale

Current estimator is Random Forest with class balancing and randomized search.

Why this baseline:

- robust with mixed tabular features,
- low preprocessing burden for non-linear effects,
- generally stable for medium-size tabular tasks.

Why F1 objective:

- churn tasks are commonly imbalanced,
- F1 better captures precision/recall balance than accuracy.

Known reality:

- synthetic data can produce unrealistically high metrics.
- real production quality must be judged on real holdout/backtest data.

### 0.10 Explainability strategy

Interpretability was not treated as optional; it is integrated in training flow.

SHAP is used to answer:

- global: which features matter most across customers?
- local: why did this specific customer score high risk?

A summary plot is generated to `reports/figures/shap_summary.png` so results are visible in docs
without requiring someone to rerun an analysis notebook.

### 0.11 API serving decisions

Serving is intentionally lightweight:

- model artifact loaded at startup,
- strict request schema using Pydantic,
- single `/predict` endpoint returning probability and binary class.

Logging was added for basic observability:

- startup status,
- request receipt,
- prediction outputs.

This is enough for local production-style operation and debugging, while remaining simple.

### 0.12 Testing strategy used

Tests currently focus on high-leverage failure points:

- custom transformer correctness,
- pipeline save/load + predict path.

Reasoning:

- custom transformers are where silent feature bugs often occur,
- serialization integrity is critical for API runtime reliability.

Future test expansion should include API contract tests and regression tests on feature schema.

### 0.13 Runtime constraints encountered and design response

Parallel CV workers caused instability in this environment (worker termination / access violations).

Mitigation applied:

- reduced search intensity defaults (`n_iter`, `cv`),
- set `n_jobs=1` for model and search to prioritize stability.

This is an environment-driven tradeoff, not a modeling preference. In a stable compute environment,
these can be scaled up.

### 0.14 Documentation strategy

Two documentation audiences are intentionally separated:

- `README.md`: concise, recruiter/stakeholder-facing pitch and run steps.
- `docs/MAINTAINER_CONTEXT.md`: deep engineering context and operational guidance.

This prevents one file from trying to satisfy incompatible needs.

### 0.15 How to think when extending this system

Recommended extension workflow:

1. Clarify prediction-time data availability.
2. Add/modify features in pipeline components, not ad-hoc scripts.
3. Re-run training + SHAP generation.
4. Validate tests and API compatibility.
5. Update this maintainer doc with rationale, not only code references.

If this discipline is followed, the project remains understandable as it grows.

### 0.16 Non-goals (current scope boundaries)

The current system intentionally does not include:

- streaming ingestion,
- online feature store joins,
- model registry platform integration,
- advanced MLOps orchestration,
- drift monitoring services.

Those are valid next steps, but were kept out to maintain a focused, clean baseline.

### 0.17 One-page mental model for new maintainers

If you remember only one thing, remember this:

**Everything revolves around preserving feature consistency and temporal correctness from ABT build
through training to API inference.**

That principle explains most current design choices.

---

## 1) Current Objective

The project demonstrates an end-to-end churn system that is both:

- analytically credible (temporal features, F1-oriented tuning, SHAP explainability)
- engineering-ready (modular `src/`, artifact serialization, FastAPI serving, tests, Docker)

The current setup uses synthetic relational data to simulate realistic business pipelines.

---

## 2) Current System Overview

### Data and feature foundation

Source files:

- `simulate_messy_data.py`
- `generate_messy_data.py` (import alias/entry wrapper)

Key behavior:

- Simulates relational tables:
  - `customers_df`
  - `transactions_df`
  - `logs_df`
- Builds one-row-per-customer ABT with joins and aggregation.
- Churn label: no login in the last 30 days from `reference_date`.
- Added stronger temporal transaction features:
  - `total_spend_last_30d`
  - `avg_transaction_value`
  - `days_since_last_transaction`
  - `txn_count_last_30d`

### Preprocessing and custom transformers

Source file:

- `src/pipeline.py`

Custom transformers:

- `CustomerTenureTransformer`:
  - derives `customer_tenure_days` from `signup_date` and `reference_date`
- `DateRecencyTransformer`:
  - converts raw dates into recency features
  - drops raw datetime columns and `customer_id`

Pipeline design:

- numeric imputation: median
- categorical handling: most-frequent imputation + one-hot (`region`)
- deterministic feature engineering inside sklearn pipeline for train/inference parity

### Training and interpretation

Source files:

- `src/train.py`
- `src/interpret.py`
- `scripts/serialize_pipeline.py`

Current modeling setup:

- estimator: `RandomForestClassifier(class_weight="balanced")`
- selection: `RandomizedSearchCV`
- objective: `f1`
- artifact: `models/churn_pipeline.joblib`
- SHAP summary output: `reports/figures/shap_summary.png`

### API serving

Source file:

- `app.py`

Current API behavior:

- startup loads `models/churn_pipeline.joblib`
- endpoints:
  - `GET /health`
  - `POST /predict`
- request model: `CustomerFeatures(BaseModel)` (Pydantic validation)
- response model: churn probability + binary prediction
- logging added for:
  - service startup
  - prediction request received
  - prediction result emitted

### Testing and packaging

- tests:
  - `tests/test_pipeline.py`
    - validates `CustomerTenureTransformer`
    - validates saved pipeline load + predict path
- Docker:
  - `Dockerfile` runs `uvicorn app:app`

---

## 3) Module Responsibilities

## `src/data.py`

- ingestion boundary for training data (`ingest_training_data`)
- currently backs onto synthetic generator
- includes `load_abt_csv` for persisted ABT workflows

## `src/data_gen.py`

- local data generation entry script
- writes ABT CSV to `data/abt.csv`

## `src/pipeline.py`

- owns all reusable preprocessing and custom transformations
- should be treated as source of truth for model input schema handling

## `src/train.py`

- builds model pipeline
- splits data leakage-safely
- tunes model hyperparameters
- evaluates holdout metrics
- saves trained artifact
- executable as script (`python src/train.py`)

## `src/interpret.py`

- SHAP helper functions for fitted pipeline
- currently writes summary plot for test set

## `scripts/serialize_pipeline.py`

- orchestration script for train + save + SHAP report
- preferred path to produce deployable model file

## `app.py`

- inference-only runtime
- no training logic
- strict schema input via Pydantic

---

## 4) Leakage Controls

Current leakage prevention strategy:

1. Split train/test before fitting (`train_test_split`).
2. Fit preprocessing only on train; apply to test/inference.
3. Date/tenure features are deterministic against a fixed reference date.
4. Target is not used in feature transformers.

High-risk future changes:

- adding post-outcome features into ABT windows
- changing date windows without updating offline/online parity
- bypassing pipeline and hand-transforming inference payloads

---

## 5) Operational Runbook

### Local setup

```bash
pip install -r requirements.txt
```

### Generate ABT CSV

```bash
python src/data_gen.py
```

Expected output:

- `data/abt.csv`

### Train model and generate SHAP summary

```bash
python src/train.py
```

Expected outputs:

- `models/churn_pipeline.joblib`
- `reports/figures/shap_summary.png`

### Start API

```bash
uvicorn app:app --reload
```

### Docker run

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---

## 6) Known Runtime Caveats

- On this environment, aggressive parallel CV caused worker termination in sklearn/loky.
- To stabilize execution:
  - `RandomizedSearchCV(n_jobs=1)`
  - model `n_jobs=1`
  - reduced search defaults (`n_iter=12`, `cv=3`)

If moved to a stronger and stable environment, these can be increased.

---

## 7) Maintenance Checklist

When editing feature logic:

- update `src/pipeline.py` only (avoid duplicate transforms elsewhere)
- rerun `python src/train.py`
- rerun `python -m pytest tests/test_pipeline.py`
- verify API still predicts with representative payload

When editing model/search:

- keep F1 objective unless business objective changes
- record tradeoff impacts (precision/recall, not only aggregate score)
- verify SHAP code still supports model output shape

When editing API schema:

- maintain backward compatibility where possible
- ensure required fields still align with pipeline input expectations

---

## 8) Recommended Next Steps

1. Add API tests with FastAPI `TestClient`.
2. Add CI pipeline (lint + tests + training smoke check).
3. Add threshold optimization by business cost matrix.
4. Add calibration and drift checks for production monitoring.
5. Replace synthetic ingestion with warehouse connector and data contracts.

---

## 9) Quick File Map

- `generate_messy_data.py`: wrapper alias for generator module
- `simulate_messy_data.py`: relational simulation + ABT
- `src/data.py`: ingestion interface
- `src/data_gen.py`: ABT file generation script
- `src/pipeline.py`: custom transformers + preprocessing pipeline
- `src/train.py`: training/tuning/evaluation/artifact script
- `src/interpret.py`: SHAP report generation
- `scripts/serialize_pipeline.py`: alternate train+save+SHAP orchestration
- `app.py`: FastAPI inference service
- `tests/test_pipeline.py`: unit tests for transformer and pipeline IO
- `README.md`: recruiter-facing project summary
