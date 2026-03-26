from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.features import build_preprocessing_pipeline


@dataclass
class TrainConfig:
    target_col: str = "churned"
    test_size: float = 0.2
    random_state: int = 42
    n_iter: int = 20
    cv: int = 5
    reference_date: str = "2024-12-31"


def build_model_pipeline(reference_date: str = "2024-12-31") -> Pipeline:
    """Build end-to-end preprocessing + model pipeline."""
    preprocess = build_preprocessing_pipeline(reference_date=reference_date)
    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


def split_data(
    abt_df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create train/test split before fitting pipeline to avoid leakage."""
    if target_col not in abt_df.columns:
        raise ValueError(f"Target column '{target_col}' is missing.")

    X = abt_df.drop(columns=[target_col])
    y = abt_df[target_col].astype(int)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def tune_and_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    reference_date: str = "2024-12-31",
    n_iter: int = 20,
    cv: int = 5,
    random_state: int = 42,
) -> RandomizedSearchCV:
    """Tune RandomForest using F1 objective for churn imbalance."""
    pipeline = build_model_pipeline(reference_date=reference_date)
    param_distributions = {
        "model__n_estimators": [150, 250, 400, 600],
        "model__max_depth": [None, 6, 10, 14, 18, 24],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", None],
        "model__bootstrap": [True, False],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1",
        n_jobs=-1,
        cv=cv,
        random_state=random_state,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_model(
    fitted_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate fitted pipeline on holdout test data."""
    y_pred = fitted_pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return {"f1": float(f1), "report": report}


def save_pipeline(pipeline: Pipeline, output_path: str | Path) -> Path:
    """Serialize trained pipeline with joblib."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out)
    return out
