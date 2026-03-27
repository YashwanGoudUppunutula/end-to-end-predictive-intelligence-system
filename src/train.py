from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import build_preprocessing_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    target_col: str = "churned"
    test_size: float = 0.2
    random_state: int = 42
    n_iter: int = 12
    cv: int = 3
    reference_date: str = "2024-12-31"


def build_model_pipeline(reference_date: str = "2024-12-31") -> Pipeline:
    """Build end-to-end preprocessing + model pipeline."""
    preprocess = build_preprocessing_pipeline(reference_date=reference_date)
    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
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
        n_jobs=1,
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


def main() -> None:
    from src.data import DataConfig, ingest_training_data
    from src.interpret import generate_shap_summary_plot, save_shap_explainer_bundle

    data_cfg = DataConfig(n_customers=2500, seed=7, reference_date="2024-12-31")
    train_cfg = TrainConfig(reference_date=data_cfg.reference_date)

    logger.info("Loading training data...")
    abt_df = ingest_training_data(data_cfg)
    logger.info("Data loaded with shape: %s", abt_df.shape)

    logger.info("Creating train/test split...")
    X_train, X_test, y_train, y_test = split_data(
        abt_df=abt_df,
        target_col=train_cfg.target_col,
        test_size=train_cfg.test_size,
        random_state=train_cfg.random_state,
    )
    logger.info("Split complete. X_train=%s, X_test=%s", X_train.shape, X_test.shape)

    logger.info(
        "Starting hyperparameter search (n_iter=%d, cv=%d)...",
        train_cfg.n_iter,
        train_cfg.cv,
    )
    search = tune_and_train(
        X_train=X_train,
        y_train=y_train,
        reference_date=train_cfg.reference_date,
        n_iter=train_cfg.n_iter,
        cv=train_cfg.cv,
        random_state=train_cfg.random_state,
    )
    logger.info("Training complete. Best CV F1 Score: %.4f", search.best_score_)
    best_pipeline = search.best_estimator_
    metrics = evaluate_model(best_pipeline, X_test, y_test)
    logger.info("Holdout evaluation complete. Test F1 Score: %.4f", metrics["f1"])

    logger.info("Saving trained pipeline artifact...")
    output_path = save_pipeline(best_pipeline, "models/churn_pipeline.joblib")
    logger.info("Pipeline saved to %s", output_path)

    logger.info("Generating SHAP summary figure...")
    shap_path = generate_shap_summary_plot(best_pipeline, X_test, "reports/figures/shap_summary.png")
    logger.info("SHAP summary saved to %s", shap_path)
    explainer_path = save_shap_explainer_bundle(best_pipeline, "models/shap_explainer.joblib")
    logger.info("SHAP explainer bundle saved to %s", explainer_path)

    logger.info("Best parameters: %s", search.best_params_)
    logger.info("Classification report:\n%s", metrics["report"])


if __name__ == "__main__":
    main()
