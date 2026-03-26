from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DataConfig, ingest_training_data
from src.train import TrainConfig, evaluate_model, save_pipeline, split_data, tune_and_train


def main() -> None:
    data_cfg = DataConfig(n_customers=2500, seed=7, reference_date="2024-12-31")
    train_cfg = TrainConfig(
        target_col="churned",
        test_size=0.2,
        random_state=42,
        n_iter=20,
        cv=5,
        reference_date=data_cfg.reference_date,
    )

    abt_df = ingest_training_data(data_cfg)
    X_train, X_test, y_train, y_test = split_data(
        abt_df=abt_df,
        target_col=train_cfg.target_col,
        test_size=train_cfg.test_size,
        random_state=train_cfg.random_state,
    )

    search = tune_and_train(
        X_train=X_train,
        y_train=y_train,
        reference_date=train_cfg.reference_date,
        n_iter=train_cfg.n_iter,
        cv=train_cfg.cv,
        random_state=train_cfg.random_state,
    )
    best_pipeline = search.best_estimator_
    metrics = evaluate_model(best_pipeline, X_test, y_test)

    output_path = save_pipeline(best_pipeline, "models/churn_pipeline.joblib")

    print("Best params:", search.best_params_)
    print(f"Best CV F1: {search.best_score_:.4f}")
    print(f"Test F1: {metrics['f1']:.4f}")
    print("\nClassification report:")
    print(metrics["report"])
    print(f"\nSaved pipeline to: {output_path}")


if __name__ == "__main__":
    main()
