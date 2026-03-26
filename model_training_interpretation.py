from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from eda_feature_pipeline import build_preprocessing_pipeline
from simulate_messy_data import SimulationConfig, simulate_messy_data


def build_model_pipeline(reference_date: str = "2024-12-31") -> Pipeline:
    """Create end-to-end pipeline: preprocessing + classifier."""
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


def tune_model_with_randomized_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    reference_date: str,
    n_iter: int = 25,
) -> RandomizedSearchCV:
    """Tune RandomForest hyperparameters optimizing for F1 score."""
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
        cv=5,
        random_state=42,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def _to_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return X


def _extract_positive_class_shap(shap_values, n_features: int) -> np.ndarray:
    """
    Normalize SHAP outputs to a 2D matrix for positive class.

    Handles common outputs:
    - list of class arrays (TreeExplainer older behavior)
    - 3D ndarray [n_samples, n_features, n_classes]
    - 2D ndarray [n_samples, n_features]
    """
    if isinstance(shap_values, list):
        if len(shap_values) == 0:
            raise ValueError("Empty SHAP values list returned by explainer.")
        return np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])

    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim == 3:
        return shap_arr[:, :, 1]
    if shap_arr.ndim == 2 and shap_arr.shape[1] == n_features:
        return shap_arr
    raise ValueError(f"Unexpected SHAP shape: {shap_arr.shape}")


def compute_and_visualize_shap(
    best_pipeline: Pipeline,
    X_test: pd.DataFrame,
    output_dir: str = "model_outputs",
) -> dict:
    """Create global and local SHAP visualizations for test predictions."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    preprocess_pipe = best_pipeline.named_steps["preprocess"]
    model = best_pipeline.named_steps["model"]

    X_test_transformed = preprocess_pipe.transform(X_test)
    X_test_dense = _to_dense(X_test_transformed)

    column_transformer = preprocess_pipe.named_steps["preprocess"]
    feature_names = list(column_transformer.get_feature_names_out())

    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(X_test_dense)
    shap_values_pos = _extract_positive_class_shap(raw_shap_values, len(feature_names))

    # Global explanation: SHAP summary plot (beeswarm).
    plt.figure(figsize=(11, 8))
    shap.summary_plot(
        shap_values_pos,
        X_test_dense,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    global_plot_path = str(Path(output_dir) / "shap_summary_test_set.png")
    plt.tight_layout()
    plt.savefig(global_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Identify one high-risk customer (max predicted churn probability).
    churn_proba = best_pipeline.predict_proba(X_test)[:, 1]
    high_risk_position = int(np.argmax(churn_proba))
    high_risk_row = X_test.iloc[high_risk_position]
    high_risk_customer_id = (
        int(high_risk_row["customer_id"]) if "customer_id" in X_test.columns else high_risk_position
    )

    instance_shap = shap_values_pos[high_risk_position]
    abs_rank = np.argsort(np.abs(instance_shap))[::-1][:15]
    top_features = [feature_names[idx] for idx in abs_rank]
    top_values = instance_shap[abs_rank]

    # Local explanation: top SHAP contributors for one high-risk customer.
    plt.figure(figsize=(10, 6))
    colors = ["#d62728" if val > 0 else "#1f77b4" for val in top_values]
    plt.barh(top_features[::-1], top_values[::-1], color=colors[::-1])
    plt.axvline(0, color="black", linewidth=1)
    plt.title(
        f"Top SHAP Contributions for High-Risk Customer {high_risk_customer_id}\n"
        f"Predicted churn probability = {churn_proba[high_risk_position]:.3f}"
    )
    plt.xlabel("SHAP value impact on churn prediction")
    plt.tight_layout()
    local_plot_path = str(Path(output_dir) / "shap_high_risk_customer.png")
    plt.savefig(local_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "global_plot": global_plot_path,
        "local_plot": local_plot_path,
        "high_risk_customer_id": high_risk_customer_id,
        "high_risk_probability": float(churn_proba[high_risk_position]),
    }


def main() -> None:
    cfg = SimulationConfig(n_customers=2000, seed=7, reference_date="2024-12-31")
    _, _, _, abt_df = simulate_messy_data(cfg)

    X = abt_df.drop(columns=["churned"])
    y = abt_df["churned"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    search = tune_model_with_randomized_search(
        X_train=X_train,
        y_train=y_train,
        reference_date=cfg.reference_date,
        n_iter=20,
    )
    best_pipeline = search.best_estimator_

    y_pred = best_pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    print("\nBest parameters:")
    print(search.best_params_)
    print(f"Best CV f1: {search.best_score_:.4f}")
    print(f"Test f1: {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    shap_outputs = compute_and_visualize_shap(
        best_pipeline=best_pipeline,
        X_test=X_test,
        output_dir="model_outputs",
    )
    print("\nSHAP outputs:")
    print(f" - Global summary plot: {shap_outputs['global_plot']}")
    print(f" - High-risk customer plot: {shap_outputs['local_plot']}")
    print(
        " - High-risk customer:",
        shap_outputs["high_risk_customer_id"],
        f"(p_churn={shap_outputs['high_risk_probability']:.3f})",
    )


if __name__ == "__main__":
    main()
