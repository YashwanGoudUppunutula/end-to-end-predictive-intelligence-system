from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from simulate_messy_data import SimulationConfig, simulate_messy_data


@dataclass
class PipelineArtifacts:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocess_pipeline: Pipeline
    X_train_processed: pd.DataFrame
    X_test_processed: pd.DataFrame


class DateFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Create model-friendly recency features from datetime columns.

    This transformer is deterministic and does not inspect target values.
    """

    def __init__(self, reference_date: str = "2024-12-31") -> None:
        self.reference_date = pd.Timestamp(reference_date)
        self.date_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DateFeatureEngineer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DateFeatureEngineer expects a pandas DataFrame as input.")
        self.date_columns_ = [
            col
            for col in [
                "signup_date",
                "last_transaction_date",
                "last_login_feature_date",
                "last_login_date",
            ]
            if col in X.columns
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        for col in self.date_columns_:
            date_series = pd.to_datetime(X_out[col], errors="coerce")
            X_out[f"{col}_days_from_ref"] = (self.reference_date - date_series).dt.days

        # Keep ID out of the model matrix while retaining it in the original ABT.
        drop_cols = self.date_columns_ + [col for col in ["customer_id"] if col in X_out.columns]
        X_out = X_out.drop(columns=drop_cols, errors="ignore")
        return X_out


def suggest_eda_visualizations(abt_df: pd.DataFrame, output_dir: str = "eda_outputs") -> list[str]:
    """
    Generate three high-signal EDA plots for churn understanding.

    1) Churn rate by region
    2) Numeric distribution by churn (example: login_count)
    3) Correlation heatmap across numeric features + target
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_paths: list[str] = []

    sns.set_theme(style="whitegrid")

    # 1) Churn rate by region
    region_churn = (
        abt_df.groupby("region", as_index=False)["churned"].mean().sort_values("churned", ascending=False)
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=region_churn, x="region", y="churned", hue="region", dodge=False, legend=False)
    plt.title("Churn Rate by Region")
    plt.ylabel("Mean churn probability")
    plt.xlabel("Region")
    path_1 = str(Path(output_dir) / "churn_rate_by_region.png")
    plt.tight_layout()
    plt.savefig(path_1, dpi=140)
    plt.close()
    plot_paths.append(path_1)

    # 2) Distribution of login_count by churn target
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=abt_df, x="churned", y="login_count")
    plt.title("Login Count Distribution by Churn")
    plt.xlabel("Churned (0=no, 1=yes)")
    plt.ylabel("Login count")
    path_2 = str(Path(output_dir) / "login_count_by_churn.png")
    plt.tight_layout()
    plt.savefig(path_2, dpi=140)
    plt.close()
    plot_paths.append(path_2)

    # 3) Correlation heatmap for numeric features + target
    numeric_cols = abt_df.select_dtypes(include=["number"]).columns.tolist()
    keep_cols = [col for col in numeric_cols if col != "customer_id"]
    corr = abt_df[keep_cols].corr(numeric_only=True)
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3)
    plt.title("Correlation Heatmap (Numeric Features + churned)")
    path_3 = str(Path(output_dir) / "correlation_heatmap_numeric.png")
    plt.tight_layout()
    plt.savefig(path_3, dpi=140)
    plt.close()
    plot_paths.append(path_3)

    return plot_paths


def build_preprocessing_pipeline(reference_date: str = "2024-12-31") -> Pipeline:
    """
    Build leakage-safe preprocessing pipeline.

    - DateFeatureEngineer creates recency features from datetime columns.
    - Numeric branch does median imputation.
    - Categorical branch does most-frequent imputation + OneHotEncoder.
    """
    numeric_features = [
        "age",
        "txn_count",
        "total_amount",
        "avg_amount",
        "max_amount",
        "min_amount",
        "days_since_last_transaction",
        "login_count",
        "total_seconds_active",
        "avg_seconds_active",
        "max_seconds_active",
        "days_since_last_login_feature",
        "days_since_last_login",
        "customer_tenure_days",
        "signup_date_days_from_ref",
        "last_transaction_date_days_from_ref",
        "last_login_feature_date_days_from_ref",
        "last_login_date_days_from_ref",
    ]
    categorical_features = ["region"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    col_transform = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    full_pipeline = Pipeline(
        steps=[
            ("date_engineering", DateFeatureEngineer(reference_date=reference_date)),
            ("preprocess", col_transform),
        ]
    )
    return full_pipeline


def split_and_apply_pipeline(
    abt_df: pd.DataFrame,
    target_col: str = "churned",
    test_size: float = 0.2,
    random_state: int = 42,
    reference_date: str = "2024-12-31",
) -> PipelineArtifacts:
    """Split train/test and fit transform pipeline without leakage."""
    if target_col not in abt_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in ABT.")

    X = abt_df.drop(columns=[target_col])
    y = abt_df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocess_pipeline = build_preprocessing_pipeline(reference_date=reference_date)
    X_train_processed = preprocess_pipeline.fit_transform(X_train, y_train)
    X_test_processed = preprocess_pipeline.transform(X_test)

    return PipelineArtifacts(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocess_pipeline=preprocess_pipeline,
        X_train_processed=X_train_processed,
        X_test_processed=X_test_processed,
    )


def main() -> None:
    cfg = SimulationConfig(n_customers=1500, seed=7, reference_date="2024-12-31")
    _, _, _, abt_df = simulate_messy_data(cfg)

    plot_files = suggest_eda_visualizations(abt_df, output_dir="eda_outputs")
    artifacts = split_and_apply_pipeline(
        abt_df=abt_df,
        target_col="churned",
        test_size=0.2,
        random_state=42,
        reference_date=cfg.reference_date,
    )

    print("Saved EDA plots:")
    for plot in plot_files:
        print(f" - {plot}")

    print("\nTrain/test split:")
    print("X_train shape:", artifacts.X_train.shape)
    print("X_test shape:", artifacts.X_test.shape)
    print("y_train mean (churn rate):", round(float(artifacts.y_train.mean()), 4))
    print("y_test mean (churn rate):", round(float(artifacts.y_test.mean()), 4))

    print("\nProcessed matrices:")
    print("X_train_processed shape:", artifacts.X_train_processed.shape)
    print("X_test_processed shape:", artifacts.X_test_processed.shape)


if __name__ == "__main__":
    main()
