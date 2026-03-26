from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SimulationConfig:
    n_customers: int = 1000
    seed: int = 42
    start_date: str = "2023-01-01"
    reference_date: str = "2024-12-31"


def _as_timestamp(date_value: str | pd.Timestamp) -> pd.Timestamp:
    """Convert input to pandas Timestamp with validation."""
    ts = pd.Timestamp(date_value)
    if pd.isna(ts):
        raise ValueError(f"Invalid date value: {date_value}")
    return ts


def generate_customers_df(config: SimulationConfig) -> pd.DataFrame:
    """Simulate customer demographics with intentional missingness in age."""
    rng = np.random.default_rng(config.seed)
    start_date = _as_timestamp(config.start_date)
    ref_date = _as_timestamp(config.reference_date)
    day_span = max((ref_date - start_date).days, 1)

    customer_ids = np.arange(1, config.n_customers + 1)
    signup_offsets = rng.integers(0, day_span, size=config.n_customers)
    signup_dates = start_date + pd.to_timedelta(signup_offsets, unit="D")

    regions = rng.choice(
        ["north", "south", "east", "west", "central"],
        size=config.n_customers,
        p=[0.22, 0.2, 0.2, 0.18, 0.2],
    )

    ages = rng.normal(loc=37, scale=11, size=config.n_customers).round().astype(float)
    ages = np.clip(ages, 18, 85)
    age_missing_mask = rng.random(config.n_customers) < 0.11
    ages[age_missing_mask] = np.nan

    customers_df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "signup_date": pd.to_datetime(signup_dates),
            "region": regions,
            "age": ages,
        }
    )
    return customers_df


def generate_transactions_df(
    customers_df: pd.DataFrame,
    reference_date: str | pd.Timestamp,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate transaction logs with skewed amounts and sparse/active users."""
    rng = np.random.default_rng(seed + 101)
    ref_date = _as_timestamp(reference_date)

    records: list[dict] = []
    categories = ["subscription", "addon", "support", "hardware", "discount"]
    category_probs = np.array([0.45, 0.2, 0.12, 0.13, 0.1])

    for row in customers_df.itertuples(index=False):
        customer_id = int(row.customer_id)
        signup_date = _as_timestamp(row.signup_date)
        customer_age_days = max((ref_date - signup_date).days, 1)

        if rng.random() < 0.2:
            txn_count = 0
        else:
            txn_count = int(rng.poisson(lam=6) + rng.integers(1, 5))

        for _ in range(txn_count):
            day_offset = int(rng.integers(0, customer_age_days))
            transaction_date = signup_date + pd.Timedelta(days=day_offset)

            amount = float(rng.lognormal(mean=3.45, sigma=0.9))
            if rng.random() < 0.02:
                amount = -abs(amount * rng.uniform(0.2, 1.0))

            category = str(rng.choice(categories, p=category_probs))
            records.append(
                {
                    "customer_id": customer_id,
                    "transaction_date": transaction_date,
                    "amount": round(amount, 2),
                    "category": category,
                }
            )

    transactions_df = pd.DataFrame(records)
    if transactions_df.empty:
        return pd.DataFrame(columns=["customer_id", "transaction_date", "amount", "category"])
    transactions_df["transaction_date"] = pd.to_datetime(transactions_df["transaction_date"])
    return transactions_df


def generate_logs_df(
    customers_df: pd.DataFrame,
    reference_date: str | pd.Timestamp,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate app usage logs with irregular cadence and inactivity windows."""
    rng = np.random.default_rng(seed + 202)
    ref_date = _as_timestamp(reference_date)

    records: list[dict] = []
    for row in customers_df.itertuples(index=False):
        customer_id = int(row.customer_id)
        signup_date = _as_timestamp(row.signup_date)
        lifetime_days = max((ref_date - signup_date).days, 1)

        engagement_type = rng.choice(
            ["high", "medium", "low", "dormant"],
            p=[0.2, 0.38, 0.3, 0.12],
        )
        if engagement_type == "high":
            login_count = int(rng.poisson(45) + 10)
        elif engagement_type == "medium":
            login_count = int(rng.poisson(20) + 5)
        elif engagement_type == "low":
            login_count = int(rng.poisson(8) + 1)
        else:
            login_count = int(rng.poisson(1))

        for _ in range(login_count):
            if engagement_type in {"low", "dormant"} and rng.random() < 0.35:
                max_boost = min(200, lifetime_days)
                min_boost = min(35, max_boost)
                recency_boost = int(rng.integers(min_boost, max_boost + 1))
                day_offset = max(lifetime_days - recency_boost, 0)
            else:
                day_offset = int(rng.integers(0, lifetime_days))

            login_date = signup_date + pd.Timedelta(days=day_offset)
            seconds_active = int(max(10, rng.lognormal(mean=6.1, sigma=0.8)))
            records.append(
                {
                    "customer_id": customer_id,
                    "login_date": login_date,
                    "seconds_active": seconds_active,
                }
            )

    logs_df = pd.DataFrame(records)
    if logs_df.empty:
        return pd.DataFrame(columns=["customer_id", "login_date", "seconds_active"])
    logs_df["login_date"] = pd.to_datetime(logs_df["login_date"])
    return logs_df


def compute_churn_labels(
    customers_df: pd.DataFrame,
    logs_df: pd.DataFrame,
    reference_date: str | pd.Timestamp,
    inactivity_days: int = 30,
) -> pd.DataFrame:
    """
    Label churn based on inactivity before reference_date.

    Churn logic:
    - churned = 1 if no login in the previous `inactivity_days` days.
    - churned = 0 otherwise.
    """
    ref_date = _as_timestamp(reference_date)
    if inactivity_days <= 0:
        raise ValueError("inactivity_days must be > 0")

    safe_logs = logs_df.copy()
    if not safe_logs.empty:
        safe_logs["login_date"] = pd.to_datetime(safe_logs["login_date"])
        safe_logs = safe_logs[safe_logs["login_date"] <= ref_date]

    last_login = (
        safe_logs.groupby("customer_id", as_index=False)["login_date"]
        .max()
        .rename(columns={"login_date": "last_login_date"})
    )

    label_df = customers_df[["customer_id"]].merge(last_login, on="customer_id", how="left")
    label_df["days_since_last_login"] = (ref_date - label_df["last_login_date"]).dt.days
    label_df["churned"] = (
        label_df["last_login_date"].isna() | (label_df["days_since_last_login"] > inactivity_days)
    ).astype(int)
    return label_df


def _build_transaction_features(
    transactions_df: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    if transactions_df.empty:
        return pd.DataFrame(columns=["customer_id"])

    tx = transactions_df.copy()
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"])
    tx = tx[tx["transaction_date"] <= reference_date]

    agg = tx.groupby("customer_id", as_index=False).agg(
        txn_count=("amount", "size"),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        max_amount=("amount", "max"),
        min_amount=("amount", "min"),
        last_transaction_date=("transaction_date", "max"),
    )
    agg["days_since_last_transaction"] = (
        reference_date - agg["last_transaction_date"]
    ).dt.days

    category_counts = (
        tx.pivot_table(
            index="customer_id",
            columns="category",
            values="amount",
            aggfunc="size",
            fill_value=0,
        )
        .add_prefix("txn_cat_count_")
        .reset_index()
    )

    out = agg.merge(category_counts, on="customer_id", how="left")
    return out


def _build_log_features(logs_df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    if logs_df.empty:
        return pd.DataFrame(columns=["customer_id"])

    lg = logs_df.copy()
    lg["login_date"] = pd.to_datetime(lg["login_date"])
    lg = lg[lg["login_date"] <= reference_date]

    agg = lg.groupby("customer_id", as_index=False).agg(
        login_count=("login_date", "size"),
        total_seconds_active=("seconds_active", "sum"),
        avg_seconds_active=("seconds_active", "mean"),
        max_seconds_active=("seconds_active", "max"),
        last_login_feature_date=("login_date", "max"),
    )
    agg["days_since_last_login_feature"] = (
        reference_date - agg["last_login_feature_date"]
    ).dt.days
    return agg


def build_abt(
    customers_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    logs_df: pd.DataFrame,
    reference_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Build one-row-per-customer analytical base table with engineered features and churn label."""
    ref_date = _as_timestamp(reference_date)

    customers = customers_df.copy()
    required_cols = {"customer_id", "signup_date", "region", "age"}
    missing_cols = required_cols - set(customers.columns)
    if missing_cols:
        raise ValueError(f"customers_df missing required columns: {sorted(missing_cols)}")

    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    if customers["customer_id"].duplicated().any():
        raise ValueError("customers_df must contain unique customer_id values")

    txn_features = _build_transaction_features(transactions_df, ref_date)
    log_features = _build_log_features(logs_df, ref_date)
    churn_labels = compute_churn_labels(customers, logs_df, ref_date)

    abt_df = customers.merge(txn_features, on="customer_id", how="left")
    abt_df = abt_df.merge(log_features, on="customer_id", how="left")
    abt_df = abt_df.merge(churn_labels, on="customer_id", how="left")
    abt_df["customer_tenure_days"] = (ref_date - abt_df["signup_date"]).dt.days.clip(lower=0)

    fill_zero_cols = [
        "txn_count",
        "total_amount",
        "avg_amount",
        "max_amount",
        "min_amount",
        "login_count",
        "total_seconds_active",
        "avg_seconds_active",
        "max_seconds_active",
    ]
    for col in fill_zero_cols:
        if col in abt_df.columns:
            abt_df[col] = abt_df[col].fillna(0)

    category_cols = [col for col in abt_df.columns if col.startswith("txn_cat_count_")]
    for col in category_cols:
        abt_df[col] = abt_df[col].fillna(0)

    for recency_col in [
        "days_since_last_transaction",
        "days_since_last_login_feature",
        "days_since_last_login",
    ]:
        if recency_col in abt_df.columns:
            abt_df[recency_col] = abt_df[recency_col].fillna(abt_df["customer_tenure_days"])

    abt_df["churned"] = abt_df["churned"].fillna(1).astype(int)
    assert abt_df["customer_id"].is_unique, "ABT must be one row per customer"
    return abt_df


def simulate_messy_data(
    config: Optional[SimulationConfig] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience orchestrator for generating all tables plus ABT."""
    config = config or SimulationConfig()
    customers_df = generate_customers_df(config)
    transactions_df = generate_transactions_df(customers_df, config.reference_date, seed=config.seed)
    logs_df = generate_logs_df(customers_df, config.reference_date, seed=config.seed)
    abt_df = build_abt(customers_df, transactions_df, logs_df, config.reference_date)
    return customers_df, transactions_df, logs_df, abt_df


def main() -> None:
    config = SimulationConfig(n_customers=1500, seed=7, reference_date="2024-12-31")
    customers_df, transactions_df, logs_df, abt_df = simulate_messy_data(config)

    print("customers_df shape:", customers_df.shape)
    print("transactions_df shape:", transactions_df.shape)
    print("logs_df shape:", logs_df.shape)
    print("abt_df shape:", abt_df.shape)
    print("\nABT preview:")
    print(abt_df.head(10))
    print("\nABT dtypes:")
    print(abt_df.dtypes)
    print("\nChurn rate:", round(abt_df["churned"].mean(), 4))

    expected_columns = {"customer_id", "signup_date", "region", "age", "churned"}
    missing_abt_cols = expected_columns - set(abt_df.columns)
    assert not missing_abt_cols, f"Missing expected ABT columns: {missing_abt_cols}"
    assert abt_df["customer_id"].is_unique, "customer_id must be unique in ABT"
    assert set(abt_df["churned"].unique()).issubset({0, 1}), "churned must be binary"


if __name__ == "__main__":
    main()
