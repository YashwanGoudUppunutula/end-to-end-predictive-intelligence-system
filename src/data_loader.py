from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_and_merge_customer_sources(
    customer_info_path: str | Path,
    transaction_logs_path: str | Path,
) -> pd.DataFrame:
    """
    Load separate customer and transaction CSV files and build customer-level ABT.

    Expected columns:
    - customer_info.csv: customer_id, signup_date, region, age
    - transaction_logs.csv: customer_id, transaction_date, amount
    """
    customers = pd.read_csv(customer_info_path)
    tx = pd.read_csv(transaction_logs_path)

    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"], errors="coerce")

    tx_agg = tx.groupby("customer_id", as_index=False).agg(
        txn_count=("amount", "size"),
        total_amount=("amount", "sum"),
        avg_transaction_value=("amount", "mean"),
        last_transaction_date=("transaction_date", "max"),
    )

    abt = pd.merge(customers, tx_agg, on="customer_id", how="left")
    return abt
