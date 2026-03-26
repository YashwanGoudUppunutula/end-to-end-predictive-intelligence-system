from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from generate_messy_data import SimulationConfig, simulate_messy_data


@dataclass
class DataConfig:
    n_customers: int = 2500
    seed: int = 7
    reference_date: str = "2024-12-31"


def ingest_training_data(config: DataConfig | None = None) -> pd.DataFrame:
    """
    Ingest data for model development.

    In production this function can be swapped to database ingestion.
    For now, it generates a realistic ABT from synthetic relational sources.
    """
    cfg = config or DataConfig()
    sim_cfg = SimulationConfig(
        n_customers=cfg.n_customers,
        seed=cfg.seed,
        reference_date=cfg.reference_date,
    )
    _, _, _, abt_df = simulate_messy_data(sim_cfg)
    return abt_df


def load_abt_csv(path: str | Path) -> pd.DataFrame:
    """Optional helper to load a persisted ABT from disk."""
    df = pd.read_csv(path)
    date_cols = [
        "signup_date",
        "last_transaction_date",
        "last_login_feature_date",
        "last_login_date",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
