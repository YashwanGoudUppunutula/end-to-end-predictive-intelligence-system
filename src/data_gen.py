from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DataConfig, ingest_training_data


def main() -> None:
    cfg = DataConfig(n_customers=2500, seed=7, reference_date="2024-12-31")
    abt_df = ingest_training_data(cfg)
    out_path = Path("data/abt.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    abt_df.to_csv(out_path, index=False)
    print(f"Saved ABT to: {out_path}")
    print("Shape:", abt_df.shape)


if __name__ == "__main__":
    main()
