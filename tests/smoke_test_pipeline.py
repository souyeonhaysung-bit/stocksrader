"""Smoke test to verify run_pipeline outputs parquet files with expected schema."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from run_pipeline import run_pipeline


EXPECTED_MOMENTUM_COLUMNS = [
    "entity_level",
    "entity_name",
    "continent",
    "horizon_label",
    "horizon_days",
    "avg_momentum",
    "median_momentum",
    "num_constituents",
]

EXPECTED_MOVERS_COLUMNS = [
    "symbol",
    "name",
    "continent",
    "country",
    "sector",
    "horizon_label",
    "momentum_z",
    "return_pct",
    "last_price",
]

DATA_DIR = Path("data/processed")


def assert_columns(df: pd.DataFrame, expected: list[str], label: str) -> None:
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise AssertionError(f"{label} missing expected columns: {missing}")


def test_pipeline_outputs() -> None:
    run_pipeline()

    momentum_path = DATA_DIR / "momentum_scores.parquet"
    movers_path = DATA_DIR / "top_movers.parquet"

    if not momentum_path.exists():
        raise AssertionError(f"Momentum scores parquet not found: {momentum_path}")
    if not movers_path.exists():
        raise AssertionError(f"Top movers parquet not found: {movers_path}")

    momentum_df = pd.read_parquet(momentum_path)
    movers_df = pd.read_parquet(movers_path)

    assert_columns(momentum_df, EXPECTED_MOMENTUM_COLUMNS, "momentum_scores.parquet")
    assert_columns(movers_df, EXPECTED_MOVERS_COLUMNS, "top_movers.parquet")

    print("Smoke test passed: pipeline outputs present with expected schema.")


if __name__ == "__main__":
    test_pipeline_outputs()
