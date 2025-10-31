"""Pipeline to generate sample momentum outputs for the Streamlit app.

The current implementation fabricates price data for the symbols defined in
`config/hierarchy_map.csv`. It calculates multi-horizon log returns, derives
simple momentum z-scores, aggregates by continent, and stores two parquet
datasets under `data/processed/`:

- `momentum_scores.parquet`
- `top_movers.parquet`

These artifacts power the interactive dashboard in `streamlit_app.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


HORIZON_MAP: Dict[int, str] = {
    1: "1D",
    5: "1W",
    21: "1M",
    63: "3M",
    126: "6M",
    252: "1Y",
}

DATA_DIR = Path("data/processed")
HIERARCHY_PATH = Path("config/hierarchy_map.csv")


@dataclass
class PriceHistoryConfig:
    start_price: float = 100.0
    business_days: int = 320


def ensure_output_dir(directory: Path) -> None:
    """Create the output directory if it does not exist."""

    directory.mkdir(parents=True, exist_ok=True)


def load_hierarchy(path: Path) -> pd.DataFrame:
    """Load the hierarchy mapping; create a minimal template when absent."""

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        sample = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "NVDA", "TSM", "BP"],
                "continent": [
                    "North America",
                    "North America",
                    "North America",
                    "Asia",
                    "Europe",
                ],
                "country": ["US", "US", "US", "TW", "GB"],
                "sector": [
                    "Technology",
                    "Technology",
                    "Technology",
                    "Technology",
                    "Energy",
                ],
            }
        )
        sample.to_csv(path, index=False)

    hierarchy = pd.read_csv(path)
    if "continent" not in hierarchy.columns:
        raise ValueError("`config/hierarchy_map.csv` must include a 'continent' column.")

    if "country" not in hierarchy.columns:
        hierarchy["country"] = ""
    if "sector" not in hierarchy.columns:
        hierarchy["sector"] = "Multi"

    hierarchy["country"] = hierarchy["country"].fillna("")
    hierarchy["sector"] = hierarchy["sector"].fillna("Multi")

    return hierarchy


def generate_price_history(symbols: Iterable[str], config: PriceHistoryConfig) -> pd.DataFrame:
    """Fabricate simple geometric random-walk price histories."""

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=config.business_days)
    rng = np.random.default_rng(seed=12345)
    records: List[Dict[str, float]] = []

    for symbol in symbols:
        drift = rng.normal(0.0005, 0.0002)
        vol = rng.uniform(0.01, 0.03)
        shocks = rng.normal(drift, vol, size=len(dates))
        log_prices = np.log(config.start_price) + np.cumsum(shocks)
        prices = np.exp(log_prices)

        for date, price in zip(dates, prices):
            records.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "adj_close": float(price),
                }
            )

    prices_df = pd.DataFrame(records)
    prices_df.sort_values(["symbol", "timestamp"], inplace=True)
    prices_df.reset_index(drop=True, inplace=True)
    return prices_df


def compute_log_returns(prices: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    """Calculate log returns for each horizon."""

    working = prices.copy()
    working["log_price"] = np.log(working["adj_close"])

    for horizon in horizons:
        col = f"log_return_{horizon}"
        working[col] = (
            working.groupby("symbol")["log_price"].transform(lambda s: s - s.shift(horizon))
        )

    return working


def prepare_latest_snapshot(prices_with_returns: pd.DataFrame) -> pd.DataFrame:
    """Extract the latest row per symbol with computed log returns."""

    latest = (
        prices_with_returns.sort_values(["symbol", "timestamp"])
        .groupby("symbol")
        .tail(1)
        .reset_index(drop=True)
    )
    return latest


def build_top_movers(
    latest: pd.DataFrame,
    hierarchy: pd.DataFrame,
) -> pd.DataFrame:
    """Generate the top movers table across all horizons."""

    movers_frames: List[pd.DataFrame] = []

    merged = latest.merge(hierarchy, on="symbol", how="left")
    merged["country"] = merged["country"].fillna("")
    merged["sector"] = merged["sector"].fillna("Multi")

    for horizon, label in HORIZON_MAP.items():
        col = f"log_return_{horizon}"
        if col not in merged.columns:
            continue

        horizon_df = merged[["symbol", "adj_close", "continent", "country", "sector", col]].copy()
        horizon_df.dropna(subset=[col], inplace=True)

        if horizon_df.empty:
            continue

        mean = horizon_df[col].mean()
        std = horizon_df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            z_scores = np.zeros(len(horizon_df))
        else:
            z_scores = (horizon_df[col] - mean) / std

        horizon_df["momentum_z"] = z_scores
        horizon_df["return_pct"] = np.expm1(horizon_df[col])
        horizon_df["horizon_label"] = label
        horizon_df["horizon_days"] = horizon
        horizon_df["name"] = horizon_df["symbol"]
        horizon_df["continent"] = horizon_df["continent"].fillna("Unmapped")

        horizon_df.rename(columns={"adj_close": "last_price"}, inplace=True)

        horizon_df = horizon_df[
            [
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
        ]

        horizon_df.sort_values(by="momentum_z", ascending=False, inplace=True)
        movers_frames.append(horizon_df.head(50))

    if not movers_frames:
        return pd.DataFrame(
            columns=[
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
        )

    return pd.concat(movers_frames, ignore_index=True)


def build_momentum_scores(
    top_movers: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate momentum scores at the continent level."""

    if top_movers.empty:
        return pd.DataFrame(
            columns=[
                "entity_level",
                "entity_name",
                "continent",
                "horizon_label",
                "horizon_days",
                "avg_momentum",
                "median_momentum",
                "num_constituents",
            ]
        )

    agg = (
        top_movers.groupby(["continent", "horizon_label"], dropna=False)
        .agg(
            avg_momentum=("momentum_z", "mean"),
            median_momentum=("momentum_z", "median"),
            num_constituents=("symbol", "nunique"),
        )
        .reset_index()
    )

    horizon_lookup = {label: days for days, label in HORIZON_MAP.items()}
    agg["horizon_days"] = agg["horizon_label"].map(horizon_lookup)
    agg["entity_level"] = "continent"
    agg.rename(columns={"continent": "entity_name"}, inplace=True)
    agg["continent"] = agg["entity_name"]

    columns = [
        "entity_level",
        "entity_name",
        "continent",
        "horizon_label",
        "horizon_days",
        "avg_momentum",
        "median_momentum",
        "num_constituents",
    ]

    agg = agg[columns]
    return agg


def persist_outputs(momentum_scores: pd.DataFrame, top_movers: pd.DataFrame) -> None:
    """Write dataframes to parquet files."""

    ensure_output_dir(DATA_DIR)
    momentum_path = DATA_DIR / "momentum_scores.parquet"
    top_movers_path = DATA_DIR / "top_movers.parquet"

    momentum_scores.to_parquet(momentum_path, index=False)
    top_movers.to_parquet(top_movers_path, index=False)

    print(f"Momentum scores written to {momentum_path}")
    print(f"Top movers written to {top_movers_path}")


def run_pipeline() -> None:
    hierarchy = load_hierarchy(HIERARCHY_PATH)
    symbols = hierarchy["symbol"].dropna().unique().tolist()

    prices = generate_price_history(symbols, PriceHistoryConfig())
    with_returns = compute_log_returns(prices, HORIZON_MAP.keys())
    latest = prepare_latest_snapshot(with_returns)

    top_movers = build_top_movers(latest, hierarchy)
    momentum_scores = build_momentum_scores(top_movers)

    persist_outputs(momentum_scores, top_movers)


if __name__ == "__main__":
    run_pipeline()

