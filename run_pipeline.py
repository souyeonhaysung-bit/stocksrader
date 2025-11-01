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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")


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
USE_YFINANCE = os.getenv("USE_YFINANCE", "false").lower() in {"1", "true", "yes"}


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


def fetch_yfinance_history(symbols: Iterable[str]) -> pd.DataFrame:
    """Retrieve two years of daily adjusted close data via yfinance."""

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dependency guarantee
        raise RuntimeError("yfinance is required when USE_YFINANCE is True") from exc

    records: List[Dict[str, float]] = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2y", interval="1d", auto_adjust=False)
        except Exception as exc:  # pragma: no cover - network variability
            print(f"[WARN] Failed to download {symbol}: {exc}")
            continue

        if df.empty or "Adj Close" not in df.columns:
            print(f"[WARN] No adjusted close data for {symbol}; skipping.")
            continue

        df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df.rename(columns={date_col: "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df["adj_close"] = df["Adj Close"].astype(float)
        df = df[["timestamp", "adj_close"]]
        df["symbol"] = symbol
        records.extend(df.to_dict(orient="records"))

    if not records:
        print("[WARN] yfinance download returned no data; falling back to simulator.")
        return pd.DataFrame()

    prices_df = pd.DataFrame(records)
    if "adj_close" not in prices_df.columns:
        print("[WARN] Adjusted close column missing from yfinance output; falling back to simulator.")
        return pd.DataFrame()

    prices_df.dropna(subset=["adj_close"], inplace=True)

    if prices_df.empty:
        print("[WARN] yfinance provided only NaN adjusted closes; falling back to simulator.")
        return pd.DataFrame()

    prices_df["timestamp"] = pd.to_datetime(prices_df["timestamp"])
    prices_df.sort_values(["symbol", "timestamp"], inplace=True)
    prices_df.reset_index(drop=True, inplace=True)
    return prices_df


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


def build_momentum_scores(top_movers: pd.DataFrame) -> pd.DataFrame:
    """Aggregate momentum scores at continent, country, and sector levels."""

    base_columns = [
        "entity_level",
        "entity_name",
        "continent",
        "horizon_label",
        "horizon_days",
        "avg_momentum",
        "median_momentum",
        "num_constituents",
    ]

    if top_movers.empty:
        return pd.DataFrame(columns=base_columns)

    top_movers = top_movers.copy()
    top_movers["continent"] = top_movers["continent"].fillna("Unmapped")
    top_movers["country"] = top_movers["country"].fillna("")
    top_movers["sector"] = top_movers["sector"].fillna("Unspecified")

    horizon_lookup = {label: days for days, label in HORIZON_MAP.items()}

    level_configs = {
        "continent": {
            "group_cols": ["continent"],
            "entity_col": "continent",
        },
        "country": {
            "group_cols": ["continent", "country"],
            "entity_col": "country",
        },
        "sector": {
            "group_cols": ["continent", "sector"],
            "entity_col": "sector",
        },
    }

    frames: List[pd.DataFrame] = []

    for level, config in level_configs.items():
        entity_col = config["entity_col"]

        level_df = top_movers[top_movers[entity_col] != ""].copy()
        if level_df.empty:
            continue

        grouped = (
            level_df.groupby(config["group_cols"] + ["horizon_label"], dropna=False)
            .agg(
                avg_momentum=("momentum_z", "mean"),
                median_momentum=("momentum_z", "median"),
                num_constituents=("symbol", "nunique"),
            )
            .reset_index()
        )

        grouped["horizon_days"] = grouped["horizon_label"].map(horizon_lookup)
        grouped["entity_level"] = level
        grouped["entity_name"] = grouped[entity_col]

        if level != "continent":
            grouped = grouped.drop(columns=[entity_col])

        grouped = grouped[base_columns]
        frames.append(grouped)

    if not frames:
        return pd.DataFrame(columns=base_columns)

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["entity_level", "horizon_days", "entity_name"], inplace=True)
    return combined.reset_index(drop=True)


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

    prices = pd.DataFrame()
    if USE_YFINANCE:
        prices = fetch_yfinance_history(symbols)

    if prices.empty:
        prices = generate_price_history(symbols, PriceHistoryConfig())

    with_returns = compute_log_returns(prices, HORIZON_MAP.keys())
    latest = prepare_latest_snapshot(with_returns)

    top_movers = build_top_movers(latest, hierarchy)
    momentum_scores = build_momentum_scores(top_movers)

    persist_outputs(momentum_scores, top_movers)


if __name__ == "__main__":
    run_pipeline()

