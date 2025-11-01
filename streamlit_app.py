"""Streamlit dashboard for the Momentum Radar prototype.

The app expects `run_pipeline.py` to materialize two parquet datasets:

1. `data/processed/momentum_scores.parquet`
   - Columns: [`entity_level`, `entity_name`, `continent`, `horizon_label`,
     `horizon_days`, `avg_momentum`, `median_momentum`, `num_constituents`]

2. `data/processed/top_movers.parquet`
   - Columns: [`symbol`, `name`, `continent`, `country`, `sector`,
     `horizon_label`, `momentum_z`, `return_pct`, `last_price`]

When these files are missing, the app generates sample data so that the UI is still
usable during development.

Usage
-----
1. Execute `run_pipeline.py` to refresh the parquet files.
2. Launch the Streamlit app:

   ```bash
   streamlit run streamlit_app.py
   ```

3. Use the sidebar filters (horizon, continent, entity level) to explore
   aggregated momentum heatmaps and the strongest movers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


DATA_DIR = Path("data/processed")
MOMENTUM_PATH = DATA_DIR / "momentum_scores.parquet"
TOP_MOVERS_PATH = DATA_DIR / "top_movers.parquet"


@st.cache_data(show_spinner=False, ttl=300)
def load_parquet(path: Path) -> pd.DataFrame | None:
    """Load a parquet file if present; return *None* when missing."""

    if not path.exists():
        return None
    return pd.read_parquet(path)


def make_sample_momentum() -> pd.DataFrame:
    """Create a synthetic momentum table for local prototyping."""

    continents = ["North America", "Europe", "Asia"]
    sectors = ["Technology", "Financials", "Industrials", "Energy"]
    horizons = [("1D", 1), ("1W", 5), ("1M", 21), ("3M", 63)]

    rows = []
    rng = np.random.default_rng(42)

    for continent in continents:
        for sector in sectors:
            for label, days in horizons:
                rows.append(
                    {
                        "entity_level": "sector",
                        "entity_name": sector,
                        "continent": continent,
                        "horizon_label": label,
                        "horizon_days": days,
                        "avg_momentum": rng.normal(loc=0.5, scale=0.5),
                        "median_momentum": rng.normal(loc=0.4, scale=0.4),
                        "num_constituents": rng.integers(20, 120),
                    }
                )
    return pd.DataFrame(rows)


def make_sample_top_movers() -> pd.DataFrame:
    """Create a synthetic movers table for local prototyping."""

    symbols = ["AAPL", "MSFT", "NVDA", "HSBC", "TSM", "BP", "SHOP", "ADBE"]
    continents = ["North America", "Europe", "Asia"]
    sectors = ["Technology", "Financials", "Energy"]
    horizons = ["1D", "1W", "1M", "3M"]
    rng = np.random.default_rng(7)

    rows = []
    for symbol in symbols:
        rows.append(
            {
                "symbol": symbol,
                "name": f"{symbol} Inc.",
                "continent": rng.choice(continents),
                "country": rng.choice(["US", "CA", "GB", "DE", "JP"]),
                "sector": rng.choice(sectors),
                "horizon_label": rng.choice(horizons),
                "momentum_z": rng.normal(loc=1.0, scale=0.5),
                "return_pct": rng.normal(loc=5.0, scale=3.0),
                "last_price": rng.uniform(20, 400),
            }
        )
    return pd.DataFrame(rows)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Load momentum and top-movers data, falling back to synthetic samples."""

    momentum = load_parquet(MOMENTUM_PATH)
    movers = load_parquet(TOP_MOVERS_PATH)

    using_samples = False
    if momentum is None or momentum.empty:
        momentum = make_sample_momentum()
        using_samples = True
    if movers is None or movers.empty:
        movers = make_sample_top_movers()
        using_samples = True

    return momentum, movers, using_samples


def build_filters(momentum: pd.DataFrame, movers: pd.DataFrame) -> Tuple[str, str, str]:
    """Render sidebar filters and return selected horizon, continent, and level."""

    horizons = sorted(momentum["horizon_label"].unique(), key=lambda x: momentum[momentum["horizon_label"] == x]["horizon_days"].iloc[0])
    continents = ["All"] + sorted(momentum["continent"].dropna().unique())
    levels = sorted(momentum["entity_level"].dropna().unique())

    st.sidebar.header("Filters")
    selected_horizon = st.sidebar.selectbox("Horizon", horizons, index=max(0, horizons.index("1M")) if "1M" in horizons else 0)
    selected_continent = st.sidebar.selectbox("Continent", continents, index=0)
    selected_level = st.sidebar.selectbox("Entity Level", levels, index=0)

    if st.sidebar.button("Refresh Data", type="primary"):
        load_parquet.clear()
        st.experimental_rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Data provided by `run_pipeline.py`. Re-run the pipeline to refresh the parquet files before launching the dashboard."
    )

    return selected_horizon, selected_continent, selected_level


def render_heatmap(momentum: pd.DataFrame, horizon: str, continent: str, entity_level: str) -> None:
    """Display an average momentum heatmap across entities."""

    filtered = momentum[momentum["horizon_label"] == horizon]
    if continent != "All":
        filtered = filtered[filtered["continent"] == continent]

    filtered = filtered[filtered["entity_level"] == entity_level]

    if filtered.empty:
        st.info("No momentum data available for the selected filters.")
        return

    pivot = (
        filtered
        .pivot_table(
            index="entity_name",
            columns="continent",
            values="avg_momentum",
            aggfunc="mean",
        )
        .sort_index()
    )

    fig = px.imshow(
        pivot,
        labels=dict(color="Avg Momentum"),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(
        title=f"Average Momentum Heatmap ? {entity_level.title()} ? {horizon}",
        xaxis_title="Continent",
        yaxis_title=entity_level.title(),
        margin=dict(l=60, r=20, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_top_movers(movers: pd.DataFrame, horizon: str, continent: str) -> None:
    """Display a table of the top momentum movers."""

    filtered = movers[movers["horizon_label"] == horizon].copy()
    if continent != "All":
        filtered = filtered[filtered["continent"] == continent]

    if filtered.empty:
        st.info("No movers detected for the selected filters.")
        return

    filtered = filtered.sort_values(by="momentum_z", ascending=False).head(25)
    filtered["return_pct"] = filtered["return_pct"].round(2)
    filtered["momentum_z"] = filtered["momentum_z"].round(2)
    filtered["last_price"] = filtered["last_price"].round(2)

    st.dataframe(
        filtered[
            [
                "symbol",
                "name",
                "continent",
                "country",
                "sector",
                "momentum_z",
                "return_pct",
                "last_price",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title="Momentum Radar", layout="wide")
    st.title("Momentum Radar Dashboard")
    st.caption(
        "Interactive view of multi-horizon momentum, driven by the latest run of `run_pipeline.py`."
    )

    momentum_df, movers_df, using_samples = load_data()
    horizon, continent, level = build_filters(momentum_df, movers_df)

    if using_samples:
        st.warning(
            "Sample data in use. Execute `run_pipeline.py` to refresh real momentum outputs in "
            f"{MOMENTUM_PATH} and {TOP_MOVERS_PATH}."
        )

    heatmap_tab, movers_tab = st.tabs(["Heatmap", "Top Movers"])

    with heatmap_tab:
        render_heatmap(momentum_df, horizon, continent, level)

    with movers_tab:
        render_top_movers(movers_df, horizon, continent)


if __name__ == "__main__":
    # Running via `python streamlit_app.py` will simply launch Streamlit's CLI message.
    # The recommended entry point is `streamlit run streamlit_app.py`.
    main()

