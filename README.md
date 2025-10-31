Momentum Radar Prototype
========================

Overview
--------
This repository currently contains the seed for a "Momentum Radar" monitoring system. The goal is to surface capital flows across geographic and sector hierarchies and highlight downstream laggards that may benefit from momentum spillovers.

Folder Structure
----------------

```
/workspace
??? main.py
```

- `main.py`: placeholder entry point. It will evolve into the orchestrator that triggers ingestion, processing, and visualization tasks as the pipeline is implemented.

Planned Data Flow
-----------------
1. **Ingestion**: Pull OHLCV data for global equities and ETFs via APIs such as Yahoo Finance (`yfinance`) or Alpha Vantage. Normalize symbols and enrich them with a continent ? region ? country ? sector ? industry ? company hierarchy.
2. **Processing**: Adjust prices for corporate actions, compute multi-horizon log returns (1Y, 6M, 3M, 1M, 1W, 1D), and derive z-score momentum metrics within each horizon. Persist cleaned datasets as Parquet/DuckDB tables.
3. **Analytics**: Roll momentum scores up the hierarchy to highlight hot spots, measure breadth, and detect lead/lag patterns along value chains.
4. **Visualization**: Produce heatmaps or dashboards that allow drill-down navigation and surface lagging candidates for further review.

Example Usage
-------------
At this stage the project is a placeholder. Run the entry point to confirm the environment is wired correctly:

```
python3 main.py
```

Future iterations will replace `main.py` with a CLI (e.g., `python -m momentum_radar ingest`) or workflow scheduler integration.

Next Steps Toward an Interactive Dashboard
-----------------------------------------
1. **Scaffold Package**: Create `src/momentum_radar` with modules for ingestion, processing, analytics, and visualization. Convert the conceptual code snippets into concrete Python files with tests.
2. **Data Layer**: Implement symbol metadata loaders, historical price fetchers, and persistence (Parquet/DuckDB or PostgreSQL). Add retry logic, logging, and data quality checks.
3. **Computation Layer**: Build momentum calculation and aggregation functions, parameterize horizons, and design value-chain relationships for lag detection.
4. **Streamlit App**: Develop a `streamlit_app.py` that loads the latest momentum tables, renders Plotly heatmaps/treemaps, and enables continent ? company drill-down plus lagging candidate watchlists. Use `st.cache_data` to control refresh intervals and schedule background updates via Prefect/APSheduler.
5. **Deployment & Ops**: Containerize the service, manage secrets via environment variables or Vault, and instrument monitoring for API quota usage and data latency.

Refer to the project plan in the docstring comments for implementation guidance.
