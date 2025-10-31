import pandas as pd
from src.ingestion.fetcher import fetch_history
from src.processing.momentum import compute_returns, zscore_momentum
from src.processing.ranking import rank_level
from src.visualization.heatmap import heatmap

UNIVERSE = ["SPY","EWY","EPI","EWJ","EWO"]  # 샘플

if __name__ == "__main__":
    prices = fetch_history(UNIVERSE, lookback_days=800)
    if prices.empty:
        raise SystemExit("No data fetched. Check tickers or network policy.")

    rets = compute_returns(prices)
    mom = zscore_momentum(rets, window=126)

    hier = pd.read_csv("config/hierarchy_map.csv")
    top_continents = rank_level(mom, hier, level="continent", horizon_days=63, top_n=10)
    print("\nTop continents (63d):\n", top_continents)

    # 간단 heatmap: horizon x continent 평균 모멘텀
    grid = (mom.merge(hier, on="symbol")
               .groupby(["horizon_days","continent"])["momentum_z"].mean().reset_index())
    heatmap(grid, x="continent", y="horizon_days", v="momentum_z",
            title="Avg Momentum by Continent & Horizon")
