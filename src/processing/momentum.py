import pandas as pd
import numpy as np

def compute_returns(prices: pd.DataFrame, horizons=(252,126,63,21,5,1)):
    prices = prices.sort_values(["symbol","timestamp"]).copy()
    out = []
    for h in horizons:
        col = f"ret_{h}d"
        prices[col] = (
            np.log(prices.groupby("symbol")["adj_close"].shift(0))
            - np.log(prices.groupby("symbol")["adj_close"].shift(h))
        )
        out.append(col)
    melt = prices[["timestamp","symbol"]+out].melt(
        id_vars=["timestamp","symbol"], var_name="horizon", value_name="log_return"
    )
    melt["horizon_days"] = melt["horizon"].str.extract(r"ret_(\d+)d").astype(int)
    melt = melt.dropna(subset=["log_return"])
    return melt

def zscore_momentum(returns: pd.DataFrame, window=126):
    df = returns.sort_values(["symbol","timestamp"]).copy()
    df["roll_mean"] = (
        df.groupby("symbol")["log_return"].rolling(window).mean().reset_index(level=0, drop=True)
    )
    df["roll_std"] = (
        df.groupby("symbol")["log_return"].rolling(window).std().reset_index(level=0, drop=True)
    )
    df["momentum_z"] = (df["log_return"] - df["roll_mean"]) / df["roll_std"]
    return df.dropna(subset=["momentum_z"])
