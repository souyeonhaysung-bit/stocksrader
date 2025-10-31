import pandas as pd

def rank_level(momentum_df: pd.DataFrame, hierarchy_df: pd.DataFrame,
               level: str = "continent", horizon_days: int = 63, top_n: int = 10):
    cur = momentum_df[momentum_df["horizon_days"] == horizon_days]
    merged = cur.merge(hierarchy_df, on="symbol", how="left")
    agg = (merged.groupby(level)["momentum_z"]
           .agg(avg_momentum="mean", median_momentum="median", breadth="count")
           .reset_index()
           .sort_values("avg_momentum", ascending=False))
    return agg.head(top_n)
