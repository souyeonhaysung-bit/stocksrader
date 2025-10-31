from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

def fetch_history(symbols, lookback_days=420):
    start = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    frames = []
    for s in symbols:
        t = yf.Ticker(s)
        df = t.history(start=start, auto_adjust=True)  # adj close 포함
        if df.empty:
            continue
        df = df.reset_index().rename(columns={"Date":"timestamp","Close":"adj_close"})
        df["symbol"] = s
        frames.append(df[["timestamp","symbol","adj_close"]])
    if not frames:
        return pd.DataFrame(columns=["timestamp","symbol","adj_close"])
    return pd.concat(frames, ignore_index=True)
