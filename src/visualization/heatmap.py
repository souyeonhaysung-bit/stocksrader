import pandas as pd
import matplotlib.pyplot as plt

def heatmap(df: pd.DataFrame, x: str, y: str, v: str, title: str = ""):
    pivot = df.pivot(index=y, columns=x, values=v)
    plt.figure(figsize=(10,5))
    im = plt.imshow(pivot, aspect="auto")
    plt.colorbar(im, label=v)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title(title or f"{v} heatmap")
    plt.tight_layout()
    plt.show()
